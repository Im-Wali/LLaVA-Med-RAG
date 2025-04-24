import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from datasets import logging as ds_logging
from PIL import Image
from trl.trainer.utils import DataCollatorForCompletionOnlyLM

import transformers
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    MistralForCausalLM,
    Trainer,
)
from transformers import logging as hf_logging


torch.backends.cudnn.enabled = False


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True, metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


class DataCollatorForSupervised(DataCollatorForCompletionOnlyLM):
    def _create_attention_mask(self, input_length_ls):
        total_length = sum(input_length_ls)
        attention_mask = torch.full((1, 1, total_length, total_length), torch.finfo(torch.bfloat16).min)

        start_idx, end_idx = 0, 0
        for length in input_length_ls:
            end_idx += length
            one_tensor = torch.ones((length, length), dtype=torch.float32)
            mask = torch.tril(one_tensor, diagonal=0).to(dtype=torch.bool)
            attention_mask[0, 0, start_idx:end_idx, start_idx:end_idx][mask] = 0
            start_idx = end_idx

        return attention_mask

    def torch_call(self, features_ls):
        input_ids_ls, labels_ls, position_ids_ls, input_length_ls, pixel_values_ls = [], [], [], [], []
        for features in features_ls:
            batch = super().torch_call([{"input_ids": features["input_ids"]}])
            input_ids, labels = batch.input_ids[0], batch.labels[0]
            length = len(input_ids)

            labels_ls.append(labels)
            input_ids_ls.append(input_ids)
            input_length_ls.append(length)
            position_ids_ls.append(torch.arange(length))

            if features["pixel_values"] is not None:
                pixel_values_ls.append(features["pixel_values"])

        attention_mask = self._create_attention_mask(input_length_ls)

        batch = {
            "labels": torch.concat(labels_ls)[None],
            "input_ids": torch.concat(input_ids_ls)[None],
            "position_ids": torch.concat(position_ids_ls)[None],
            "attention_mask": attention_mask,
        }

        if pixel_values_ls:
            batch["pixel_values"] = torch.stack(pixel_values_ls, dim=0)

        return batch


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")


os.environ["TOKENIZERS_PARALLELISM"] = "0"


def train():
    def preprocess_func(example):
        img_dir = Path("/workspace/LLaVA-Med-RAG/data/mimic_cxr_annotaion_json")
        process_finish_ls = list()
        for row_dataset in list(zip(*[example[key] for key in example])):
            row_dataset = {key: value for key, value in zip(example.keys(), row_dataset)}  # noqa: C416

            image = Image.open(img_dir / row_dataset["image"]).convert("RGB")

            prompt, answer = row_dataset["conversations"]
            prompt, answer = prompt["value"].strip(), answer["value"].strip()

            text = f"<s>A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: \n{prompt} ASSISTANT: {answer}</s>"
            outputs = processor(text=text, images=image, return_tensors="np")

            process_finish_ls.append(
                {
                    "input_ids": outputs.input_ids[0].tolist(),
                    "pixel_values": outputs.pixel_values[0].tolist(),
                    training_args.length_column_name: len(outputs.input_ids[0]),
                }
            )

        return_dict = dict()
        for res in process_finish_ls:
            for key, value in res.items():
                return_dict.setdefault(key, []).append(value)

        return return_dict

    parser = transformers.HfArgumentParser([ModelArguments, DataArguments, TrainingArguments])  # type: ignore
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args: TrainingArguments
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    ds_logging.set_verbosity(log_level)
    hf_logging.set_verbosity(log_level)
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    model = LlavaForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args,
        cache_dir=training_args.cache_dir,
    )
    model.multi_modal_projector.requires_grad_(False)
    model.config.use_cache = False

    # pip install git+https://github.com/jp1924/Liger-Kernel.git@add_llava
    from liger_kernel.transformers import apply_liger_kernel_to_llava

    apply_liger_kernel_to_llava(
        fused_linear_cross_entropy=True,
        swiglu=True,
        rope=True,
        rms_norm=True,
        model=model,
    )

    model = torch.compile(model)

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    dataset: Dataset = load_dataset("json", data_files=data_args.data_path, split="train")
    dataset = dataset.map(
        preprocess_func,
        batched=True,
        num_proc=4,
        batch_size=50,
        remove_columns=dataset.column_names,
        load_from_cache_file=True,
        cache_file_name="/workspace/LLaVA-Med-RAG/.dataset_cache/MIMIC.arrow",
        # features=Features(
        #     {
        #         "input_ids": Sequence(Value("int32")),
        #         "pixel_values": Array3D(shape=(3, 336, 336), dtype="float32"),
        #         "length": Value("int32"),
        #     }
        # ),
    )
    dataset = dataset.with_format("torch")

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        if training_args.bits in [4, 8]:
            from peft.tuners.lora import LoraLayer

            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    if training_args.bf16:
                        module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if "lm_head" in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        if training_args.bf16 and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)

    data_collator = DataCollatorForSupervised(
        tokenizer=processor.tokenizer,
        response_template=[8602, 8048, 12738, 28747],
    )
    print(model)
    trainer = Trainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    train()
