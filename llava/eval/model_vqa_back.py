import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
import math
from transformers import set_seed, logging

import random
import numpy as np

logging.set_verbosity_error()

# 이미지 정규화 함수
def normalize_image(image, max_size=512, pad_square=False):
    """이미지 크기를 조정하고, 필요시 정사각형 패딩을 추가"""
    
    # 1️⃣ 리사이즈 (비율 유지)
    image.thumbnail((max_size, max_size))

    if pad_square:
        # 2️⃣ 정사각형 패딩 추가 (중앙 정렬)
        width, height = image.size
        max_dim = max(width, height)
        new_image = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))  # 검은색 패딩
        new_image.paste(image, ((max_dim - width) // 2, (max_dim - height) // 2))
        return new_image
    
    return image

def eval_model(args):
    set_seed(0)
    disable_torch_init()

    # 모델 로드
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = questions[args.chunk_idx::args.num_chunks]  # 청크 처리
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_path = line.get("image_path", "")
        image_file = line["image"]
        qs = line["text"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
        cur_prompt = qs

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        # 이미지 로드 및 정규화
        image_path_full = os.path.join(args.image_folder, image_path, image_file)
        image = Image.open(image_path_full)
        # image = Image.open(image_path_full).convert("RGB")
        # image = normalize_image(image, max_size=512, pad_square=True)  # 크기 조정 + 정사각형 패딩
        image_tensor = process_images([image], image_processor, model.config)[0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        # `sliding_window` 값이 없으면 기본값 (512) 설정
        if not hasattr(model.config, "sliding_window") or model.config.sliding_window is None:
            model.config.sliding_window = 512  
        
        setting_pythorch_seed()

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # max_new_tokens=1024,
                max_new_tokens=512,
                use_cache=True
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }) + "\n")
        ans_file.flush()
    ans_file.close()

def setting_pythorch_seed(seed=2021):
    deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
