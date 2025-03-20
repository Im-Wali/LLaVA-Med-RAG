from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from llava.model import LlavaMistralForCausalLM

# 기본 모델(Backbone) 로드
base_model_name = "microsoft/llava-med-v1.5-mistral-7b"
print(f"Loading base model: {base_model_name}")

base_model = LlavaMistralForCausalLM.from_pretrained(
    base_model_name, 
    torch_dtype="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True  # 추가된 옵션
)

# LoRA 가중치 로드
lora_model_path = "/workspace/LLaVA-Med-RAG/checkpoints/llava-med-rag-v1.5-7b-mimic-8K"
print(f"Loading LoRA model from: {lora_model_path}")

lora_model = PeftModel.from_pretrained(base_model, lora_model_path)

# LoRA 병합
print("Merging LoRA weights...")
merged_model = lora_model.merge_and_unload()

# 병합된 모델 저장
save_path = "/workspace/LLaVA-Med-RAG/checkpoints/llava-med-rag-v1.5-7b-mimic-8K-merged"
print(f"Saving merged model to: {save_path}")

merged_model.save_pretrained(save_path)

# 토크나이저도 저장
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(save_path)

print("LoRA merge complete!")
