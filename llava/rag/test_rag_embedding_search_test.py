from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from transformers import CLIPProcessor, CLIPModel 
from PIL import Image
import torch
import uuid
# from langchain.chat_models import ChatOpenAI
import os
import json
from transformers import pipeline

# ▶️ 환경설정
COLLECTION_NAME = "mimic_test_dataset_rag"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
base_image_path="/workspace/"

# ▶️ 1. Qdrant 클라이언트 연결

# Qdrant 연결
client = QdrantClient(
    url="https://d4d97f9c-07f1-4931-86f9-cadc235fc91b.us-west-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.tw1O_qcpywHq2wr7uyXSidgDywWpVdMxd5y-kgBJ5RY",
)


# ▶️ 3. CLIP 모델 불러오기
# clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
# clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn") # text 요약



# ▶️ 6. 검색 함수 (이미지 or 텍스트 질의)
def search_with_query(query_input, mode="image", top_k=3):
    if mode == "image":
        image = Image.open(query_input).convert("RGB")
        inputs = clip_processor(images=[image], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            query_vec = clip_model.get_image_features(**inputs)[0].cpu().tolist()
    else:
        inputs = clip_processor(text=[query_input], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            query_vec = clip_model.get_text_features(**inputs)[0].cpu().tolist()

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k
    )
    return results


input_jsonl_path = "/workspace/LLaVA-Med-RAG/data/test/llava_med_instruct_1k_mimic_cxr_train_test_convert_rag.json"
output_txt_path = "/workspace/LLaVA-Med-RAG/data/test/llava_med_instruct_1k_mimic_cxr_train_test_convert_rag.txt"

with open(input_jsonl_path, "r", encoding="utf-8") as f, open(output_txt_path, "w", encoding="utf-8") as out:
    for idx, line in enumerate(f):
        item = json.loads(line.strip())
        image_rel_path = item.get("image")
        if not image_rel_path:
            continue

        full_image_path = os.path.join(base_image_path, image_rel_path)
        if not os.path.exists(full_image_path):
            out.write(f"[{idx+1}] 이미지 파일 없음: {full_image_path}\n")
            continue

        out.write(f"\n[{idx+1}] 이미지 기반 질의: {image_rel_path}\n→ 검색 결과:\n")

        try:
            results = search_with_query(full_image_path, mode="image", top_k=3)
        except Exception as e:
            out.write(f"  ❌ 검색 오류: {e}\n")
            continue

        out.write(f"gpt4_answer : {item.get('gpt4_answer')}\n")
        out.write(f"chexpert_labels_mapped : {item.get('chexpert_labels_mapped')}\n")

        for i, r in enumerate(results):
            out.write(f"  ({i+1})\n")
            for key, value in r.payload.items():
                value_str = str(value)  # 문자열로 변환
                value_str = value_str.replace('/workspace/', '')  # 경로 일부 제거
                out.write(f"     {key}: {value_str}\n")