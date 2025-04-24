from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
# from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import CLIPProcessor, CLIPModel 
from PIL import Image
import torch
import uuid
# from langchain.chat_models import ChatOpenAI
import os
from transformers import pipeline
import json
from tqdm import tqdm


# ▶️ 환경설정
COLLECTION_NAME = "mimic_test_dataset_rag_2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
base_image_path="/workspace/"

# ▶️ 1. Qdrant 클라이언트 연결

# Qdrant 연결
client = QdrantClient(
    url="https://d4d97f9c-07f1-4931-86f9-cadc235fc91b.us-west-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.tw1O_qcpywHq2wr7uyXSidgDywWpVdMxd5y-kgBJ5RY",
)

# ▶️ 2. Collection 생성 (이미 존재하면 무시)
# client.recreate_collection(
#     collection_name=COLLECTION_NAME,
#     vectors_config=VectorParams(size=512, distance=Distance.COSINE),
#     timeout=6000  
# )
if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        timeout=6000
    )
    print(f"✅ '{COLLECTION_NAME}' 컬렉션이 새로 생성되었습니다.")
else:
    print(f"⚠️ '{COLLECTION_NAME}' 컬렉션이 이미 존재합니다. recreate는 생략합니다.")

# ▶️ 3. CLIP 모델 불러오기
# clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
# clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn") # text 요약


# ▶️ 4. 이미지 + 텍스트 임베딩 함수
def embed_image_text_pair(image_path, text):
    image = Image.open(base_image_path + image_path).convert("RGB")
    inputs = clip_processor(text=[text], images=[image], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        embedding = (outputs.image_embeds[0] + outputs.text_embeds[0]) / 2  # [512]
    return embedding.cpu().tolist()

# ▶️ 5. Qdrant에 저장
def store_to_qdrant(image_path, text, label,item):
    vec = embed_image_text_pair(image_path, text)
    gpt_response = ""
    for conv in item.get("conversations", []):
        if conv.get("from") == "gpt":
            gpt_response = conv.get("value")
            break  # 첫 GPT 응답만 가져옴
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "text_report": text,
                    "image_path": base_image_path + image_path,
                    "label": label,
                    "view" : item["view"],
                    "reason" : item["reason"],
                    "gpt" : gpt_response
                }
            )
        ]
    )
def store_to_qdrant_from_file(image_path, text_path, label, item):
    if not os.path.exists(base_image_path + text_path):
        print(f"[경고] 텍스트 파일 없음: {base_image_path + text_path}")
        return
    with open(base_image_path + text_path, "r") as f:
        text = f.read()
    text = summarizer(text, max_length=60, truncation=True)[0]['summary_text']
    # print(text)
    store_to_qdrant(image_path, text, label,item) 

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

# ▶️ 7. LangChain LLM 연결 (OpenAI 사용 시)
# llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# def ask_rag(query, search_results):
#     context = "\n".join(
#         f"Label: {r.payload['label']}\nText: {r.payload['text']}" for r in search_results
#     )
#     prompt = f"""
# <context>
# {context}
# </context>
# 질문: {query}
# 답변:
# """
#     return llm.predict(prompt)

# ▶️ 8. 예시 실행
# ✅ (1) 데이터 삽입
json_file = "/workspace/LLaVA-Med-RAG/data/mimic_cxr_annotaion_json/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1_rag_indexing.json";

# JSON 파일 로딩
with open(json_file, "r") as f:
    data = json.load(f)

# ✅ 중단된 인덱스부터 재시작
start_idx = 0

for item in tqdm(data[start_idx:], desc="🔄 Qdrant 저장 진행 중"):
    store_to_qdrant_from_file(item["image"], item["txt_report"], item["chexpert_labels_mapped"],item)

# ✅ (2) 이미지 기반 검색 후 RAG 응답
# image_results = search_with_query("/workspace/mimic/p10/p10046166/s50051329/427446c1-881f5cce-85191ce1-91a58ba9-0a57d3f5.jpg, mode="image")
# print(ask_rag("Describe the findings of the chest x-ray.", image_results))

# # # ✅ (3) 텍스트 기반 검색 후 RAG 응답
# text_results = search_with_query("The patient has a history of metastatic melanoma. The patient is presenting with confusion and somnolence. There is no evidence of acute cardiopulmonary process. Surgical clips and vascular markers in the thorax are related to prior CABG surgery.", mode="text")
# print(ask_rag("Describe the findings of the chest x-ray", text_results))


# ▶️ 9. Qdrant 벡터 검색만 수행하는 샘플 실행 예시

# (1) 텍스트 기반 검색
# print("===== 텍스트 기반 검색 결과 =====")
# query_text = "The patient has a history of metastatic melanoma. The patient is presenting with confusion and somnolence."
# results = search_with_query(query_text, mode="text", top_k=3)

# for i, r in enumerate(results):
#     print(f"[{i+1}] label: {r.payload['label']}")
#     print(f"     text_report: {r.payload['text_report']}")
#     print(f"     image_path: {r.payload['image_path']}")
#     print()

# # (2) 이미지 기반 검색
# print("===== 이미지 기반 검색 결과 =====")
# image_query_path = "/workspace/mimic/p10/p10046166/s57379357/6e511483-c7e1601c-76890b2f-b0c6b55d-e53bcbf6.jpg"
# results = search_with_query(image_query_path, mode="image", top_k=3)

# for i, r in enumerate(results):
#     print(f"[{i+1}] label: {r.payload['label']}")
#     print(f"     text_report: {r.payload['text_report']}")
#     print(f"     image_path: {r.payload['image_path']}")
#     print()