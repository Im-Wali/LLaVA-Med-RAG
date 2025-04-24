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

# qdrant 필요 라이브러리
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from transformers import CLIPProcessor, CLIPModel 

logging.set_verbosity_error()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# qrant 검색 함수.
def search_with_query(query_input, args,clip_processor,clip_model,client, mode="image"):
    
    if mode == "image":
        print(f"query_input : {query_input}")
        image = Image.open(query_input).convert("RGB")
        inputs = clip_processor(images=[image], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            query_vec = clip_model.get_image_features(**inputs)[0].cpu().tolist()
    else:
        inputs = clip_processor(text=[query_input], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            query_vec = clip_model.get_text_features(**inputs)[0].cpu().tolist()

    results = client.search(
        collection_name=args.qdrant_collection_name,
        query_vector=query_vec,
        limit=args.top_k
    )
    return results


def eval_model(args):
    set_seed(150)
    disable_torch_init()

    # qdrant 셋팅(CLIP 모델 다른 모델 사용. 추후 변경할지는 결정.)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Qdrant 연결
    client = QdrantClient(
        url=args.qdrant_url, 
        api_key=args.qdrant_api_key,
    )

    # 모델 로드
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = questions[args.chunk_idx::args.num_chunks]  # 청크 처리
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    existing_ids = set()
    if os.path.exists(answers_file):
        with open(answers_file, "r") as f:
            for line in f:
                try:
                    existing = json.loads(line)
                    existing_ids.add(existing["question_id"])
                except:
                    continue
    
    # 결과 파일 이어쓰기 모드로 열기
    ans_file = open(answers_file, "a")

    # ans_file = open(answers_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_path = line.get("image_path", "")
        image_file = line["image"]
        qs = line["text"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
        cur_prompt = qs
        assitant = "\n<context>\n"

        # 있는 idx인 경우 생략
        if idx in existing_ids:
            continue

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        results = search_with_query(args.image_folder + image_file, args, clip_processor, clip_model, client, mode="image")

        # top-k 만큼 조회된 결과를 기존 프롬프트에 추가.
        for i, r in enumerate(results):
            value_str = str(r.payload['text_report'])  # 문자열로 변환
            value_str = value_str.replace('/workspace/', '')  # 경로 일부 제거
            assitant += f"{value_str}\n"
        assitant += "</context>\n"

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], assitant)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        # 이미지 로드
        image_path_full = os.path.join(args.image_folder, image_path, image_file)
        image = Image.open(image_path_full)
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
    parser.add_argument("--qdrant-collection-name", type=str, default=None)
    parser.add_argument("--top-k", type=float, default=3)
    parser.add_argument("--qdrant-api-key", type=str, default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.tw1O_qcpywHq2wr7uyXSidgDywWpVdMxd5y-kgBJ5RY")
    parser.add_argument("--qdrant-url", type=str, default="https://d4d97f9c-07f1-4931-86f9-cadc235fc91b.us-west-1-0.aws.cloud.qdrant.io:6333")
    args = parser.parse_args()

    eval_model(args)
