import json
import nltk
import argparse
import numpy as np
import re

from nltk.stem import WordNetLemmatizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support
import sacrebleu
from bert_score import score as bert_score  # BERTScore 추가

# 필수 리소스 다운로드
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import TreebankWordTokenizer

# 토크나이저 및 표제어 처리기
tokenizer = TreebankWordTokenizer()
lemmatizer = WordNetLemmatizer()

# 간단한 의료 표현 동의어 정규화 테이블 (확장 가능)
MEDICAL_SYNONYMS = {
    "effusion": "fluid",
    "opacity": "lesion",
    "calcified": "calcification",
    "heart size": "cardiac size",
    "no focal consolidation": "clear",
    "wire": "surgical wire",
    "pneumothorax": "no air leak",
}

# 의료 관련 주요 개체 키워드 리스트
MEDICAL_ENTITIES = [
    "effusion", "pneumothorax", "edema", "consolidation", "granuloma",
    "opacity", "wire", "fracture", "cardiomegaly", "calcified", "catheter",
    "stent", "line", "nodule", "mass", "clip", "fibrosis", "airspace", "atelectasis"
]

def normalize_text(text):
    """텍스트를 정규화: 소문자화 + 문장부호 제거 + 표제어화 + 동의어 치환"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = tokenizer.tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    normalized = [MEDICAL_SYNONYMS.get(w, w) for w in lemmatized]
    return normalized

def jaccard_similarity(set1, set2):
    """Soft Exact Match를 위한 Jaccard 유사도 계산"""
    intersection = set(set1).intersection(set2)
    union = set(set1).union(set2)
    return len(intersection) / len(union) if union else 0

def extract_medical_entities(tokens):
    """정규화된 토큰 리스트에서 의료 개체 추출"""
    return set([token for token in tokens if token in MEDICAL_ENTITIES])

class llava_med_rag_eval:
    def __init__(self, args):
        self.question_file = args.question_file
        self.answer_file = args.answers_file
        self.output_file = args.output_file

    def load_vqa_results(self):
        """질문 및 정답 JSONL 파일 불러오기"""
        with open(self.answer_file, "r") as f:
            preds = [json.loads(line) for line in f]
        with open(self.question_file, "r") as f:
            gts = [json.loads(line) for line in f]
        return preds, gts

    def compute_metrics(self, preds, gts):
        """BLEU, ROUGE, METEOR, Soft Match, BERTScore, Medical Entity 평가"""
        assert len(preds) == len(gts), "Prediction과 Ground Truth 개수가 다름"

        bleu_scores = []
        rouge_l_scores = []
        exact_match = []
        ref_texts = []
        hyp_texts = []

        entity_precisions = []
        entity_recalls = []
        entity_f1s = []

        for pred, gt in zip(preds, gts):
            ref = normalize_text(gt["gpt4_answer"])
            hyp = normalize_text(pred["text"])

            # BLEU Score
            smoothie = SmoothingFunction().method1
            bleu = sentence_bleu([ref], hyp, smoothing_function=smoothie)
            bleu_scores.append(bleu)

            # ROUGE-L Score
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            rouge_l = scorer.score(" ".join(ref), " ".join(hyp))["rougeL"].fmeasure
            rouge_l_scores.append(rouge_l)

            # METEOR용 원문 저장
            ref_texts.append(" ".join(ref))
            hyp_texts.append(" ".join(hyp))

            # Soft Exact Match
            similarity = jaccard_similarity(ref, hyp)
            exact_match.append(1 if similarity >= 0.85 else 0)

            # 의료 개체 평가
            ref_entities = extract_medical_entities(ref)
            hyp_entities = extract_medical_entities(hyp)

            tp = len(ref_entities & hyp_entities)
            precision = tp / len(hyp_entities) if hyp_entities else 0.0
            recall = tp / len(ref_entities) if ref_entities else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            entity_precisions.append(precision)
            entity_recalls.append(recall)
            entity_f1s.append(f1)

        # METEOR Score
        meteor = sacrebleu.corpus_bleu(hyp_texts, [ref_texts]).score

        # BERTScore
        P, R, F1 = bert_score(hyp_texts, ref_texts, lang='en', rescale_with_baseline=True)

        # F1 Score (기존 exact match 기반)
        precision, recall, f1_macro, _ = precision_recall_fscore_support(
            exact_match, exact_match, average='binary'
        )

        # 전체 메트릭 반환
        metrics = {
            "BLEU": np.mean(bleu_scores),
            "ROUGE-L": np.mean(rouge_l_scores),
            "METEOR": meteor,
            "Soft Exact Match (Jaccard >= 0.85)": np.mean(exact_match),
            "F1 Score": f1_macro,
            "Precision": precision,
            "Recall": recall,
            "BERTScore_F1": F1.mean().item(),
            "MedEntity_Precision": np.mean(entity_precisions),
            "MedEntity_Recall": np.mean(entity_recalls),
            "MedEntity_F1": np.mean(entity_f1s)
        }

        return metrics

    def evaluate(self):
        preds, gts = self.load_vqa_results()
        metrics = self.compute_metrics(preds, gts)

        print("VQA 평가 결과:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        with open(self.output_file, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"평가 결과 저장 완료: {self.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--output-file", type=str, default="vqa_evaluation.json")
    args = parser.parse_args()

    evaluator = llava_med_rag_eval(args)
    evaluator.evaluate()
