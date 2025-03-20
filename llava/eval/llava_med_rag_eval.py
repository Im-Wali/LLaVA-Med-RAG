import json
import nltk
import argparse
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support
import sacrebleu

nltk.download('wordnet')


class llava_med_rag_eval:
    def __init__(self, args):
        self.question_file = args.question_file
        self.answer_file = args.answers_file
        self.output_file = args.output_file

    def load_vqa_results(self):
        """JSONL 파일에서 질문-정답-예측값 리스트 반환"""
        with open(self.answer_file, "r") as f:
            preds = [json.loads(line) for line in f]
        with open(self.question_file, "r") as f:
            gts = [json.loads(line) for line in f]

        return preds, gts

    def compute_metrics(self, preds, gts):
        """VQA 성능을 평가하는 주요 평가지표 계산"""
        assert len(preds) == len(gts), "Prediction과 Ground Truth 개수가 다름"

        bleu_scores = []
        rouge_l_scores = []
        exact_match = []
        ref_texts = []
        hyp_texts = []

        for pred, gt in zip(preds, gts):
            ref = gt["gpt4_answer"].lower().split()  # 정답 ： GPT-4를 이용한 답변
            hyp = pred["text"].lower().split()  # 예측값

            # BLEU Score
            smoothie = SmoothingFunction().method1
            bleu = sentence_bleu([ref], hyp, smoothing_function=smoothie)
            bleu_scores.append(bleu)

            # ROUGE-L Score
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            rouge_l = scorer.score(" ".join(ref), " ".join(hyp))["rougeL"].fmeasure
            rouge_l_scores.append(rouge_l)

            # METEOR Score
            ref_texts.append(" ".join(ref))
            hyp_texts.append(" ".join(hyp))

            # Exact Match (정확히 일치하는 경우 1, 아니면 0)
            exact_match.append(int(" ".join(ref) == " ".join(hyp)))

        # METEOR Score
        meteor = sacrebleu.corpus_bleu(hyp_texts, [ref_texts]).score

        # F1 Score
        precision, recall, f1, _ = precision_recall_fscore_support(exact_match, exact_match, average='binary')

        metrics = {
            "BLEU": np.mean(bleu_scores),
            "ROUGE-L": np.mean(rouge_l_scores),
            "METEOR": meteor,
            "Exact Match": np.mean(exact_match),
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall
        }

        return metrics

    def evaluate(self):
        """VQA 평가 수행 및 결과 저장"""
        preds, gts = self.load_vqa_results()
        metrics = self.compute_metrics(preds, gts)

        print("VQA 평가 결과:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # JSON 파일로 결과 저장
        with open(self.output_file, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"평가 결과 저장 완료: {self.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--output-file", type=str, default="vqa_evaluation.json")

    args = parser.parse_args()

    # 평가 실행
    evaluator = llava_med_rag_eval(args)
    evaluator.evaluate()
