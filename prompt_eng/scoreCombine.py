import os
import re
import csv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import math

FILES = [
    "cot + few.txt",
    "cot + few_gpt.txt",
    "cot.txt",
    "few_shot.txt",
    "zero_shot.txt",
    "few_shot_gpt.txt"
]

REFERENCES = [
    "expected_output.txt",
    "expected_output_gpt.txt"
]

def extract_last_report(text):
    """Finds and returns the last 'Poročilo:' or 'Odgovor:' block in a file."""
    matches = re.findall(r"(Poročilo:|Odgovor:)\s*(.*?)(?=(Poročilo:|Odgovor:|$))", text, re.DOTALL)
    if matches:
        return matches[-1][1].strip()
    return ""

def load_lines(filepath):
    with open(filepath, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def calculate_bleu(reference, candidate):
    reference_tokens = [ref.split() for ref in reference]
    candidate_tokens = candidate.split()
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=SmoothingFunction().method1)

def calculate_rouge_l(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(reference, candidate)['rougeL'].fmeasure

def calculate_bert_score(references, candidates):
    _, _, f1 = bert_score(candidates, references, lang="sl", rescale_with_baseline=True)
    return float(f1.mean())

def main():
    output_csv = "metrics_bleu_rouge_bert_dual.csv"
    subdirs = sorted([d for d in os.listdir() if os.path.isdir(d) and d.startswith("data_")])

    with open(output_csv, mode="w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Mapa", "Datoteka", "Referenca", "Ujemanje", "BLEU", "ROUGE-L", "BERTScore"])

        for subdir in subdirs:
            for gen_file in FILES:
                gen_path = os.path.join(subdir, gen_file)
                if not os.path.exists(gen_path):
                    continue

                try:
                    with open(gen_path, encoding="utf-8") as gf:
                        generated_blocks = [extract_last_report(block) for block in gf.read().split("\n\n") if "Poročilo:" in block or "Odgovor:" in block]
                except Exception as e:
                    print(f"[ERROR] Napaka pri branju {gen_path}: {e}")
                    continue

                for ref_file in REFERENCES:
                    ref_path = os.path.join(subdir, ref_file)
                    if not os.path.exists(ref_path):
                        continue

                    references = load_lines(ref_path)
                    min_len = min(len(generated_blocks), len(references))

                    if len(generated_blocks) != len(references):
                        print(f"[INFO] {gen_path} vs {ref_path}: mismatch ({len(generated_blocks)} vs {len(references)})")

                    bleu_scores = []
                    rouge_scores = []

                    for ref, gen in zip(references[:min_len], generated_blocks[:min_len]):
                        bleu_scores.append(calculate_bleu([ref], gen))
                        rouge_scores.append(calculate_rouge_l(ref, gen))

                    # BERT score (poskusimo, če je mogoče izračunati)
                    try:
                        bert_avg = calculate_bert_score(references[:min_len], generated_blocks[:min_len]) if min_len > 0 else 0.0
                    except Exception as e:
                        print(f"[WARNING] BERTScore napaka za {gen_path} vs {ref_path}: {e}")
                        bert_avg = 0.0

                    # Varnostna zaščita pred napačnimi metrikami
                    if (not bleu_scores or not rouge_scores or
                        math.isnan(bert_avg) or math.isinf(bert_avg) or bert_avg > 1.0):
                        avg_bleu = 0.0
                        avg_rouge = 0.0
                        bert_avg = 0.0
                    else:
                        avg_bleu = sum(bleu_scores) / len(bleu_scores)
                        avg_rouge = sum(rouge_scores) / len(rouge_scores)

                    writer.writerow([
                        subdir,
                        gen_file,
                        ref_file,
                        f"{len(generated_blocks)} vs {len(references)}",
                        f"{avg_bleu:.4f}",
                        f"{avg_rouge:.4f}",
                        f"{bert_avg:.4f}"
                    ])
                    print(f"{subdir}/{gen_file} vs {ref_file} → BLEU: {avg_bleu:.4f}, ROUGE-L: {avg_rouge:.4f}, BERTScore: {bert_avg:.4f}")

if __name__ == "__main__":
    main()
