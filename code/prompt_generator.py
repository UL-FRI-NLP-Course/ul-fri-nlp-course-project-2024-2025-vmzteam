import csv
import datetime
import json
import os
import ollama
from bert_score import BERTScorer
import html2text
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

html_converter = html2text.HTML2Text()
html_converter.ignore_links = True
html_converter.ignore_images = True
html_converter.body_width = 0

LANGUAGE_MODEL = "hf.co/tknez/GaMS-9B-Instruct-GGUF:F16"
TARGET_TIME = datetime.datetime.strptime("2024-03-30 08:00", "%Y-%m-%d %H:%M")
TIME_WINDOW = 60  # minutes

MAIN_PROMPT = """S spodnjih podatkov generiraj besedilo o prometu za branje po radiju.
Besedilo naj bo jedrnato. Bolj pomembne informacije daj na začetek besedila.

Podatki:"""

RELEVANT_FIELDS = [
"ContentNesreceSLO",
"ContentZastojiSLO",
"ContentOvireSLO",
"ContentDeloNaCestiSLO",
"ContentVremeSLO",
"ContentMednarodneInformacijeSLO",
"ContentSplosnoSLO"
]

category_columns = [
    ("TitleNesreceSLO", "ContentNesreceSLO", "Nesreče"),
    ("TitleZastojiSLO", "ContentZastojiSLO", "Zastoji"),
    ("TitleOvireSLO", "ContentOvireSLO", "Ovire"),
    ("TitleDeloNaCestiSLO", "ContentDeloNaCestiSLO", "Delo na cesti"),
    ("TitleVremeSLO", "ContentVremeSLO", "Vreme"),
    ("TitleMednarodneInformacijeSLO", "ContentMednarodneInformacijeSLO", "Mednarodne informacije"),
    ("TitleSplosnoSLO", "ContentSplosnoSLO", "Splošno"),
]


def calculate_bert(reference_text, generated_text):
    scorer = BERTScorer(model_type='bert-base-uncased')
    precision, recall, f1 = scorer.score([generated_text], [reference_text])
    return [precision.item(), recall.item(), f1.item()]


def calculate_bleu(reference_prompts, generated_prompt):
    """
    Calculate the BLEU score for a generated prompt against reference prompts.

    Args:
        reference_prompts (list of list of str): A list of reference prompts, where each reference is a list of words.
        generated_prompt (list of str): The generated prompt as a list of words.

    Returns:
        float: The BLEU score.
    """
    # Use smoothing to avoid zero scores for short sentences
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference_prompts, generated_prompt, smoothing_function=smoothing_function)
    return bleu_score


def read_relevant_lines(file_path : str, target_time : datetime.datetime, time_window : int = 30) -> list:
    """Reads lines from csv file in the specified time window."""
    try:
        relevant_rows = []
        with open("../podatki.csv", newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                date = row.get("Datum")
                if date:
                    date = datetime.datetime.strptime(date, "%m/%d/%y %H:%M")
                    # append row if its date and time is lesser than target time for time window
                    if target_time - datetime.timedelta(minutes=time_window) <= date <= target_time:
                        # only append relevant fields
                        relevant_row = {field: row[field] for field in RELEVANT_FIELDS if field in row}
                        relevant_rows.append(relevant_row)

        return relevant_rows
    except Exception as e:
        print(f"Error reading file {file_path} {e}")
        return []


def clean_html(content):
    if content is None:
        return ""
    return html_converter.handle(str(content)).strip()

def concatenate_prompt(relevant_lines : list) -> str:
    prompt = ""
    for title_key, content_key, default_title in category_columns:
            vsebina = ""
            unique_lines = set()
            for line in relevant_lines:
                line_content = line.get(content_key, "")
                if line_content:
                    line_content = clean_html(line_content)
                    if line_content not in unique_lines:
                        unique_lines.add(line_content)
                        vsebina = vsebina + line_content + "\n"
            if vsebina:
                prompt += f"{default_title}:\n{vsebina}\n\n"
    return prompt


def send_prompt(instruction_prompt : str, input_query : str) -> str:
    """Sends a prompt to the LLM and returns the response."""
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'user', 'content': MAIN_PROMPT + input_query},
        ],
        stream=True,
    )

    # print the response from the chatbot in real-time
    print('Chatbot response:')
    response = ''
    for chunk in stream:
        content = chunk['message']['content']
        print(content, end='', flush=True)
        response += content

    return response

def save_to_file(target_dir : str, content : str, date : datetime.datetime) -> None:
    """Saves the content to a file."""
    try:
        os.makedirs(target_dir, exist_ok=True)

        file_path = os.path.join(target_dir, f"llm_report_{date.strftime('%d_%m_%Y')}_{date.strftime('%H_%M')}.rtf")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
            print("File saved successfully:", file_path)

    except Exception as e:
        print(f"Error saving file {file_path}: {e}")


# TODO - implementtion
def calculate_similarities(text1 : str, text2 : str):
    """
    Calculate the similarity between two texts using different methods from similarity_calculator.py.
    """
    # Compute BLEU score
    bleu_score = calculate_bleu(text1, text2)

    # Compute BERT similarity
    bert_sim = calculate_bert(text1, text2)

    return bleu_score, bert_sim


def deduplicate_block_text(text_block, threshold=0.9):
    lines = [line.strip() for line in text_block.split("\n") if line.strip()]
    if len(lines) <= 1:
        return "\n".join(lines)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(lines)
    sim_matrix = cosine_similarity(tfidf_matrix)

    # poišči podvojene vrstice
    seen = set()
    keep = []
    for i in range(len(lines)):
        if i in seen:
            continue
        keep.append(lines[i])
        for j in range(i + 1, len(lines)):
            if sim_matrix[i, j] > threshold:
                seen.add(j)
    return "\n".join(keep)

def generate_llm_reports_for_matching_files(file_path : str) -> None:
    # Open the CSV file for writing similarity scores
    with open("similarity_scores.csv", "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["report_date", "bleu_score", "bert_precision", "bert_recall", "bert_f1"])

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                report_name = item.get("output")
                report_date = datetime.datetime.strptime(report_name.split("report_")[1].replace(".rtf", ""), "%d_%m_%Y_%H_%M")

                relevant_lines = read_relevant_lines("../podatki.csv", report_date, TIME_WINDOW)

                updated_lines = concatenate_prompt(relevant_lines)

                updated_lines = deduplicate_block_text(updated_lines)
                print(updated_lines)
                llm_response = send_prompt(MAIN_PROMPT, updated_lines)

                bleu_score, bert_sim = calculate_similarities(item.get("output_text"), llm_response)

                print("Similarity scores:", bleu_score, bert_sim)
                save_to_file("../llm_outputs", "\n\n".join([llm_response, item.get("output_text")]), report_date)

                # Write the scores and date to the CSV
                writer.writerow([
                    report_date.strftime("%Y-%m-%d %H:%M"),
                    bleu_score,
                    bert_sim[0],  # precision
                    bert_sim[1],  # recall
                    bert_sim[2],  # f1
                ])


if __name__ == "__main__":
    # Define the path to the test_data folder
    test_data_folder = r"C:\Users\zanlu\Documents\FRI\MAG\ONJ\ul-fri-nlp-course-project-2024-2025-vmzteam\llm_outputs"

    # Read relevant lines from the CSV file
    #relevant_lines = read_relevant_lines("../podatki.csv", TARGET_TIME, TIME_WINDOW)
    #print("relevant_lines", relevant_lines)

    #llm_response = send_prompt(MAIN_PROMPT, str(relevant_lines))

    # Save the LLM response to a file
    #save_to_file(test_data_folder, llm_response, TARGET_TIME)

    generate_llm_reports_for_matching_files("useful_matches.json")