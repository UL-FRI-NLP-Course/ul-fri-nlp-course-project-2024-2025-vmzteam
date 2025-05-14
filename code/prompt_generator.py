import csv
import datetime
import os
import ollama
from similarity_calculator import calculate_bleu, bert_similarity, compute_tfidf_cosine_similarity

import html2text

html_converter = html2text.HTML2Text()
html_converter.ignore_links = True
html_converter.ignore_images = True
html_converter.body_width = 0

LANGUAGE_MODEL = "hf.co/tknez/GaMS-9B-Instruct-GGUF:Q6_K"
TARGET_TIME = datetime.datetime.strptime("2024-03-30 08:00", "%Y-%m-%d %H:%M")
TIME_WINDOW = 120  # minutes

MAIN_PROMPT = """THIS IS A TEST PROMPT"""

RELEVANT_FIELDS = [
"ContentNesreceSLO",
"ContentZastojiSLO",
"ContentOvireSLO",
"ContentDeloNaCestiSLO",
"ContentVremeSLO",
"ContentMednarodneInformacijeSLO",
"ContentSplosnoSLO"
]


def read_relevant_lines(file_path : str, target_time : datetime.datetime, time_window : int = 30) -> list:
    """Reads lines from csv file in the specified time window."""
    try:
        relevant_rows = []
        with open("../podatki.csv", newline='', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                date = row.get("Datum")
                if date:
                    date = datetime.datetime.strptime(date, "%d.%m.%Y %H:%M")
                    # append row if its date and time is lesser than target time for time window
                    if target_time - datetime.timedelta(minutes=time_window) <= date <= target_time:
                        # only append relevant fields
                        relevant_row = {field: row[field] for field in RELEVANT_FIELDS if field in row}
                        relevant_rows.append(relevant_row)

        return relevant_rows
    except Exception as e:
        print(f"Error reading file {file_path} {e}")
        return []


def send_prompt(instruction_prompt : str, input_query : str) -> str:
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
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



if __name__ == "__main__":
    # Define the path to the test_data folder
    test_data_folder = r"C:\Users\zanlu\Documents\FRI\MAG\ONJ\ul-fri-nlp-course-project-2024-2025-vmzteam\llm_outputs"

    # Read relevant lines from the CSV file
    relevant_lines = read_relevant_lines("../podatki.csv", TARGET_TIME, TIME_WINDOW)
    print("relevant_lines", relevant_lines)

    llm_response = send_prompt(MAIN_PROMPT, str(relevant_lines))

    # Save the LLM response to a file
    save_to_file(test_data_folder, llm_response, TARGET_TIME)