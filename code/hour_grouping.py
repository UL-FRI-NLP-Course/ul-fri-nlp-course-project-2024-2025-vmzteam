import csv
import html2text
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
import os
import pypandoc

# Nastavi html2text pretvornik
html_converter = html2text.HTML2Text()
html_converter.ignore_links = True
html_converter.ignore_images = True
html_converter.body_width = 0

i = 0

# Definiraj pare (naslov, vsebina)
category_columns = [
    ("TitleNesreceSLO", "ContentNesreceSLO", "Nesreče"),
    ("TitleZastojiSLO", "ContentZastojiSLO", "Zastoji"),
    ("TitleOvireSLO", "ContentOvireSLO", "Ovire"),
    ("TitleDeloNaCestiSLO", "ContentDeloNaCestiSLO", "Delo na cesti"),
    ("TitleVremeSLO", "ContentVremeSLO", "Vreme"),
    ("TitleMednarodneInformacijeSLO", "ContentMednarodneInformacijeSLO", "Mednarodne informacije"),
    ("TitleSplosnoSLO", "ContentSplosnoSLO", "Splošno"),
]

# Define the path to the test_data folder
test_data_folder = r"C:\Users\zanlu\Documents\FRI\MAG\ONJ\ul-fri-nlp-course-project-2024-2025-vmzteam\outputs"

def read_file_content(file_path):
    """Reads the content of a file."""
    try:
        with open(file_path, 'r', encoding="ascii", errors='replace') as f:
            raw_content = f.read()

        # Convert the content to plain text using pypandoc
        content = pypandoc.convert_text(raw_content, 'plain', format='rtf')
        return content

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def clean_html(content):
    if content is None:
        return ""
    return html_converter.handle(str(content)).strip()


def round_to_half_hour(time_obj):
    """Rounds a time object to the nearest half-hour."""
    if time_obj.minute < 30:
        return time_obj.replace(minute=30, second=0, microsecond=0)
    else:
        return (time_obj.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))

def sestavi_prompt(vrstica):
    datum = vrstica.get("Datum", "")
    prompt = f"Datum: {datum}\n\n"

    for title_key, content_key, default_title in category_columns:
        naslov = vrstica.get(title_key, "").strip()
        if naslov == "":
            naslov = default_title

        vsebina = clean_html(vrstica.get(content_key, ""))
        if vsebina:
            prompt += f"{naslov}:\n{vsebina}\n\n"

    # Construct the file name based on the Datum field

    try:
        global i
        date_part, time_part = datum.split(" ")
        #year, month, day = date_part.split("/")[0:3]
        #time_obj = datetime.strptime(time_part, "%H:%M")

        # Round the time to the nearest half-hour
        #rounded_time = round_to_half_hour(time_obj)
        #time_formatted = rounded_time.strftime("%H%M")

        date_formatted = datetime.strptime(date_part, "%d.%m.%Y").strftime("%d_%m_%Y")

        time_obj = datetime.strptime(time_part, "%H:%M")
        rounded_time = round_to_half_hour(time_obj)
        time_formatted = rounded_time.strftime("%H_%M")
        #time_formatted = datetime.strptime(time_part, "%H:%M").strftime("%H_%M")

        file_name = f"report_{date_formatted}_{time_formatted}.rtf"

        #file_name = f"report_{day}_{month}_{year}_{time_formatted}.rtf"

        file_path = os.path.join(test_data_folder, file_name)
        # Append the file content if it exists
        if os.path.exists(file_path):
            i += 1
            file_content = read_file_content(file_path)
            prompt += f"Pričakovani rezultat:\n{file_content}\n\n"
            print(f"File content for {file_name}:\n{file_content}\n")
    except Exception as e:
        print(f"Error constructing file name for Datum '{datum}': {e}")
    
    return prompt.strip()


def group_reports_by_half_hour(reader):
    grouped_reports = defaultdict(list)
    for vrstica in reader:
        datum = vrstica.get("Datum", "").strip()
        try:
            # Parse the time from the "Datum" field
            date_part, time_part = datum.split()  # Assuming time is the last part of the "Datum" field
            time_obj = datetime.strptime(time_part, "%H:%M")
            
            # Determine the half-hour group
            if time_obj.minute < 30:
                group_key = time_obj.replace(minute=30, second=0, microsecond=0).strftime("%H:%M")
            else:
                group_key = (time_obj.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)).strftime("%H:%M")

            group_key = f"{date_part} {group_key}"
            # Add the report to the corresponding group
            grouped_reports[group_key].append(vrstica)
        except ValueError as e:
            print(f"Error parsing date/time: {e}")
            # If parsing fails, skip this row
            continue
    return grouped_reports

# Preberi CSV in shrani promte
with open("../podatki.csv", newline='', encoding="utf-8") as csvfile, \
    open("llm_prompts.txt", "w", encoding="utf-8") as outfile:
    reader = csv.DictReader(csvfile, delimiter=";")
    grouped_reports = group_reports_by_half_hour(reader)

    for idx, (time_range, reports) in enumerate(grouped_reports.items(), start=1):
        outfile.write(f"--- GROUPED REPORT #{idx} ---\n")
        outfile.write(f"Časovni interval: {time_range}\n\n")
        for report in reports:
            prompt = sestavi_prompt(report)
            if prompt:
                outfile.write(prompt + "\n\n")
        outfile.write("="*50 + "\n\n")

print(f"✅ Skupaj {i} datotek")
print("✅ Prompti so shranjeni v datoteko 'llm_prompts.txt'")