import os
import pypandoc
import re
import chardet
from datetime import datetime
import shutil

pypandoc.download_pandoc()
date_time_pattern = r'\b(\d{1,2})[.,]?\s*(\d{1,2})[.,]?\s*(\d{4})\b\s*-?\s*(\d{1,2})[.:,](\d{2})\b'
def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        #print(f"Detected encoding: {encoding} (Confidence: {confidence})")
        return encoding


def parse_rtf_files(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".rtf"):
                file_path = os.path.join(root, file)
                detected_encoding = detect_file_encoding(file_path)
                try:
                    # Read the file content with a specific encoding
                    with open(file_path, 'r', encoding=detected_encoding, errors='replace') as f:
                        raw_content = f.read()

                    # Convert the content to plain text using pypandoc
                    content = pypandoc.convert_text(raw_content, 'plain', format='rtf')
                    #print(f"\n--- Content of {file_path} ---\n")
                    #print(content)

                    # Search for dates and times in the content
                    match = re.search(date_time_pattern, content)
                    if match:
                        day, month, year, hour, minute = map(int, match.groups())
                        # Create a datetime object
                        dt = datetime(year, month, day, hour, minute)
                        # Format the file name
                        file_name = f"report_{dt.strftime('%d_%m_%Y_%H_%M')}.rtf"
                        print(f"Generated file name: {file_name}")

                        # apply the new name to the file
                        new_file_path = os.path.join(root, file_name)
                        os.rename(file_path, new_file_path)
                        print(f"Renamed {file_path} to {new_file_path}")
                    else:
                        print(f"\nNo valid date and time found for file {file_path}. Contetnt:\n{content}")

                except Exception as e:
                    print(f"Error reading {file_path}: {e}")


def move_report_files_to_outputs(base_folder):
    # Create the outputs folder if it doesn't exist
    outputs_folder = os.path.join(base_folder, "outputs")
    os.makedirs(outputs_folder, exist_ok=True)

    # Walk through all subdirectories
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.startswith("report_"):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(outputs_folder, file)
                try:
                    # Move the file to the outputs folder
                    shutil.move(source_path, destination_path)
                    print(f"Moved: {source_path} -> {destination_path}")
                except Exception as e:
                    print(f"Error moving file {source_path}: {e}")


if __name__ == "__main__":
    folder_path = r"C:\Users\zanlu\Documents\FRI\MAG\ONJ\RTVSlo\RTVSlo\Podatki - rtvslo.si\Promet 2024 - test"  # Replace with your folder path
    #parse_rtf_files(folder_path)

    move_report_files_to_outputs(folder_path)

    #file_path = r"C:\Users\zanlu\Documents\FRI\MAG\ONJ\RTVSlo\RTVSlo\Podatki - rtvslo.si\Promet 2024\April 2024\TMP4-2024-121.rtf"  # Replace with your file path
    #detected_encoding = detect_file_encoding(file_path)