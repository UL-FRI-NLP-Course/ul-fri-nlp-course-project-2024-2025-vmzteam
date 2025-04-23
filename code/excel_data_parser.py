import csv

# List of expected columns (in exact order)
expected_columns = [
    "Datum", "Operater", "A1", "B1", "C1", "A2", "B2", "C2",
    "TitlePomembnoSLO", "ContentPomembnoSLO",
    "TitleNesreceSLO", "ContentNesreceSLO",
    "TitleZastojiSLO", "ContentZastojiSLO",
    "TitleVremeSLO", "ContentVremeSLO",
    "TitleOvireSLO", "ContentOvireSLO",
    "TitleDeloNaCestiSLO", "ContentDeloNaCestiSLO",
    "TitleOpozorilaSLO", "ContentOpozorilaSLO",
    "TitleMednarodneInformacijeSLO", "ContentMednarodneInformacijeSLO",
    "TitleSplosnoSLO", "ContentSplosnoSLO"
]

def parse_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        missing = [col for col in expected_columns if col not in reader.fieldnames]
        if missing:
            print(f"Missing columns: {missing}")
            return
        
        for i, row in enumerate(reader, 1):
            print(f"\n--- Row {i} ---")
            for col in expected_columns:
                print(f"{col}: {row.get(col, '')}")

# Example usage
if __name__ == "__main__":
    parse_csv("your_file.csv")  # Change to your actual file name
