import pandas as pd
from datetime import datetime, timedelta
import re
import os
import pypandoc
import matplotlib.pyplot as plt
import json
import concurrent.futures
from threading import Lock

# new imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#pypandoc.download_pandoc()
date_time_pattern = r'\b(\d{1,2})[.,]?\s*(\d{1,2})[.,]?\s*(\d{4})\b\s*-?\s*(\d{1,2})[.:,](\d{2})\b'
number_of_decent = 0
similarities = []
useful = []

similarity_lock = Lock()
number_of_decent_lock = Lock()
useful_lock = Lock()

df = pd.read_csv('../podatki.csv', sep=';')
df['ts'] = pd.to_datetime(df['Datum'], format='mixed', dayfirst=True)

def strip_html(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'<.*?>', '', text).strip()

def find_best_match(df, rtf_ts, rtf_body, rtf_path):
    try:
        # 3. Time-window filter
        candidates = df[
            (df.ts >= rtf_ts - timedelta(minutes=180)) &
            (df.ts <= rtf_ts + timedelta(minutes=5))
        ].copy()


        # 4. Build a “document” for each candidate
        docs = []
        for idx, row in candidates.iterrows():
            parts = []
            for col in [
                "TitlePomembnoSLO","ContentPomembnoSLO",
                "TitleNesreceSLO","ContentNesreceSLO",
                "TitleZastojiSLO","ContentZastojiSLO",
                "TitleVremeSLO","ContentVremeSLO",
                "TitleOvireSLO","ContentOvireSLO",
                "TitleDeloNaCestiSLO","ContentDeloNaCestiSLO",
                "TitleOpozorilaSLO","ContentOpozorilaSLO",
                "ContentMednarodneInformacijeSLO",
                "TitleSplosnoSLO","ContentSplosnoSLO"
            ]:
                parts.append(strip_html(row.get(col, "")))
            docs.append("\n".join(parts))

        # 5. Vectorize
        #    - we fit on all candidate docs + the single RTF body,
        #      so term-space is shared
        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents='unicode'
        )

        # create a corpus: all candidate docs, plus rtf_body at the end
        corpus = docs + [rtf_body]
        tfidf = vectorizer.fit_transform(corpus)

        # 6. Compute cosine similarities between each candidate and the rtf_body
        #    the rtf_body vector is the last row of tfidf
        rtf_vec     = tfidf[-1]
        cand_vecs   = tfidf[:-1]
        sims        = cosine_similarity(cand_vecs, rtf_vec.reshape(1, -1)).flatten()

        # 7. Pick best match
        candidates = candidates.reset_index(drop=True)
        candidates['tfidf_sim'] = sims
        best_idx   = candidates['tfidf_sim'].idxmax()
        best_row   = candidates.loc[best_idx]

        print("Best match (similarity = {:.3f}):".format(best_row['tfidf_sim']))
        global number_of_decent, similarities, useful
        similarities.append(best_row['tfidf_sim'])
        if best_row['tfidf_sim'] > 0.6:
            number_of_decent += 1
            useful.append({"input": best_row['LegacyId'], "output": rtf_path, "similarity": best_row['tfidf_sim']})
            print(best_row, rtf_body)
        #print(best_row)

        return best_row
    except Exception as e:
        print(e)
        return ""

def iterate_outputs(base_folder):
    global df
    # Iterate through all files in the outputs folder
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".rtf"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding="ascii", errors='replace') as f:
                    raw_content = f.read()

                # Convert the content to plain text using pypandoc
                rtf_body = pypandoc.convert_text(raw_content, 'plain', format='rtf')

                match = re.search(date_time_pattern, rtf_body)
                if match:
                    day, month, year, hour, minute = map(int, match.groups())
                    # Create a datetime object
                    rtf_ts = datetime(year, month, day, hour, minute)

                    find_best_match(df, rtf_ts, rtf_body, file_path)


def process_file(file_path):
    global df, number_of_decent, similarities, useful

    with open(file_path, 'r', encoding="ascii", errors='replace') as f:
        raw_content = f.read()

    # Convert the content to plain text using pypandoc
    rtf_body = pypandoc.convert_text(raw_content, 'plain', format='rtf')

    match = re.search(date_time_pattern, rtf_body)
    if match:
        day, month, year, hour, minute = map(int, match.groups())
        # Create a datetime object
        rtf_ts = datetime(year, month, day, hour, minute)

        best_row = find_best_match(df, rtf_ts, rtf_body, file_path)
        return best_row
    return None


def iterate_outputs_threaded(base_folder, max_workers=8):
    file_paths = []

    # First collect all file paths
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".rtf"):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    # Then process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_file, file_paths)



if __name__ == "__main__":
    base_folder = "../outputs"
    #iterate_outputs(base_folder)
    iterate_outputs_threaded(base_folder)
    print("Number of decent matches: ", number_of_decent)
    # plot the similarities

    plt.plot(similarities)
    plt.title("Cosine Similarities")
    plt.show()

    # save the useful matches to a file
    serializable_useful = []
    for item in useful:
        serializable_useful.append({
            "input": int(item["input"]) if hasattr(item["input"], "item") else item["input"],
            "output": item["output"],
            "similarity": float(item["similarity"]) if hasattr(item["similarity"], "item") else item["similarity"]
        })

    with open("useful_matches.json", "w") as f:
        json.dump(serializable_useful, f, indent=4)
