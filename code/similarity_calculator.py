from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens')

def compute_tfidf_cosine_similarity(text1, text2):
    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform both texts
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Compute cosine similarity between the two vectors
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    # Return the similarity score (it’s a 2D array, so extract [0][0])
    return similarity[0][0]


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
    bleu_score = sentence_bleu([reference_prompts.split()], generated_prompt.split(), smoothing_function=smoothing_function)
    return bleu_score

def bert_similarity(text1, text2):
    embeddings = model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def process_grouped_reports(file_path, similarity_threshold=0.8, window_size=5):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    grouped_reports = content.split('--- GROUPED REPORT')
    report_data_list = []  # Store all report_data for later comparison
    expected_results = []  # Store all expected_results for later comparison

    # Parse the reports and store data
    for report in grouped_reports[1:]:  # Skip the first split as it's empty
        lines = report.strip().split('\n')
        report_number = lines[0].strip().split('#')[-1].strip()

        # Split the report text by "Pričakovani rezultat:"
        if "Pričakovani rezultat:" in report:
            report_data, expected_result = report.split("Pričakovani rezultat:", 1)
            report_data = report_data.strip()
            expected_result = expected_result.strip()
        else:
            report_data = report.strip()
            expected_result = None

        report_data_list.append(report_data)
        expected_results.append((report_number, expected_result))

    # Compare each expected_result with nearby report_data
    for i, (report_number, expected_result) in enumerate(expected_results):
        if expected_result:
            for j in range(max(0, i - window_size), min(len(report_data_list), i + window_size + 1)):  # Skip comparing the report with itself
                #similarity = compute_tfidf_cosine_similarity(expected_result, report_data_list[j])
                #similarity = calculate_bleu(expected_result, report_data_list[j])
                similarity = bert_similarity(expected_result, report_data_list[j])
                if similarity >= similarity_threshold:
                    print(f"Expected result {report_number} similar to Report #{j + 1} with similarity: {similarity:.4f}")

if __name__ == "__main__":
    # Path to the llm_prompts.txt file
    file_path = "/Users/matija.tomazic/Documents/Faks/ONJ/project_local_ollama/data_setup/llm_prompts.txt"
    
    # Process the grouped reports
    process_grouped_reports(file_path, window_size=5, similarity_threshold=0.92)