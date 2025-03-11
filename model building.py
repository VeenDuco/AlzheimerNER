#-----note---- update 11.3.2025
# file laoded in hard copy
# 1. cleaning text from json files to extract the right text
# 2. train model : assign the extracted text into semantic group from dictionary given
#                 - save the the train model in excel files
# ratio split : 80% training, 20% testing
# feauture used : crf

# problem :
# 1. testing model dont pickup the right entity. majority entity is missing out
# 2. overlapping detection of semantic and entity

import re
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report

# Define directories
# Get the current script's directory (where "Alzheimer ML NER Project.py" is located)
BASE_DIR = Path(__file__).resolve().parent
JSON_DIR = BASE_DIR / "Inclusion_Criteria_Json_File"
EXCEL_FILE = BASE_DIR / "output_text_extraction.xlsx"
SEMANTIC_DIR = BASE_DIR / "Semantic_Entity_Dictionary"
TRAIN_DATA_FILE = BASE_DIR / "output_train_data.xlsx"
TEST_DATA_FILE = BASE_DIR / "output_test_data.xlsx"
REPORT_FILE = BASE_DIR / "output_result_model_evaluation.txt"


#### 1. Text Cleaning and Extraction with Filename
# Function to clean text
def clean_text(text):
    """
    Extracts text starting after various inclusion criteria indicators and stops before '~exclusion'.
    Cleans unnecessary characters like backslashes and excess spaces.
    """
    pattern = (
        r"(?:"
        r"key inclusion criteria for part \d+(?:, \d+)*:~|"  # Example 1
        r"inclusion criteria:~?|"  # Example 2, 3
        r"inclusion criteria for patients:~|"  # Example 4
        r"\(abbreviated\):~?"  # Example 5
        r")(.*?)(?=~exclusion)"
    )

    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    extracted_text = match.group(1) if match else text  # Fallback to full text if no match
    extracted_text = re.sub(r'\\', '', extracted_text)
    extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
    extracted_text = re.sub(r'^"|"$', '', extracted_text)
    return extracted_text


# Load data from JSON files
def load_json_files(directory):
    data = []
    for file in directory.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            content = json.load(f)
            text = content.get("content", "")  # Ensure correct key
            cleaned_text = clean_text(text)
            data.append((file.name, cleaned_text))
    data.sort(key=lambda x: x[0])
    return data


# Load semantic dictionary
def load_semantics(directory):
    semantics = {}
    for file in directory.glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            semantics[file.stem] = [line.strip() for line in f]
    return semantics


# Prepare dataset for training
def prepare_dataset(json_data, semantics):
    dataset = []
    for filename, text in json_data:
        for semantic, entities in semantics.items():
            for entity in entities:
                if entity in text:
                    dataset.append((filename, semantic, entity))

    df = pd.DataFrame(dataset, columns=["filename", "semantic", "entity"])

    # Sort by filename
    return df.sort_values(by="filename").reset_index(drop=True)


# Train model
def train_ner_model(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Sort by filename while maintaining grouping
    train_data = train_data.sort_values(by=["filename", "semantic"]).reset_index(drop=True)
    test_data = test_data.sort_values(by=["filename", "semantic"]).reset_index(drop=True)

    # Save train and test datasets to Excel
    train_data.to_excel(TRAIN_DATA_FILE, index=False)
    test_data.to_excel(TEST_DATA_FILE, index=False)
    print("Training and test datasets saved.")

    # Prepare training features
    X_train = [[word for word in text.split()] for text in train_data["entity"]]
    y_train = [[semantic] * len(words) for words, semantic in zip(X_train, train_data["semantic"])]

    crf = CRF(algorithm="lbfgs", max_iterations=100, all_possible_transitions=True)
    crf.fit(X_train, y_train)

    return crf, test_data




# Evaluate model
def evaluate_model(model, test_data):
    X_test = [[word for word in text.split()] for text in test_data["entity"]]
    y_test = [[semantic] * len(words) for words, semantic in zip(X_test, test_data["semantic"])]

    y_pred = model.predict(X_test)
    report = flat_classification_report(y_test, y_pred)

    # Save report to a text file
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print("Model evaluation report saved:", REPORT_FILE)
    print(report)


# Main script
if __name__ == "__main__":
    json_data = load_json_files(JSON_DIR)
    df = pd.DataFrame(json_data, columns=["filename", "text"])
    df.to_excel(EXCEL_FILE, index=False)
    print("Dataset saved to", EXCEL_FILE)

    semantics = load_semantics(SEMANTIC_DIR)
    dataset = prepare_dataset(json_data, semantics)

    if not dataset.empty:
        model, test_data = train_ner_model(dataset)
        print("Model trained successfully")
        evaluate_model(model, test_data)
    else:
        print("No valid data for training the model.")

