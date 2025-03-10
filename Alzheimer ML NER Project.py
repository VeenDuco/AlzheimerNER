#################################### COMPLETE COMMAND  #######################

#### able to read the txt file for respective semantic as data dictionary
#### extract into csv files
#### extract the right format of input  ✓
#### extract the right semantic_entity  ✓
#### construct the right format of dataframe from start, end, entity, semantic ✓
#### ----- PROBLEM ----
# cannot proceed for building model. stuck how to build, train and testing. need assistant to complete the task ✗
# the data extracted in the csv file show some error where it become "date" format. has been troubleshoot a lot


# import packages
import os
import json
import pandas as pd
import re
import spacy
from spacy.matcher import PhraseMatcher
from tqdm import tqdm
from pathlib import path

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


# Function to clean text using regular expressions
def clean_text(text):
    text = re.sub(r'~exclusion"\r\n', '', text)  # Remove unwanted parts
    text = re.sub(r'~', ' ', text)
    text = re.sub(r'\u003e', '>', text)
    text = re.sub(r'\\', '', text)  # Remove backslash symbol
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'^\"|\"$', '', text)  # Remove leading and trailing quotation marks
    return text


# Function to load entities from TXT files into a dictionary
def load_entities_from_txt(txt_folder_path, categories):
    entities = {}
    for category in categories:
        try:
            txt_file_path = os.path.join(txt_folder_path, f"{category}.txt")
            with open(txt_file_path, "r", encoding="utf-8") as file:
                lines = file.readlines()[1:]  # Skip the header line
                entities[category] = [line.strip() for line in lines if line.strip()]
        except Exception as e:
            print(f"Error loading {category}.txt: {e}")
            entities[category] = []
    return entities


# Function to match entities using PhraseMatcher
def match_entities(text, entities):
    doc = nlp(text)
    matched_entities = {category: [] for category in entities}
    for category, entity_list in entities.items():
        matcher = PhraseMatcher(nlp.vocab)
        patterns = [nlp.make_doc(entity) for entity in entity_list]
        matcher.add(category, None, *patterns)

        matches = matcher(doc)
        for _, start, end in matches:
            matched_entities[category].append(doc[start:end].text)
    return matched_entities


# Function to add custom entities, print them dynamically, and store them in a list
def add_custom_entities(doc, matched_entities, cleaned_input, file_name):
    entities_info = []
    seen_entities = {}  # To track entities already added

    # Collect entity spans and indices
    for category, matched in matched_entities.items():
        for entity in matched:
            start_idx = cleaned_input.find(entity)
            end_idx = start_idx + len(entity)

            # Check if this entity has already been added for this category and file
            if category not in seen_entities:
                seen_entities[category] = []

            # Check if this entity overlaps with any previously added entities
            overlap = False
            for prev_entity in seen_entities[category]:
                prev_start, prev_end = prev_entity["Start"], prev_entity["End"]
                # If overlap occurs (start or end positions overlap), skip adding
                if (start_idx < prev_end and end_idx > prev_start):
                    overlap = True
                    break

            if not overlap:
                # Add the entity to the list if it does not overlap
                entities_info.append({
                    "Start": start_idx,
                    "End": end_idx,
                    "Entity": entity,
                    "Semantic": category,
                    "File Name": file_name
                })
                seen_entities[category].append({
                    "Start": start_idx,
                    "End": end_idx,
                    "Entity": entity
                })

    # Print extracted entities dynamically
    print(f"\nExtracted Entities for {file_name}")
    print(f"{'Start':<10} {'End':<10} {'Entity':<50} {'Semantic':<15}")
    for entity_info in entities_info:
        print(
            f"{entity_info['Start']:<10} {entity_info['End']:<10} {entity_info['Entity']:<50} {entity_info['Semantic']:<15}")

    return entities_info


# Define file paths and semantic categories

# Get the current script's directory (where "Alzheimer ML NER Project.py" is located)
BASE_DIR = Path(__file__).resolve().parent
# Define the folder path relative to the GitHub repository structure
txt_folder_path = BASE_DIR / "Inclusion_Criteria_Sematic"  
# Define the semantic categories
semantic_categories = ["caregiver", "condition", "demography", "drug", "measurement", "procedure", "time", "value"]
print(f"Loading files from: {txt_folder_path}")
semantic_categories = ["caregiver", "condition", "demography", "drug", "measurement", "procedure", "time", "value"]

# Load entities from TXT files
entities = load_entities_from_txt(txt_folder_path, semantic_categories)

### Folder containing JSON files
# Get the current script's directory (where "Alzheimer ML NER Project.py" is located)
BASE_DIR = Path(__file__).resolve().parent
# Define the JSON folder path relative to the GitHub repository
json_folder = BASE_DIR / "Inclusion_Raw_File"
print(f"Loading JSON files from: {json_folder}")

extracted_data = []

# Process each JSON file
print("Processing JSON files...")
for json_file in tqdm(os.listdir(json_folder)):
    if json_file.endswith(".json"):
        json_path = os.path.join(json_folder, json_file)
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                input_text = data.get("content", "")
                cleaned_input = clean_text(input_text)

                # Extract file name
                file_name = os.path.splitext(json_file)[0]

                # Match entities
                matched_entities = match_entities(cleaned_input, entities)
                doc = nlp(cleaned_input)  # Create spaCy doc

                # Add custom entities and store in extracted data
                extracted_data.extend(add_custom_entities(doc, matched_entities, cleaned_input, file_name))

        except FileNotFoundError:
            print(f"File not found: {json_file}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {json_file}")

# Convert extracted data to a DataFrame
df = pd.DataFrame(extracted_data)

# Sort by File Name and Start position
df = df.sort_values(by=["File Name", "Start"]).reset_index(drop=True)

# Save the sorted data to a CSV file
output_file = "extracted_entities.csv"
df.to_csv(output_file, index=False)
print(f"Extraction completed. Sorted unique results saved to {output_file}")
