#################################### COMPLETE COMMAND  #######################

#### extract the right format of input  ✓
#### extract the right semantic_entity  ✓
#### construct the right format of dataframe from start, end, entity, semantic ✓
#### allow displacy to highlight ✓
#### detect all the entity in the displacy highlight ✓

# load the function
import pandas as pd
import re
import spacy
from spacy.matcher import PhraseMatcher
import json
from spacy import displacy
from pathlib import Path
import os

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

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


# Function to clean text using regular expressions
def clean_text(text):
    text = re.sub(r'~exclusion"\r\n', '', text)  # Remove unwanted parts
    text = re.sub(r'~', ' ', text)
    text = re.sub(r'\u003e', '>', text)
    text = re.sub(r'\\', '', text)  # Remove backslash symbol
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'^\"|\"$', '', text)  # Remove leading and trailing quotation marks
    return text

# Function to remove duplicates using lemmatization
def remove_duplicates(matched_entities):
    unique_entities = {category: [] for category in matched_entities}
    for category, entities in matched_entities.items():
        seen = set()
        filtered_entities = []
        for entity in entities:
            lemmatized = " ".join([token.lemma_ for token in nlp(entity)])
            if lemmatized.lower() not in seen:
                seen.add(lemmatized.lower())
                if not any(lemmatized.lower() in e.lower() for e in filtered_entities):
                    filtered_entities.append(entity)
        unique_entities[category] = filtered_entities
    return unique_entities

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
    return remove_duplicates(matched_entities)


# Reset entities and add new ones
def add_custom_entities(doc, matched_entities, cleaned_input):
    doc.set_ents([])  # Reset pre-existing entities
    entities_info = []

    # Collect entity spans and indices
    for category, matched in matched_entities.items():
        for entity in matched:
            start_idx = cleaned_input.find(entity)
            end_idx = start_idx + len(entity)
            entities_info.append((start_idx, end_idx, entity, category))

    # Sort entities by start index
    entities_info.sort(key=lambda x: x[0])

    # Define new spans and prevent overlap
    new_spans = []
    last_end = -1  # To track the end of the previous span

    for start_idx, end_idx, entity, category in entities_info:
        # Ensure no overlap with the previous span
        if start_idx >= last_end:
            span = doc.char_span(start_idx, end_idx, label=category)
            if span:  # Only add valid spans
                new_spans.append(span)
                last_end = end_idx  # Update last_end to the current span's end index

    doc.set_ents(new_spans)  # Update document entities

    # Print added entities
    print("\nAdded Entities:")
    print(f"{'Start':<10} {'End':<10} {'Entity':<50} {'Semantic':<15}")
    for span in new_spans:
        print(f"{span.start_char:<10} {span.end_char:<10} {span.text:<50} {span.label_:<15}")

    return doc

    # Print added entities
    print("\nAdded Entities:")
    print(f"{'Start':<10} {'End':<10} {'Entity':<50} {'Semantic':<15}")
    for span in new_spans:
        print(f"{span.start_char:<10} {span.end_char:<10} {span.text:<50} {span.label_:<15}")
    return doc

# Define file paths and categories
# Get the current script's directory (where "Alzheimer ML NER Project.py" is located)
BASE_DIR = Path(__file__).resolve().parent
# Define the folder path relative to the GitHub repository structure
txt_folder_path = BASE_DIR / "Semantic_Entity_Dictionary"
# Define the semantic categories
semantic_categories = ["caregiver", "condition", "demography", "drug", "measurement", "procedure", "time", "value"]
print(f"Loading files from: {txt_folder_path}")
semantic_categories = ["caregiver", "condition", "demography", "drug", "measurement", "procedure", "time", "value"]

# Load entities from TXT files
entities = load_entities_from_txt(txt_folder_path, semantic_categories)

###### Load JSON content
# Define the base directory (script's parent directory)
BASE_DIR = Path(__file__).resolve().parent
# Define the JSON folder path
json_folder = BASE_DIR / "Inclusion_Criteria_Json_File"
# Define the specific JSON file name
json_file = json_folder / "NCT00141661.json"
# Debugging: Check if the folder exists
if not json_folder.exists():
    print(f"Error: Folder not found -> {json_folder}")
elif not json_file.exists():
    print(f"Error: JSON file not found -> {json_file}")
else:
    print(f"Folder found: {json_folder}")
    print(f"JSON file found: {json_file}")

try:
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)
        input_text = data.get("content", "")
        cleaned_input = clean_text(input_text)

        # Print the content from the JSON file
        print("Content from JSON file:")
        formatted_text = re.sub(r'^\"', '', input_text)  # Removes a leading quotation mark
        formatted_text = re.sub(r'(\.)(\s)', r'.\n',
                                formatted_text)  # Adds a new line after each period followed by a space
        print(formatted_text)  # This will print the "content" field from the JSON with better readability

        # Match entities
        matched_entities = match_entities(cleaned_input, entities)
        doc = nlp(cleaned_input)  # Create spaCy doc

        # Add custom entities and print
        doc = add_custom_entities(doc, matched_entities, cleaned_input)

        # Visualize entities with displacy
        options = {
            "colors": {
                "measurement": "#00BFFF",
                "value": "#FFD700",
                "drug": "#FF6347",
                "condition": "#98FB98",
                "time": "#C8A2C8",
                "demography": "#FFA500",
                "caregiver": "#D2B48C",
                "procedure": "#ADD8E6"
            }
        }
        displacy.serve(doc, style="ent", options=options)

except FileNotFoundError:
    print(f"File not found: {json_file}")
except json.JSONDecodeError:
    print(f"Error decoding JSON: {json_file}")





