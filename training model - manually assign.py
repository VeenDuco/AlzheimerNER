# training data : assign manually typing
# testing data : take the data from the json files
### Problem : model training and testing give the wrong output

import os
import json
import re
import spacy
import pandas as pd
from spacy.training.example import Example
from tqdm import tqdm
from pathlib import Path

# Load pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

# Define entity labels
LABELS = ["caregiver", "condition", "demography", "drug", "measurement", "procedure", "time", "value"]

# Ensure NER is in the pipeline
if "ner" not in nlp.pipe_names:
    ner = nlp.create_pipe("ner")
    nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Add labels to NER
for label in LABELS:
    ner.add_label(label)

# Training data
TRAIN_DATA = [
    ("inclusion criteria:~clinical decline of cognitive ability consistent with mild cognitive impairment~delayed recall score <= 10 on a new york university paragraph recall test~sufficient visual, hearing and communication capabilities and be willing to complete serial standard tests of cognitive function~have a consistent informant to accompany them on scheduled visits~be able to read, write and fully understand the language of the cognitive scales used in the study~exclusion",
     {"entities": [(74, 99, "condition"), (100, 126, "value"), (132, 173, "measurement"), (321, 330, "caregiver")]}),
    ("inclusion criteria:~diagnosis of probably alzheimer's disease for at least 1 year.~mini mental state exam (mmse) score between 12-26 at screening.~participants must be receiving a cholinesterase inhibitor and/or memantine for at least 4 months, and on a stable dose for at least 2 months.~exclusion",
     {"entities": [(2, 34, "condition"), (67, 82, "time"), (84, 119, "measurement"), (128, 133, "value"), (181, 205, "drug"), (213, 222, "drug"), (227, 244, "time")]}),
]

# Convert training data into spaCy examples
examples = [Example.from_dict(nlp.make_doc(text), annot) for text, annot in TRAIN_DATA]

# Train the model
optimizer = nlp.begin_training()
for epoch in tqdm(range(100), desc="Training model"):
    losses = {}
    for example in examples:
        nlp.update([example], drop=0.5, losses=losses)
    print(f"Epoch {epoch+1}: Losses {losses}")

# Save the fine-tuned model
model_path = "Alzheimer_NER_Part3"
nlp.to_disk(model_path)
print(f"Model saved to {model_path}")

# Load fine-tuned model
nlp = spacy.load(model_path)

# Function to clean text
def clean_text(text):
    text = re.sub(r'~exclusion"\r\n', '', text)
    text = re.sub(r'~', ' ', text)
    text = re.sub(r'\u003e', '>', text)
    text = re.sub(r'\\', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'^"|"$', '', text)
    return text

# Function to extract entities
def extract_entities(text, model, filename):
    doc = model(text)
    return [{
        'start': ent.start_char,
        'end': ent.end_char,
        'semantic': ent.label_,
        'entity': ent.text,
        'filename': filename
    } for ent in doc.ents]

# Define JSON input directory
BASE_DIR = Path(__file__).resolve().parent
json_folder = BASE_DIR / "Inclusion_Criteria_Json_File"

if not json_folder.exists():
    print(f"Error: Folder not found -> {json_folder}")
    exit(1)

print(f"Processing files in: {json_folder}")

# Process JSON files
extracted_data = []
for json_file in json_folder.glob("*.json"):
    try:
        with json_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
            cleaned_text = clean_text(data.get("content", ""))
            extracted_data.extend(extract_entities(cleaned_text, nlp, json_file.name))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error processing {json_file.name}: {e}")

# Convert to DataFrame
if extracted_data:
    df = pd.DataFrame(extracted_data).sort_values(by=["filename", "start"]).reset_index(drop=True)
    output_file = BASE_DIR / "output_training_model_manually.xlsx"
    df.to_excel(output_file, index=False, engine="openpyxl")
    print(f"Extraction completed. Results saved to {output_file}")
else:
    print("Warning: No entities extracted. Check input data and model performance.")


### result in csv : failed to load the expected entity
