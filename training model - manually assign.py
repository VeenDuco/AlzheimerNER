### manually assign the entities
### RESULT  in csv : failed to load the expected entity

import pandas as pd
import json
import os
import spacy
from spacy.training.example import Example
import re
from tqdm import tqdm
from pathlib import Path

# 1. Fine-Tune the NER Model

# Load the pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

# Define the labels (your semantic categories)
labels = ["caregiver", "condition", "demography", "drug", "measurement", "procedure", "time", "value"]

# Add the NER component to the pipeline (if not already present)
if "ner" not in nlp.pipe_names:
    ner = nlp.create_pipe("ner")
    nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# Add labels to the NER component
for label in labels:
    ner.add_label(label)

# Prepare training data (replace with your actual labeled data)
TRAIN_DATA = [
    (
    "inclusion criteria:~clinical decline of cognitive ability consistent with mild cognitive impairment~delayed recall score <= 10 on a new york university paragraph recall test~sufficient visual, hearing and communication capabilities and be willing to complete serial standard tests of cognitive function~have a consistent informant to accompany them on scheduled visits~be able to read, write and fully understand the language of the cognitive scales used in the study~exclusion",
    {"entities": [
        (74, 99, "condition"),
        (100, 126, "value"),
        (132, 173, "measurement"),
        (321, 330, "caregiver")
    ]}),

    (
    "inclusion criteria:~diagnosis of probably alzheimer's disease for at least 1 year.~mini mental state exam (mmse) score between 12-26 at screening.~participants must be receiving a cholinesterase inhibitor and/or memantine for at least 4 months, and on a stable dose for at least 2 months.~exclusion",
    {"entities": [
        (2, 34, "condition"),
        (67, 82, "time"),
        (84, 119, "measurement"),
        (128, 133, "value"),
        (181, 205, "drug"),
        (213, 222, "drug"),
        (227, 244, "time")
    ]}),

    # New labeled data
    (
    "recruitment of participants is performed only by study sites.~inclusion criteria:~participants must meet the following inclusion criteria to be eligible.~male or female (age 50 years and older): female must be of non-childbearing potential (i.e. surgically sterilized or at least 2 years post-menopausal).~diagnosis of probable alzheimer's disease based on the national institute of neurological and communicative disorders and stroke/alzheimer's disease and related disorders association (nincds-adrda criteria).~severity of dementia of mild to moderate degree as assessed by the mini mental state examination (mmse) performed at the screening visit.~patient must be living in the community with a reliable caregiver. participant living in an assisted living facility may be included if study medication intake is supervised and participant has a reliable caregiver.~potential participant must be treated with an acetylcholinesterase inhibitor (donepezil, galantamine or rivastigmine) and must be on stable dose for at least 4 months prior to the screening visit and during the entire study period.~participants must not have taken memantine for at least 4 months prior to the commencement of screening. the use of memantine is prohibited during the course of the study.~fluency (oral and written) in the language in which the standardized tests will be administered.~signed informed consent from potential participant or legal representative and caregiver.~exclusion",
    {"entities": [
        (155, 169, "demography"),
        (171, 174, "demography"),
        (175, 193, "value"),
        (196, 305, "condition"),
        (320, 348, "condition"),
        (582, 618, "measurement"),
        (669, 692, "demography"),
        (709, 718, "caregiver"),
        (720, 826, "demography"),
        (947, 956, "drug"),
        (958, 969, "drug"),
        (973, 985, "drug"),
        (1021, 1035, "time"),
        (1119, 1133, "time"),
        (1134, 1143, "drug"),
        (1148, 1165, "time")
    ]}),
]
# Create training examples
examples = [Example.from_dict(nlp.make_doc(text), annot) for text, annot in TRAIN_DATA]

# Fine-tune the model
optimizer = nlp.begin_training()
for i in tqdm(range(100)):  # Number of training iterations
    losses = {}
    for example in examples:
        nlp.update([example], drop=0.5, losses=losses)  # Drop to avoid overfitting
    print(f"Iteration {i} - Losses: {losses}")

# Save the fine-tuned model
nlp.to_disk("Alzheimer_NER_Part3")
print("Model saved as Alzheimer's NER Part 3")

# 2. Use the Model for Entity Extraction

# Load the fine-tuned model
nlp = spacy.load("Alzheimer_NER_Part3")

# Function to clean text using regular expressions
def clean_text(text):
    text = re.sub(r'~exclusion"\r\n', '', text)  # Remove unwanted parts
    text = re.sub(r'~', ' ', text)
    text = re.sub(r'\u003e', '>', text)
    text = re.sub(r'\\', '', text)  # Remove backslash symbol
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'^\"|\"$', '', text)  # Remove leading and trailing quotation marks
    return text


# Function to extract entities from text
def extract_entities(text, model, filename):
    doc = model(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            'start': ent.start_char,
            'end': ent.end_char,
            'semantic': ent.label_,
            'entity': ent.text,
            'filename': filename
        })
    return entities


### Folder containing JSON files
# Get the script's directory
BASE_DIR = Path(__file__).resolve().parent
# Define the JSON folder path
json_folder = BASE_DIR / "Inclusion_Raw_File"
# Debugging: Check if the folder exists
if not json_folder.exists():
    print(f"Error: Folder not found -> {json_folder}")
else:
    print(f"Folder found: {json_folder}, contains: {list(json_folder.iterdir())}")
extracted_data = []

# Process each JSON file
for json_file in os.listdir(json_folder):
    if json_file.endswith(".json"):
        json_path = os.path.join(json_folder, json_file)
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                input_text = data.get("content", "")
                cleaned_input = clean_text(input_text)

                # Extract entities using the fine-tuned NER model
                entities = extract_entities(cleaned_input, nlp, json_file)

                # Add extracted data
                extracted_data.extend(entities)

        except FileNotFoundError:
            print(f"File not found: {json_file}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {json_file}")

# Save the extracted entities to CSV
df = pd.DataFrame(extracted_data)
df.to_csv("extracted_entities_cikai_part_2.csv", index=False)
print("Entities saved to 'extracted_entities_cikai_part_2.csv'")


### result in csv : failed to load the expected entity
