import pandas as pd
from ollama import chat
import json
import sys
import os
import time
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import itertools

# Add parent directory to path to import schema.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Ensure schema.py is in the same directory or properly added to PYTHONPATH
try:
    from schema import FIELD_VALUE_SCHEMA, ALL_EVENT_SUB_TYPES, ALL_CLASSIFICATION_FIELDS
    # Function to derive event_type from event_sub_type
    def derive_event_type(sub_type: str) -> str:
        sub_type_upper = sub_type.upper().strip()
        
        # First, try to match against specific sub-types in the schema, case-insensitively
        for event_type_key, sub_types_list in FIELD_VALUE_SCHEMA['event_sub_type'].items():
            for canonical_sub_type in sub_types_list:
                if sub_type_upper == canonical_sub_type.upper().strip():
                    return event_type_key
        
        # If not a specific type, check if it's explicitly 'OTHERS' or starts with 'OTHERS:'
        if sub_type_upper == 'OTHERS' or sub_type_upper.startswith('OTHERS:'):
            return 'OTHERS'
        
        # If it's neither a specific type (even case-insensitively) nor explicitly 'OTHERS',
        # then it's an unrecognized value. Default to 'OTHERS' as a fallback.
        return 'OTHERS'
except ImportError:
    print("Error: schema.py not found. Please ensure it's in the same directory or accessible via PYTHONPATH.")
    sys.exit(1)

# Import few-shot examples
try:
    from few_shot_examples import get_few_shot_examples_str
except ImportError:
    print("Error: few_shot_examples.py not found. Please ensure it's in the same directory.")
    sys.exit(1)


# --- Global Configuration ---
GROUND_TRUTH_FOLDER = 'data/ground_truth_eng/'
GROUND_TRUTH_ANNOTATIONS_FILE = 'ground_truth_human.json'

BASE_OUTPUT_DIR = 'model_comparison_results'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

REPORT_COLUMN_NAME = 'event_info_text'
EVENT_ID_COLUMN_NAME = 'file_name'

OLLAMA_MODELS_TO_TEST = [
    {'name': 'llama3.1:8b', 'hf_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct'},
]

# Configuration for the standalone classification pipeline (for run_classification_pipeline)
# This will use the first model defined in OLLAMA_MODELS_TO_TEST by default
CLASSIFICATION_MODEL_NAME = OLLAMA_MODELS_TO_TEST[0]['name']
CLASSIFICATION_HF_MODEL_NAME = OLLAMA_MODELS_TO_TEST[0]['hf_name']
CLASSIFICATION_OUTPUT_CSV = 'output/classified_reports_standalone.csv'
CLASSIFICATION_OUTPUT_JSON = 'output/classified_reports_standalone.json'
CLASSIFICATION_METRICS_DIR = 'run_output_standalone' # For metrics of standalone run

tokenizers = {}

def get_token_count(text: str, hf_model_name: str) -> int:
    """Estimates token count using the Hugging Face tokenizer or a fallback."""
    if not text:
        return 0

    if hf_model_name not in tokenizers:
        print(f"Initializing Hugging Face Tokenizer for '{hf_model_name}'...")
        try:
            tokenizers[hf_model_name] = AutoTokenizer.from_pretrained(hf_model_name)
            print(f"Tokenizer for '{hf_model_name}' initialized successfully.")
        except Exception as e:
            print(f"Error initializing Hugging Face Tokenizer for '{hf_model_name}': {e}")
            print("Proceeding with a rough character-based token estimation as a fallback.")
            tokenizers[hf_model_name] = None

    tokenizer_for_model = tokenizers[hf_model_name]

    if tokenizer_for_model:
        return len(tokenizer_for_model.encode(text))
    else:
        return len(text) // 4 # Rough estimation if tokenizer fails

# Use the ALL_CLASSIFICATION_FIELDS from schema.py for iteration
FIELDS_TO_EXTRACT = ALL_CLASSIFICATION_FIELDS

# --- Schema Instructions Generation for Prompt ---
schema_instructions = ""
for field in FIELDS_TO_EXTRACT:
    if field == "event_type":
        # event_type is derived, LLM does not predict it
        continue
    elif field == "event_sub_type":
        schema_instructions += f"""- For `{field}`, provide the MOST SPECIFIC sub-type relevant to the incident.
            **Important Rules:**
            1. You MUST choose one of these exact values: {', '.join(ALL_EVENT_SUB_TYPES)}.
            2. If the incident absolutely and clearly does NOT fit any of these sub-types, you MUST choose 'OTHERS' for `event_sub_type` as a **LAST RESORT**.
            3. If you choose 'OTHERS' for `event_sub_type`, you MUST then generate a **very brief (1-3 words), descriptive phrase** for `event_sub_type` that accurately describes the incident but is NOT one of the predefined sub-types. This descriptive phrase will be the value for `event_sub_type` in this case. Example: "OTHERS: Animal issue" or "OTHERS: Loud noise".
            4. NEVER use 'NULL' or 'not specified' unless 'not specified' is one of the explicit options.
            5. Always try to be specific based on the transcript.\n"""
    elif field == "specified_matter":
        schema_instructions += f"""- For `{field}`, provide a comprehensive summary of the core incident details and any specific relevant information mentioned in the report. If no specific matter, use "not specified".\n"""
    else:
        definition = FIELD_VALUE_SCHEMA.get(field)
        if isinstance(definition, list):
            schema_instructions += f"- For `{field}`, select one value from: {', '.join(definition)}. If not specified, use \"not specified\".\n"
        elif definition == "text_allow_not_specified":
            if field != "specified_matter": # specified_matter is handled above
                schema_instructions += f"- For `{field}`, extract the relevant text directly from the report. If none, use \"not specified\".\n"


def validate_and_correct_classification(extracted_data: dict, original_text: str) -> dict:
    """
    Post-processes the LLM output to enforce schema rules and fix common errors,
    especially concerning 'OTHERS' event_sub_type and event_type derivation.
    """
    data = extracted_data.copy()
    processing_notes = []

    # 1. Ensure all expected fields are present, defaulting to 'not specified'
    # And clean/normalize string fields
    for field in FIELDS_TO_EXTRACT:
        if field not in data or not isinstance(data[field], str):
            data[field] = 'not specified'
        else:
            data[field] = data[field].strip()

    # 2. Process event_sub_type and derive event_type
    predicted_sub_type_raw = data.get('event_sub_type', 'not specified').strip()
    
    # Normalize predicted sub_type for lookup in schema
    normalized_predicted_sub_type_upper = predicted_sub_type_raw.upper()

    # Determine if the predicted sub-type is meant to be 'OTHERS' or a specific type
    # This block was refined in thought to remove aggressive 'OTHERS' assignment based on unmatched strings
    # The derive_event_type function now handles this more precisely.
            
    # Derive event_type based on the corrected/interpreted event_sub_type
    derived_event_type = derive_event_type(predicted_sub_type_raw)
    
    # Enforce event_type based on derived value
    if data['event_type'] != derived_event_type:
        processing_notes.append(f"Derived event_type '{derived_event_type}' based on event_sub_type '{data['event_sub_type']}'.")
        data['event_type'] = derived_event_type

    # Strict handling for 'OTHERS' event_type and event_sub_type
    if data['event_type'] == 'OTHERS':
        # If event_type is 'OTHERS', then event_sub_type should *not* be a predefined specific sub_type
        # It should either be "OTHERS" (the categorical value) or a descriptive phrase.
        if normalized_predicted_sub_type_upper in [st.upper() for st in ALL_EVENT_SUB_TYPES if st.upper() != 'OTHERS']:
             # If a specific sub_type was predicted but event_type is 'OTHERS', force event_sub_type to 'OTHERS'
             processing_notes.append(f"Event type is 'OTHERS', but sub-type '{data['event_sub_type']}' is a specific type. Correcting event_sub_type to 'OTHERS'.")
             data['event_sub_type'] = 'OTHERS'
        elif not predicted_sub_type_raw.strip() or normalized_predicted_sub_type_upper == 'NOT SPECIFIED':
            # If empty or "not specified" when it should be descriptive for 'OTHERS'
            processing_notes.append(f"Event type is 'OTHERS', but event_sub_type is empty/not specified. Setting to 'OTHERS: No specific details provided'.")
            data['event_sub_type'] = 'OTHERS: No specific details provided'

    else: # If event_type is not 'OTHERS'
        # event_sub_type MUST be one of the specific predefined types for that event_type, not 'OTHERS' or a generated phrase
        schema_specific_sub_types_for_this_type = []
        for et_key, sub_types_list in FIELD_VALUE_SCHEMA['event_sub_type'].items():
            if et_key.upper() == data['event_type'].upper():
                schema_specific_sub_types_for_this_type = [st.upper() for st in sub_types_list if st.upper() != 'OTHERS']
                break
        
        if normalized_predicted_sub_type_upper not in schema_specific_sub_types_for_this_type:
            processing_notes.append(f"Event type '{data['event_type']}' requires a specific sub-type, but '{data['event_sub_type']}' is invalid or 'OTHERS'. Correcting to 'not specified'.")
            data['event_sub_type'] = 'not specified' # Or could be corrected to the most common sub-type for that event_type, or default. "not specified" is safer.
        else:
            # Correct case for categorical values
            matched_canonical_value = next((s for s in ALL_EVENT_SUB_TYPES if s.upper() == normalized_predicted_sub_type_upper), data['event_sub_type'])
            if matched_canonical_value != data['event_sub_type']:
                processing_notes.append(f"Corrected case for event_sub_type: '{data['event_sub_type']}' to '{matched_canonical_value}'.")
                data['event_sub_type'] = matched_canonical_value


    # 3. Validate other categorical fields against schema lists
    for field, definition in FIELD_VALUE_SCHEMA.items():
        if field == "event_type" or field == "event_sub_type" or field == "specified_matter": # Already handled, skip
            continue
        if isinstance(definition, list):
            value = data.get(field, 'not specified').strip()
            if value.lower() == 'not specified' and 'Not specified' in definition:
                data[field] = 'Not specified'
            elif value not in definition and value.upper() not in [d.upper() for d in definition]:
                matched_canonical_value = next((d for d in definition if d.upper() == value.upper()), None)
                if matched_canonical_value:
                    processing_notes.append(f"Corrected case for field '{field}': '{value}' to '{matched_canonical_value}'.")
                    data[field] = matched_canonical_value
                else:
                    processing_notes.append(f"Invalid value for field '{field}': '{value}'. Setting to 'not specified'.")
                    data[field] = 'not specified'
        elif definition == "text_allow_not_specified":
            if data[field] == '' or data[field].lower() == 'null':
                data[field] = 'not specified'

    # 4. Update specified_matter with any processing notes
    if processing_notes:
        original_specified_matter = data.get('specified_matter', 'not specified')
        if original_specified_matter.lower() == 'not specified' or not original_specified_matter.strip():
            data['specified_matter'] = "; ".join(processing_notes)
        else:
            data['specified_matter'] = original_specified_matter + "; " + "; ".join(processing_notes)

    # Final pass to ensure all fields are strings (important for CSV export)
    for field in FIELDS_TO_EXTRACT:
        if not isinstance(data.get(field), str):
            data[field] = str(data.get(field))

    return data


def extract_report_data(report_text: str, ollama_model_name: str, hf_model_name: str) -> dict:
    """
    Extracts structured data from emergency call transcripts using the LLM.
    The LLM now only predicts 'event_sub_type' and other fields, 'event_type' is derived.
    """
    if not report_text or pd.isna(report_text):
        print("Skipping empty/invalid report.")
        # Ensure 'specified_matter' is also set to 'not specified' or similar for empty reports
        return {
            field: "OTHERS" if field == "event_type" else ("OTHERS: No report content" if field == "event_sub_type" else "not specified")
            for field in FIELDS_TO_EXTRACT
        } | {
            '__input_tokens': 0,
            '__output_tokens': 0,
            '__processing_time_sec': 0,
            '__tokens_per_second': 0,
            '__status': 'skipped_empty_report',
            '__error_message': 'Empty or NaN report text'
        }

    # IMPORTANT: The prompt now focuses ONLY on event_sub_type, not event_type
    # Also, it expects output in 'field_name: value' format, not JSON.
    prompt_content = f"""EMERGENCY CALL CLASSIFICATION TASK:
You are analyzing 112 emergency call transcripts. Your task is to extract structured information with these STRICT RULES:

1.  **For `event_sub_type`:** Provide the MOST SPECIFIC sub-type relevant to the incident.
    * Choose one of these exact values if applicable: {', '.join(ALL_EVENT_SUB_TYPES)}.
    * If the incident absolutely and clearly does NOT fit any of these sub-types, you MUST choose 'OTHERS' for `event_sub_type` as a **LAST RESORT**.
    * If you choose 'OTHERS' for `event_sub_type`, you MUST then generate a **very brief (1-3 words), descriptive phrase** for `event_sub_type` that accurately describes the incident but is NOT one of the predefined sub-types. This descriptive phrase will be the value for `event_sub_type` in this case. Example: "OTHERS: Animal issue" or "OTHERS: Loud noise".
    * NEVER use 'NULL' or 'not specified' unless 'not specified' is one of the explicit options.
    * Always try to be specific based on the transcript.

2.  **For `specified_matter`:** Provide a comprehensive summary of the core incident details and any specific relevant information mentioned in the report. If no specific matter, use "not specified".

3.  **For all other fields:** Extract information explicitly stated or clearly implied by the CALLER'S statements. Do not infer. If information is not present or cannot be inferred, use "not specified".
4.  **For categorical fields:** Select one of the provided exact options (case-sensitive as listed in the schema instructions below). If not specified, use "not specified".

5.  **For text fields (e.g., `incident_location`, `suspect_description`):** Extract verbatim when possible. If none, use "not specified".

**SCHEMA RULES (and examples of valid values):**
{schema_instructions}

{get_few_shot_examples_str()}

**The Current Input Transcript to Classify:**
\"\"\"{report_text}\"\"\"

**Output MUST be in EXACT format (one field per line, `field_name: value`):**
"""

    input_tokens = get_token_count(prompt_content, hf_model_name)
    start_time = time.perf_counter()
    llm_output = ""
    status = 'error'
    error_message = ''
    processing_time = 0
    output_tokens = 0
    tokens_per_second = 0

    try:
        response = chat(
            model=ollama_model_name,
            messages=[{
                "role": "user",
                "content": prompt_content
            }],
            options={'temperature': 0.2} # Keep temperature low for consistent extraction
        )

        llm_output = response['message']['content']
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        output_tokens = get_token_count(llm_output, hf_model_name)
        tokens_per_second = (input_tokens + output_tokens) / processing_time if processing_time > 0 else 0

        extracted_data = {}
        for line in llm_output.strip().split('\n'):
            if ':' in line:
                try:
                    field_name, value = line.split(':', 1)
                    field_name = field_name.strip()
                    # Only accept fields defined in FIELDS_TO_EXTRACT, and LLM should not output 'event_type'
                    if field_name in FIELDS_TO_EXTRACT and field_name != 'event_type':
                        extracted_data[field_name] = value.strip()
                except ValueError:
                    print(f"Warning: Could not parse line '{line}' from LLM output for {ollama_model_name}. Skipping.")
                    continue
        status = 'success'

        # Initialize final_data with defaults, then update with extracted data
        final_data_raw = {field: 'not specified' for field in FIELDS_TO_EXTRACT}
        for field in extracted_data:
            final_data_raw[field] = extracted_data[field]

        # Apply post-processing and validation which also derives event_type
        final_data = validate_and_correct_classification(final_data_raw, report_text)

    except Exception as e:
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        input_tokens_on_error = get_token_count(prompt_content, hf_model_name)
        output_tokens_on_error = get_token_count(llm_output, hf_model_name)
        tokens_per_second_on_error = (input_tokens_on_error + output_tokens_on_error) / processing_time if processing_time > 0 else 0

        error_message = str(e).replace('\n', ' ')[:200]
        print(f"Error processing report with {ollama_model_name}: {error_message}")

        final_data = {
            field: "OTHERS" if field == "event_type" else ("OTHERS: Classification Error" if field == "event_sub_type" else "not specified")
            for field in FIELDS_TO_EXTRACT
        }
        status = 'error'

    # Add metadata
    final_data.update({
        '__input_tokens': input_tokens,
        '__output_tokens': output_tokens,
        '__processing_time_sec': processing_time,
        '__tokens_per_second': tokens_per_second,
        '__status': status,
        '__error_message': error_message
    })
    return final_data

def load_reports_from_folder(folder_path: str) -> list:
    """
    Loads reports from text files in the specified folder. Each file is considered a single report.
    Returns a list of dictionaries, each with 'file_name' and 'event_info_text'.
    """
    reports = []
    if not os.path.exists(folder_path):
        print(f"Error: Ground truth folder '{folder_path}' does not exist.")
        return reports

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
                reports.append({
                    EVENT_ID_COLUMN_NAME: os.path.splitext(filename)[0],
                    REPORT_COLUMN_NAME: report_content
                })
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    print(f"Loaded {len(reports)} reports from {folder_path}")
    return reports

def load_ground_truth_annotations(file_path: str) -> dict:
    """Loads ground truth annotations from a JSON file."""
    if not os.path.exists(file_path):
        print(f"Warning: Ground truth annotations file '{file_path}' not found. Accuracy metrics will be skipped.")
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Ensure ground truth is a dictionary keyed by file_name for easy lookup
            if isinstance(data, list):
                # If it's a list, convert to a dict, and remove 'file_name' from inner dicts
                # as it will become the key/index
                processed_data = {}
                for item in data:
                    if 'file_name' in item:
                        file_name_key = item['file_name']
                        item_copy = item.copy()
                        del item_copy['file_name'] # Remove it so it's not duplicated as a column later
                        processed_data[file_name_key] = item_copy
                return processed_data
            
            # If it's already a dictionary, check if inner dicts have 'file_name'
            # This is less likely if the JSON is correctly formatted as a dict of records.
            if isinstance(data, dict):
                cleaned_data = {}
                for file_name_key, item in data.items():
                    item_copy = item.copy()
                    if 'file_name' in item_copy and item_copy['file_name'] == file_name_key:
                        del item_copy['file_name'] # Remove if it's redundant
                    cleaned_data[file_name_key] = item_copy
                return cleaned_data
            return data # Fallback, though likely not ideal structure
    except json.JSONDecodeError as e:
        print(f"Error decoding ground truth JSON file '{file_path}': {e}")
        return {}
    except Exception as e:
        print(f"Error loading ground truth annotations from '{file_path}': {e}")
        return {}

def calculate_detailed_metrics(df_predicted: pd.DataFrame, ground_truth_dict: dict, categorical_fields_for_matrix: list) -> dict:
    """
    Calculates detailed accuracy, completeness, and per-field/per-class metrics,
    including confusion matrices for specified categorical fields.
    """
    metrics = {
        "overall_accuracy": 0.0,
        "overall_completeness": 0.0,
        "field_accuracy": {},
        "field_precision_recall_f1": {},
        "confusion_matrices": {}
    }

    if not ground_truth_dict or df_predicted.empty:
        print("Cannot calculate detailed metrics: Missing ground truth or predicted data.")
        return metrics

    # Create gt_df, ensuring 'file_name' is handled correctly.
    # The load_ground_truth_annotations function is updated to remove 'file_name' from inner dicts.
    gt_df = pd.DataFrame.from_dict(ground_truth_dict, orient='index')
    gt_df.index.name = EVENT_ID_COLUMN_NAME # Name the index 'file_name'
    gt_df = gt_df.reset_index() # Convert the index ('file_name') into a column

    # Merge predicted and ground truth dataframes
    # Use an inner merge to only consider reports present in both for evaluation
    merged_df = pd.merge(df_predicted, gt_df, on=EVENT_ID_COLUMN_NAME, suffixes=('_pred', '_gt'))

    evaluated_reports_count = len(merged_df)
    if evaluated_reports_count == 0:
        print("No matching reports found between predicted data and ground truth for evaluation.")
        return metrics

    total_accuracy_fields = 0
    correct_accuracy_fields = 0
    
    total_gt_populated_fields = 0
    llm_populated_fields_matching_gt = 0

    field_correct_counts = defaultdict(int)
    field_total_counts = defaultdict(int)
    
    y_true_dict = defaultdict(list)
    y_pred_dict = defaultdict(list)

    for field in FIELDS_TO_EXTRACT:
        # Prepare for accuracy and completeness
        for index, row in merged_df.iterrows():
            # Check if columns exist before trying to access them
            predicted_value_raw = str(row.get(f'{field}_pred', 'not specified')).strip()
            gt_value_raw = str(row.get(f'{field}_gt', 'not specified')).strip()

            predicted_value = predicted_value_raw.lower() if predicted_value_raw.lower() == 'not specified' else predicted_value_raw
            gt_value = gt_value_raw.lower() if gt_value_raw.lower() == 'not specified' else gt_value_raw

            # Completeness: How many GT-populated fields did the LLM correctly populate?
            is_gt_populated = (gt_value != 'not specified')
            if is_gt_populated:
                total_gt_populated_fields += 1
                is_llm_populated = (predicted_value != 'not specified')
                if is_llm_populated and predicted_value == gt_value:
                    llm_populated_fields_matching_gt += 1

            # Accuracy: How many fields where GT had a specific value did the LLM get correct?
            if gt_value != 'not specified':
                total_accuracy_fields += 1
                field_total_counts[field] += 1
                if predicted_value == gt_value:
                    correct_accuracy_fields += 1
                    field_correct_counts[field] += 1
            
            # For confusion matrix and P/R/F1, collect true and predicted values
            # Include 'not specified' as a class for comprehensive evaluation
            y_true_dict[field].append(gt_value)
            y_pred_dict[field].append(predicted_value)

    metrics["overall_accuracy"] = (correct_accuracy_fields / total_accuracy_fields * 100) if total_accuracy_fields > 0 else 0
    metrics["overall_completeness"] = (llm_populated_fields_matching_gt / total_gt_populated_fields * 100) if total_gt_populated_fields > 0 else 0

    metrics["field_accuracy"] = {
        field: (field_correct_counts[field] / field_total_counts[field] * 100)
        for field in field_total_counts if field_total_counts[field] > 0
    }

    # Calculate per-class precision, recall, f1-score, and confusion matrices
    for field in categorical_fields_for_matrix:
        if field in y_true_dict and len(y_true_dict[field]) > 0:
            # Get all unique labels from both true and predicted sets to form the complete set of labels
            all_labels = sorted(list(set(y_true_dict[field] + y_pred_dict[field])))
            
            # Ensure 'not specified' is always included if it's a possible label and not already there
            if 'not specified' not in all_labels:
                # Check if 'not specified' is a valid option in the schema for this field.
                # It's explicitly allowed for many fields, and implicitly for text fields.
                # For categorical fields, it's often a fallback.
                schema_options = FIELD_VALUE_SCHEMA.get(field)
                if (isinstance(schema_options, list) and 'Not specified' in schema_options) or \
                   (field == 'event_type' and 'OTHERS' in FIELD_VALUE_SCHEMA['event_sub_type']) or \
                   (field == 'event_sub_type' and 'OTHERS' in ALL_EVENT_SUB_TYPES) or \
                   (field == 'specified_matter'): # specified_matter can legitimately be 'not specified'
                    
                    all_labels.append('not specified')
                    all_labels = sorted(list(set(all_labels))) # Re-sort after adding
            
            # For event_type and event_sub_type, ensure all possible schema values are in labels
            if field == 'event_sub_type':
                for st in ALL_EVENT_SUB_TYPES:
                    if st.lower() not in all_labels:
                        all_labels.append(st.lower())
                # Also include 'others' as a literal label, if it's not already
                if 'others' not in all_labels:
                    all_labels.append('others')
                all_labels = sorted(list(set(all_labels))) # Re-sort

            elif field == 'event_type':
                 for et in FIELD_VALUE_SCHEMA['event_sub_type'].keys():
                    if et.lower() not in all_labels:
                        all_labels.append(et.lower())
                    all_labels = sorted(list(set(all_labels))) # Final sort


            # Convert to lower case for consistent comparison
            y_true_lower = [str(x).lower() for x in y_true_dict[field]]
            y_pred_lower = [str(x).lower() for x in y_pred_dict[field]]

            # Ensure all values in y_true_lower and y_pred_lower are present in all_labels.
            # If not, add them, as sklearn's CM expects all labels to be in the provided list.
            for val in set(y_true_lower + y_pred_lower):
                if val not in all_labels:
                    all_labels.append(val)
            all_labels = sorted(all_labels) # Final sort


            # Calculate confusion matrix
            cm = confusion_matrix(y_true_lower, y_pred_lower, labels=all_labels)
            metrics["confusion_matrices"][field] = {
                'matrix': cm.tolist(),
                'labels': all_labels
            }

            # Calculate precision, recall, f1-score per class
            precision, recall, f1, _ = precision_recall_fscore_support(y_true_lower, y_pred_lower, labels=all_labels, average=None, zero_division=0)
        
            field_pr_f1 = {}
            for i, label in enumerate(all_labels):
                field_pr_f1[label] = {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1_score': f1[i]
                }
            metrics["field_precision_recall_f1"][field] = field_pr_f1

    metrics["evaluated_reports_count"] = evaluated_reports_count
    return metrics


def plot_confusion_matrix(cm, classes, title, cmap=plt.cm.Blues, output_path=None):
    """
    This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(max(8, len(classes)*0.8), max(8, len(classes)*0.8))) # Dynamic size based on number of classes
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=6) # Smaller font for numbers

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if output_path:
        plt.savefig(output_path)
    plt.close()


def generate_single_model_metrics_report(model_name: str, metrics_df: pd.DataFrame, ground_truth_dict: dict, output_dir: str):
    """Generates and saves summary statistics, detailed metrics, and plots for a single model."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Generating Metrics Report and Visualizations for {model_name} ---")

    processed_df = metrics_df[metrics_df['__status'] == 'success'].copy()

    # Identify categorical fields for confusion matrices (excluding text fields)
    # Ensure 'event_type' and 'event_sub_type' are always included
    categorical_fields_for_matrix = [f for f, defn in FIELD_VALUE_SCHEMA.items() if isinstance(defn, list)]
    if 'event_type' not in categorical_fields_for_matrix:
        categorical_fields_for_matrix.append('event_type')
    if 'event_sub_type' not in categorical_fields_for_matrix:
        categorical_fields_for_matrix.append('event_sub_type')

    # Exclude text fields that are NOT primarily for classification but detailed text extraction
    # 'specified_matter' can be treated as a text field, but often has specific categories.
    # For now, let's keep only strict categorical ones for CM.
    categorical_fields_for_matrix = [f for f in categorical_fields_for_matrix if f not in ['specified_matter', 'incident_location', 'suspect_description']]
    
    # Calculate detailed metrics including confusion matrices
    detailed_metrics = calculate_detailed_metrics(processed_df, ground_truth_dict, categorical_fields_for_matrix=categorical_fields_for_matrix)

    summary_stats = {
        'Total Reports Attempted': len(metrics_df),
        'Reports Successfully Classified': len(processed_df),
        'Reports Skipped (Empty/Invalid)': metrics_df[metrics_df['__status'] == 'skipped_empty_report'].shape[0],
        'Reports with Errors': metrics_df[metrics_df['__status'] == 'error'].shape[0],
        'Reports Used for Accuracy Calculation': detailed_metrics.get('evaluated_reports_count', 0),
        'Overall Accuracy (%)': detailed_metrics.get('overall_accuracy', np.nan),
        'Overall Completeness (%)': detailed_metrics.get('overall_completeness', np.nan),
        '--- Processing Time (seconds) ---': '',
        'Min Processing Time': processed_df['__processing_time_sec'].min() if not processed_df.empty else np.nan,
        'Max Processing Time': processed_df['__processing_time_sec'].max() if not processed_df.empty else np.nan,
        'Average Processing Time': processed_df['__processing_time_sec'].mean() if not processed_df.empty else np.nan,
        'Median Processing Time': processed_df['__processing_time_sec'].median() if not processed_df.empty else np.nan,
        'Std Dev Processing Time': processed_df['__processing_time_sec'].std() if not processed_df.empty else np.nan,
        '--- Tokens per Second ---': '',
        'Min Tokens per Second': processed_df['__tokens_per_second'].min() if not processed_df.empty else np.nan,
        'Max Tokens per Second': processed_df['__tokens_per_second'].max() if not processed_df.empty else np.nan,
        'Average Tokens per Second': processed_df['__tokens_per_second'].mean() if not processed_df.empty else np.nan,
        'Median Tokens per Second': processed_df['__tokens_per_second'].median() if not processed_df.empty else np.nan,
        'Std Dev Tokens per Second': processed_df['__tokens_per_second'].std() if not processed_df.empty else np.nan,
        '--- Input Tokens ---': '',
        'Min Input Tokens': processed_df['__input_tokens'].min() if not processed_df.empty else np.nan,
        'Max Input Tokens': processed_df['__input_tokens'].max() if not processed_df.empty else np.nan,
        'Average Input Tokens': processed_df['__input_tokens'].mean() if not processed_df.empty else np.nan,
        'Median Input Tokens': processed_df['__input_tokens'].median() if not processed_df.empty else np.nan,
        'Std Dev Input Tokens': processed_df['__input_tokens'].std() if not processed_df.empty else np.nan,
        '--- Output Tokens ---': '',
        'Min Output Tokens': processed_df['__output_tokens'].min() if not processed_df.empty else np.nan,
        'Max Output Tokens': processed_df['__output_tokens'].max() if not processed_df.empty else np.nan,
        'Average Output Tokens': processed_df['__output_tokens'].mean() if not processed_df.empty else np.nan,
        'Median Output Tokens': processed_df['__output_tokens'].median() if not processed_df.empty else np.nan,
        'Std Dev Output Tokens': processed_df['__output_tokens'].std() if not processed_df.empty else np.nan,
    }
    summary_file = os.path.join(output_dir, f'{model_name.replace(":", "_")}_metrics_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"LLM Report Classification Summary for Model: {model_name}\n")
        f.write(f"Date of Report: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n")
        for key, value in summary_stats.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}{value}\n")
        f.write("-" * 50 + "\n")
        f.write("\n--- Field-wise Accuracy (% Correct) ---\n")
        if detailed_metrics.get('field_accuracy'):
            for field, acc in detailed_metrics['field_accuracy'].items():
                f.write(f"{field}: {acc:.2f}%\n")
        else:
            f.write("No field-wise accuracy data available.\n")
        f.write("\n--- Per-Class Precision, Recall, F1-Score (for selected categorical fields) ---\n")
        if detailed_metrics.get('field_precision_recall_f1'):
            for field, metrics_data in detailed_metrics['field_precision_recall_f1'].items():
                f.write(f"\nField: {field}\n")
                for label, scores in metrics_data.items():
                    f.write(f"  {label:<20} | Precision: {scores['precision']:.2f} | Recall: {scores['recall']:.2f} | F1-Score: {scores['f1_score']:.2f}\n")
        else:
            f.write("No per-class metrics available.\n")
        errored_reports = metrics_df[metrics_df['__status'] == 'error']
        if not errored_reports.empty:
            f.write("\n--- Error Details ---\n")
            error_counts = errored_reports['__error_message'].value_counts()
            f.write(f"Unique Error Messages and Counts:\n{error_counts.to_string()}\n")
            f.write(f"\nExample Errored Reports (first 5):\n")
            for i, row in errored_reports.head(5).iterrows():
                f.write(f"Event ID: {row.get(EVENT_ID_COLUMN_NAME, 'N/A')}, Error: {row['__error_message']}, Report: {str(row.get(REPORT_COLUMN_NAME, ''))[:100]}...\n")
    print(f"Summary statistics saved to: {summary_file}")

    # Save all predicted data to a CSV for detailed inspection
    output_csv_path = os.path.join(output_dir, f'{model_name.replace(":", "_")}_predictions.csv')
    metrics_df.to_csv(output_csv_path, index=False)
    print(f"All model predictions saved to: {output_csv_path}")

    sns.set_style("whitegrid")
    plot_data_available = not processed_df.empty
    if plot_data_available:
        # Plot Processing Time Distribution
        if not processed_df['__processing_time_sec'].empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(processed_df['__processing_time_sec'], kde=True, bins=20)
            plt.title(f'Distribution of Processing Time per Report ({model_name})')
            plt.xlabel('Processing Time (seconds)')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_")}_processing_time_distribution.png'))
            plt.close()
        else:
            print(f"Not enough data to plot Processing Time Distribution for {model_name}.")

        # Plot Tokens per Second Distribution
        if not processed_df['__tokens_per_second'].empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(processed_df['__tokens_per_second'], kde=True, bins=20)
            plt.title(f'Distribution of Tokens per Second ({model_name})')
            plt.xlabel('Tokens per Second')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_")}_tokens_per_second_distribution.png'))
            plt.close()
        else:
            print(f"Not enough data to plot Tokens per Second Distribution for {model_name}.")

        # Plot Input Token Counts
        if not processed_df['__input_tokens'].empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(processed_df['__input_tokens'], kde=True, bins=20)
            plt.title(f'Distribution of Input Token Counts ({model_name})')
            plt.xlabel('Input Tokens')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_")}_input_tokens_distribution.png'))
            plt.close()
        else:
            print(f"Not enough data to plot Input Token Counts for {model_name}.")

        # Plot Output Token Counts
        if not processed_df['__output_tokens'].empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(processed_df['__output_tokens'], kde=True, bins=20)
            plt.title(f'Distribution of Output Token Counts ({model_name})')
            plt.xlabel('Output Tokens')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_")}_output_tokens_distribution.png'))
            plt.close()
        else:
            print(f"Not enough data to plot Output Token Counts for {model_name}.")

        # Plot Field-wise Accuracy
        if detailed_metrics.get('field_accuracy'):
            field_accuracy_df = pd.DataFrame(detailed_metrics['field_accuracy'].items(), columns=['Field', 'Accuracy (%)'])
            if not field_accuracy_df.empty:
                plt.figure(figsize=(12, 7))
                sns.barplot(x='Accuracy (%)', y='Field', data=field_accuracy_df.sort_values(by='Accuracy (%)', ascending=False), palette='viridis')
                plt.title(f'Field-wise Accuracy for {model_name}')
                plt.xlim(0, 100)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_")}_field_accuracy.png'))
                plt.close()
            else:
                print(f"Not enough data to plot Field-wise Accuracy for {model_name}.")
        else:
            print(f"No field accuracy data to plot for {model_name}.")

        # Plot Confusion Matrices for relevant fields
        if detailed_metrics.get('confusion_matrices'):
            for field, cm_data in detailed_metrics['confusion_matrices'].items():
                cm = np.array(cm_data['matrix'])
                labels = cm_data['labels']
                plot_confusion_matrix(cm, labels,
                                      title=f'Confusion Matrix for {field} ({model_name})',
                                      output_path=os.path.join(output_dir, f'{model_name.replace(":", "_")}_cm_{field}.png'))
        else:
            print(f"No confusion matrix data to plot for {model_name}.")
    else:
        print(f"No plot data available for {model_name} due to no successful classifications.")

    print(f"--- Finished Metrics Report and Visualizations for {model_name} ---")

def main_pipeline():
    """
    Main pipeline to run the LLM report classification and evaluation.
    This can be used as a standalone script or called as a function.
    """
    print("--- Starting LLM Report Classification and Evaluation Pipeline (Model Comparison) ---")

    # Step 1: Ensure output directory exists
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # Step 2: Load ground truth reports and annotations
    print(f"\nLoading ground truth reports from '{GROUND_TRUTH_FOLDER}'...")
    ground_truth_reports = load_reports_from_folder(GROUND_TRUTH_FOLDER)
    if not ground_truth_reports:
        print("No ground truth reports found. Exiting.")
        sys.exit(0)

    print(f"Loading ground truth annotations from '{GROUND_TRUTH_ANNOTATIONS_FILE}'...")
    ground_truth_annotations = load_ground_truth_annotations(GROUND_TRUTH_ANNOTATIONS_FILE)
    if not ground_truth_annotations:
        print("No ground truth annotations found. Accuracy metrics will be skipped.")

    # Step 3: Run classification for each model
    all_model_results = {}
    for model_config in OLLAMA_MODELS_TO_TEST:
        model_name = model_config['name']
        hf_model_name = model_config['hf_name']
        print(f"\n--- Running classification with model: {model_name} ---")
        
        model_predictions = []
        for report in ground_truth_reports:
            file_name = report[EVENT_ID_COLUMN_NAME]
            report_text = report[REPORT_COLUMN_NAME]
            
            print(f"Classifying report: {file_name}...")
            extracted_data = extract_report_data(report_text, model_name, hf_model_name)
            
            # Add file_name to the extracted data
            extracted_data[EVENT_ID_COLUMN_NAME] = file_name
            model_predictions.append(extracted_data)
        
        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame(model_predictions)
        all_model_results[model_name] = predictions_df
        print(f"Finished classification for model: {model_name}")

    # Step 4: Generate metrics and visualizations for each model
    comparison_data = [] # To store data for overall comparison plots
    for model_name, predictions_df in all_model_results.items():
        model_output_dir = os.path.join(BASE_OUTPUT_DIR, model_name.replace(":", "_"))
        generate_single_model_metrics_report(model_name, predictions_df, ground_truth_annotations, model_output_dir)
        
        # Collect overall metrics for comparative plots
        summary_file_path = os.path.join(model_output_dir, f'{model_name.replace(":", "_")}_metrics_summary.txt')
        if os.path.exists(summary_file_path):
            with open(summary_file_path, 'r') as f:
                summary_content = f.read()
                overall_accuracy = float(summary_content.split('Overall Accuracy (%): ')[1].split('\n')[0].strip()) if 'Overall Accuracy (%)' in summary_content else np.nan
                overall_completeness = float(summary_content.split('Overall Completeness (%): ')[1].split('\n')[0].strip()) if 'Overall Completeness (%)' in summary_content else np.nan
                avg_tokens_per_second = float(summary_content.split('Average Tokens per Second: ')[1].split('\n')[0].strip()) if 'Average Tokens per Second' in summary_content else np.nan
                
                comparison_data.append({
                    'Model': model_name,
                    'Overall Accuracy (%)': overall_accuracy,
                    'Overall Completeness (%)': overall_completeness,
                    'Average Tokens/Second': avg_tokens_per_second
                })

    # Step 5: Generate comparative plots across all models
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print("\n--- Generating Comparative Plots Across Models ---")
        
        # Overall Accuracy
        plt.figure(figsize=(12, 7))
        sns.barplot(x='Model', y='Overall Accuracy (%)', data=comparison_df, palette='viridis')
        plt.title('Overall Accuracy Across Models')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_OUTPUT_DIR, 'overall_accuracy_comparison.png'))
        plt.close()

        # Overall Completeness
        plt.figure(figsize=(12, 7))
        sns.barplot(x='Model', y='Overall Completeness (%)', data=comparison_df, palette='plasma')
        plt.title('Overall Completeness Across Models')
        plt.ylabel('Completeness (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_OUTPUT_DIR, 'overall_completeness_comparison.png'))
        plt.close()

        # Average Tokens/Second
        plt.figure(figsize=(12, 7))
        sns.barplot(x='Model', y='Average Tokens/Second', data=comparison_df, palette='cividis')
        plt.title('Average Tokens per Second Across Models')
        plt.ylabel('Tokens/Second')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_OUTPUT_DIR, 'avg_tokens_per_second_comparison.png'))
        plt.close()

        print("\nComparative plots generated successfully.")
    else:
        print("\nNot enough data to generate comparative plots.")

    print("--- Pipeline Finished ---")


# Defined a function to run the classification for a single model and save results
def run_classification_pipeline(input_folder: str, batch_save_interval: int = 2):
    """
    Runs the emergency call classification pipeline for a single model,
    saves results to CSV/JSON, and generates metrics.
    """
    print(f"--- Emergency Call Classification System (Standalone) ---")
    print(f"Model: {CLASSIFICATION_MODEL_NAME}")
    print(f"Input Folder: {input_folder}")
    print(f"Output CSV: {CLASSIFICATION_OUTPUT_CSV}")
    print(f"Output JSON: {CLASSIFICATION_OUTPUT_JSON}")
    print(f"Schema: {len(ALL_CLASSIFICATION_FIELDS)} fields with strict validation")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(CLASSIFICATION_OUTPUT_CSV), exist_ok=True)
    os.makedirs(CLASSIFICATION_METRICS_DIR, exist_ok=True)
    
    # Load and process reports from the specified input_folder
    reports_to_process = load_reports_from_folder(input_folder) 
    if not reports_to_process:
        print(f"No reports to process in '{input_folder}'. Exiting classification.")
        return

    df_to_process = pd.DataFrame(reports_to_process)
    all_extracted_records = []
    
    output_columns_order = [
        EVENT_ID_COLUMN_NAME,
        REPORT_COLUMN_NAME
    ] + ALL_CLASSIFICATION_FIELDS + [ # Use ALL_CLASSIFICATION_FIELDS
        '__input_tokens',
        '__output_tokens',
        '__processing_time_sec',
        '__tokens_per_second',
        '__status',
        '__error_message'
    ]

    print(f"\nStarting classification for {len(df_to_process)} calls from '{input_folder}'...")
    for index, row in df_to_process.iterrows():
        event_id = row[EVENT_ID_COLUMN_NAME]
        original_report = row[REPORT_COLUMN_NAME]
        
        print(f"Processing call {index + 1}/{len(df_to_process)} (ID: {event_id})")
        
        # Use the specific model configured for standalone runs
        extracted_record = extract_report_data(original_report, CLASSIFICATION_MODEL_NAME, CLASSIFICATION_HF_MODEL_NAME)
        final_record = {
            EVENT_ID_COLUMN_NAME: event_id,
            REPORT_COLUMN_NAME: original_report,
            **extracted_record
        }
        all_extracted_records.append(final_record)

        # --- Batch Saving Logic ---
        if (index + 1) % batch_save_interval == 0 or (index + 1) == len(df_to_process):
            print(f"Saving results after {index + 1} iterations...")
            df_current_output = pd.DataFrame(all_extracted_records, columns=output_columns_order)
            try:
                # Save to CSV
                df_current_output.to_csv(CLASSIFICATION_OUTPUT_CSV, index=False, encoding='utf-8')
                # Save to JSON
                with open(CLASSIFICATION_OUTPUT_JSON, 'w', encoding='utf-8') as f:
                    json.dump(df_current_output.to_dict(orient='records'), f, indent=4, ensure_ascii=False)
                print(f"Saved {len(all_extracted_records)} classified calls to CSV and JSON.")
            except Exception as e:
                print(f"Error saving results at iteration {index+1}: {e}")

    print("\nProcessing complete.")

    # Generate final metrics report using all collected records
    if all_extracted_records:
        metrics_df_final = pd.DataFrame(all_extracted_records)
        
        # Load ground truth annotations for metrics calculation
        ground_truth_annotations_for_metrics = load_ground_truth_annotations(GROUND_TRUTH_ANNOTATIONS_FILE)
        
        generate_single_model_metrics_report(
            CLASSIFICATION_MODEL_NAME,
            metrics_df_final,
            ground_truth_annotations_for_metrics,
            CLASSIFICATION_METRICS_DIR
        )

if __name__ == "__main__":
    # --- Choose ONE option to run ---
    
    # Option 1: Run the main pipeline for model comparison (as originally implemented)
    # main_pipeline() 

    # Option 2: Run the standalone classification pipeline for a single model
    # Specify your input folder for classification. Example: 'data/split_conversations_english'
    run_classification_pipeline('data/ground_truth_eng', batch_save_interval=2) 
    
    # You can comment out the one you don't want to use.
    # print("This script can be run standalone or its functions can be imported.")