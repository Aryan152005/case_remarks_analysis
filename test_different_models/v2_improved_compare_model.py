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
        sub_type_upper = sub_type.upper()
        for event_type_key, sub_types_list in FIELD_VALUE_SCHEMA['event_sub_type'].items():
            if sub_type_upper in [st.upper() for st in sub_types_list]:
                return event_type_key
        return 'OTHERS' # Default if sub_type not found in any category
except ImportError:
    print("Error: schema.py not found. Please ensure it's in the same directory or accessible via PYTHONPATH.")
    sys.exit(1)

# Import few-shot examples
try:
    from few_shot_examples import get_few_shot_examples_str
except ImportError:
    print("Error: few_shot_examples.py not found. Please ensure it's in the same directory.")
    sys.exit(1)


# --- Configuration ---
GROUND_TRUTH_FOLDER = 'data/ground_truth_eng/'
GROUND_TRUTH_ANNOTATIONS_FILE = 'ground_truth_human.json'

BASE_OUTPUT_DIR = 'model_comparison_results'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

REPORT_COLUMN_NAME = 'event_info_text'
EVENT_ID_COLUMN_NAME = 'file_name'

OLLAMA_MODELS_TO_TEST = [
    {'name': 'llama3.1:8b', 'hf_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct'},
    {'name': 'llama3:8b', 'hf_name': 'meta-llama/Meta-Llama-3-8B-Instruct'},
    {'name': 'gemma:7b', 'hf_name': 'google/gemma-7b-it'},
    {'name': 'mistral:7b', 'hf_name': 'mistralai/Mistral-7B-Instruct-v0.2'},
    {'name': 'granite3.2:8b', 'hf_name': 'ibm-granite/granite-3.2-8b-instruct'},
    {'name': 'phi3.5:latest', 'hf_name': 'microsoft/Phi-3-mini-128k-instruct'},
]

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
        schema_instructions += f"""- For `{field}`, provide the MOST SPECIFIC sub-type relevant to the incident from these EXACT options:
            {', '.join(ALL_EVENT_SUB_TYPES)}.
            **Important Rules:**
            1. You MUST choose one of these exact values. No variations allowed.
            2. If the incident does not clearly fit any of these sub-types, you MUST choose 'OTHERS' for event_type and generate a new `event_sub_type` that is related to he infromation provided in the report.
            3. NEVER use 'NULL' or 'not specified' unless 'not specified' is one of the explicit options.
            4. Always try to be specific based on the transcript.\n"""
    elif field == "specified_matter":
        schema_instructions += f"""- For `{field}`, summarize the core incident details. If `event_sub_type` is 'OTHERS', you MUST provide a brief explanation of why it's 'OTHERS' and suggest a more specific descriptive phrase for the incident (e.g., "OTHERS: Caller reports unusual animal behavior not covered by existing categories."). If no specific matter, use "not specified".\n"""
    else:
        definition = FIELD_VALUE_SCHEMA.get(field)
        if isinstance(definition, list):
            schema_instructions += f"- For `{field}`, select one value from: {', '.join(definition)}. If not specified, use \"not specified\".\n"
        elif definition == "text_allow_not_specified":
            # specified_matter is handled above for its special rule.
            if field != "specified_matter":
                schema_instructions += f"- For `{field}`, extract the relevant text directly from the report. If none, use \"not specified\".\n"


def validate_and_correct_classification(extracted_data: dict, original_text: str) -> dict:
    """
    Post-processes the LLM output to enforce schema rules and fix common errors.
    This function now correctly derives 'event_type' from 'event_sub_type'
    and applies strict 'OTHERS' handling as per the provided schema.
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
    normalized_predicted_sub_type = predicted_sub_type_raw.upper()

    # Attempt to find the parent event_type based on the predicted sub_type
    derived_event_type = derive_event_type(predicted_sub_type_raw)
    
    # Correct event_sub_type if it's not found in ALL_EVENT_SUB_TYPES
    # and it's not "OTHERS" (because "OTHERS" is a valid sub_type in the 'OTHERS' category)
    if normalized_predicted_sub_type not in [st.upper() for st in ALL_EVENT_SUB_TYPES]:
        if normalized_predicted_sub_type != 'OTHERS': # If it's a completely unknown sub-type
            processing_notes.append(f"Predicted sub-type '{predicted_sub_type_raw}' not found in schema. Correcting to 'OTHERS'.")
            data['event_sub_type'] = 'OTHERS'
            derived_event_type = 'OTHERS' # If sub_type is corrected to OTHERS, event_type must be OTHERS

    # Enforce event_type based on derived value
    if data['event_type'] != derived_event_type:
        processing_notes.append(f"Derived event_type '{derived_event_type}' based on event_sub_type '{data['event_sub_type']}'.")
        data['event_type'] = derived_event_type

    # Strict handling for 'OTHERS' event_type and event_sub_type
    # If the derived event_type is 'OTHERS', the event_sub_type MUST come from FIELD_VALUE_SCHEMA['OTHERS']
    if data['event_type'] == 'OTHERS':
        # Ensure event_sub_type for OTHERS event_type is from the OTHERS sub_type list
        if data['event_sub_type'].upper() not in [st.upper() for st in FIELD_VALUE_SCHEMA['event_sub_type']['OTHERS']]:
            processing_notes.append(f"Event type is 'OTHERS', but sub-type '{data['event_sub_type']}' is not a valid 'OTHERS' sub-type. Correcting to 'OTHERS'.")
            data['event_sub_type'] = 'OTHERS' # Default to 'OTHERS' within the 'OTHERS' category

    # 3. Validate categorical fields against schema lists
    for field, definition in FIELD_VALUE_SCHEMA.items():
        if field == "event_type" or field == "event_sub_type": # Already handled, skip
            continue
        if isinstance(definition, list):
            value = data.get(field, 'not specified').strip()
            # Special case for 'state_of_victim' and 'victim_gender' where 'Not specified' is a valid option.
            # Make sure it's case-insensitive matching for "not specified" but case-sensitive for other values.
            if value.lower() == 'not specified' and 'Not specified' in definition: # For schema's 'Not specified'
                    data[field] = 'Not specified'
            elif value not in definition and value.upper() not in [d.upper() for d in definition]: # Try case-insensitive
                # Try to find a case-insensitive match in the schema list
                matched_canonical_value = next((d for d in definition if d.upper() == value.upper()), None)
                if matched_canonical_value:
                    processing_notes.append(f"Corrected case for field '{field}': '{value}' to '{matched_canonical_value}'.")
                    data[field] = matched_canonical_value
                else:
                    processing_notes.append(f"Invalid value for field '{field}': '{value}'. Setting to 'not specified'.")
                    data[field] = 'not specified'
        elif definition == "text_allow_not_specified":
            if data[field] == '' or data[field].lower() == 'null': # LLMs sometimes output 'null' or empty string
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
            field: "OTHERS" if field == "event_type" or field == "event_sub_type" else "not specified"
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

1.  **For `event_sub_type`:** You MUST select one of the following exact sub-types:
    {', '.join(ALL_EVENT_SUB_TYPES)}.
    Choose the MOST SPECIFIC sub-type that accurately describes the incident.
    If the incident does not clearly fit any of these sub-types, you MUST choose 'OTHERS'.
    NEVER use 'NULL' or 'not specified' for `event_sub_type` unless 'not specified' was an explicit option provided. Always try to be specific based on the transcript.

2.  **For `specified_matter`:** Summarize the core incident details. If you select 'OTHERS' for `event_sub_type`, you MUST provide a brief explanation of why it is 'OTHERS' and suggest a more specific descriptive phrase for the incident (e.g., "OTHERS: Caller reports unusual animal behavior not covered by existing categories."). If no specific matter, use "not specified".

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
            field: "OTHERS" if field == "event_type" or field == "event_sub_type" else "not specified"
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
    Loads reports from text files in the specified folder.
    Each file is considered a single report.
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
                all_labels = sorted(list(set(all_labels))) # Re-sort

            elif field == 'event_type':
                 for et in FIELD_VALUE_SCHEMA['event_sub_type'].keys():
                    if et.lower() not in all_labels:
                        all_labels.append(et.lower())
                 all_labels = sorted(list(set(all_labels))) # Re-sort


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
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
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
            print(f"Not enough data to plot Input Tokens Distribution for {model_name}.")

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
            print(f"Not enough data to plot Output Tokens Distribution for {model_name}.")

        # Plot Classified Event Types
        if 'event_type' in processed_df.columns and not processed_df['event_type'].empty:
            plt.figure(figsize=(12, 8))
            event_type_counts = processed_df['event_type'].value_counts()
            sns.barplot(x=event_type_counts.index, y=event_type_counts.values, palette='viridis')
            plt.title(f'Count of Classified Event Types ({model_name})')
            plt.xlabel('Event Type')
            plt.ylabel('Count')
            plt.xticks(rotation=90, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_")}_event_type_classification_counts.png'))
            plt.close()
        else:
            print(f"No event type data to plot Classification Counts for {model_name}.")

        # Plot Classified Event Sub-Types
        if 'event_sub_type' in processed_df.columns and not processed_df['event_sub_type'].empty:
            plt.figure(figsize=(15, 10))
            sub_type_counts = processed_df[processed_df['event_sub_type'].str.lower() != 'not specified']['event_sub_type'].value_counts()

            if not sub_type_counts.empty:
                sns.barplot(x=sub_type_counts.index, y=sub_type_counts.values, palette='magma')
                plt.title(f'Count of Classified Event Sub-Types ({model_name})')
                plt.xlabel('Event Sub-Type')
                plt.ylabel('Count')
                plt.xticks(rotation=90, ha='right', fontsize=8)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_")}_event_sub_type_classification_counts.png'))
                plt.close()
            else:
                print(f"No valid event sub-type data to plot Classification Counts for {model_name}.")
        else:
            print(f"No event sub-type data to plot Classification Counts for {model_name}.")

        # Plot Field-wise Accuracy
        if detailed_metrics.get('field_accuracy'):
            field_acc_df = pd.DataFrame(list(detailed_metrics['field_accuracy'].items()), columns=['Field', 'Accuracy'])
            field_acc_df = field_acc_df.sort_values('Accuracy', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='Accuracy', y='Field', data=field_acc_df, palette='cubehelix')
            plt.title(f'Field-wise Accuracy for {model_name} (%)')
            plt.xlabel('Accuracy (%)')
            plt.ylabel('Field')
            plt.xlim(0, 100)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_")}_field_accuracy.png'))
            plt.close()
        else:
            print(f"No field-wise accuracy data to plot for {model_name}.")

        # Plot Confusion Matrices for specified categorical fields
        for field, cm_data in detailed_metrics['confusion_matrices'].items():
            if cm_data['matrix']:
                cm_array = np.array(cm_data['matrix'])
                labels = cm_data['labels']
                plot_confusion_matrix(cm_array, labels,
                                      title=f'Confusion Matrix for {field} ({model_name})',
                                      output_path=os.path.join(output_dir, f'{model_name.replace(":", "_")}_{field}_confusion_matrix.png'))
                print(f"Confusion matrix for '{field}' saved for {model_name}.")
            else:
                print(f"No confusion matrix data for '{field}' for {model_name}.")
    else:
        print(f"No successful classifications to generate plots for {model_name}.")


def main():
    print("Starting LLM Classification Evaluation Script...")

    # Load ground truth reports (text files)
    all_reports = load_reports_from_folder(GROUND_TRUTH_FOLDER)
    if not all_reports:
        print("No reports loaded. Exiting.")
        sys.exit(1)
    reports_df = pd.DataFrame(all_reports)
    print(f"Loaded {len(reports_df)} reports for processing.")

    # Load ground truth annotations (JSON)
    ground_truth_annotations = load_ground_truth_annotations(os.path.join(GROUND_TRUTH_FOLDER, GROUND_TRUTH_ANNOTATIONS_FILE))
    if not ground_truth_annotations:
        print("Warning: No ground truth annotations loaded. Only performance metrics (time, tokens) will be reported, not accuracy.")

    all_models_metrics = {}

    for model_info in OLLAMA_MODELS_TO_TEST:
        model_name = model_info['name']
        hf_model_name = model_info['hf_name']
        print(f"\n--- Processing with Model: {model_name} (HF: {hf_model_name}) ---")
        
        model_output_dir = os.path.join(BASE_OUTPUT_DIR, model_name.replace(":", "_").replace("/", "_"))
        os.makedirs(model_output_dir, exist_ok=True)

        model_results = []
        for index, row in reports_df.iterrows():
            file_name = row[EVENT_ID_COLUMN_NAME]
            report_text = row[REPORT_COLUMN_NAME]
            
            print(f"  Classifying report: {file_name}...")
            extracted_data = extract_report_data(report_text, model_name, hf_model_name)
            
            # Add file_name to the extracted data for easier merging/identification later
            extracted_data[EVENT_ID_COLUMN_NAME] = file_name
            model_results.append(extracted_data)
        
        # Convert results to DataFrame
        model_metrics_df = pd.DataFrame(model_results)
        all_models_metrics[model_name] = model_metrics_df

        # Generate and save reports and plots for the current model
        generate_single_model_metrics_report(model_name, model_metrics_df, ground_truth_annotations, model_output_dir)
    
    print("\n--- All models processed. Generating comparative report (if applicable) ---")
    
    # Optional: Generate a comparative report across all models
    # This part would typically involve loading all individual model results and comparing them.
    # For now, we'll just print a summary of overall accuracies.
    
    if ground_truth_annotations:
        print("\n--- Comparative Overall Accuracy & Completeness ---")
        comparison_data = []
        for model_name, df in all_models_metrics.items():
            processed_df = df[df['__status'] == 'success'].copy()
            # Re-calculate just for comparison, though detailed_metrics also has these
            # Use a dummy list for categorical_fields_for_matrix if not needed for overall comparison
            accuracy_metrics = calculate_detailed_metrics(processed_df, ground_truth_annotations, categorical_fields_for_matrix=[]) 
            comparison_data.append({
                'Model': model_name,
                'Overall Accuracy (%)': accuracy_metrics.get('overall_accuracy', 0.0),
                'Overall Completeness (%)': accuracy_metrics.get('overall_completeness', 0.0),
                'Successful Reports': len(processed_df),
                'Average Processing Time (s)': processed_df['__processing_time_sec'].mean() if not processed_df.empty else np.nan,
                'Average Tokens/Second': processed_df['__tokens_per_second'].mean() if not processed_df.empty else np.nan,
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(by='Overall Accuracy (%)', ascending=False)
        print(comparison_df.to_string(index=False))

        # Save comparative summary to a file
        comparison_summary_file = os.path.join(BASE_OUTPUT_DIR, 'overall_model_comparison_summary.txt')
        with open(comparison_summary_file, 'w') as f:
            f.write("Overall Model Performance Comparison Summary\n")
            f.write(f"Date of Report: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n")
            f.write(comparison_df.to_string(index=False))
        print(f"\nComparative summary saved to: {comparison_summary_file}")

        # Plot comparative bar charts for key metrics
        if not comparison_df.empty:
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

            print("\nComparative plots generated in 'model_comparison_results' directory.")

    print("\nLLM Classification Evaluation Script Finished.")

if __name__ == "__main__":
    main()