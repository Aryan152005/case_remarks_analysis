import pandas as pd
from ollama import chat
import json
import sys
import os
import time
from transformers import AutoTokenizer # Assuming this is installed
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import your schema
# Make sure schema.py is in the same directory or properly added to PYTHONPATH
try:
    from schema import FIELD_VALUE_SCHEMA, ALL_EVENT_SUB_TYPES
except ImportError:
    print("Error: schema.py not found. Please ensure it's in the same directory.")
    sys.exit(1)

# --- Configuration ---
GROUND_TRUTH_FOLDER = 'data/ground_truth_eng/' # Folder containing raw .txt call transcripts
# If you have actual ground truth annotations (manual labels), specify their path
# This is crucial for actual accuracy/correctness calculation.
# Example: 'data/ground_truth_eng_annotations.json'
# Structure: {'file_name_1': {'event_type': '...', 'event_sub_type': '...', ...}, ...}
GROUND_TRUTH_ANNOTATIONS_FILE = 'ground_truth_human.json' # Placeholder

# Output directories
BASE_OUTPUT_DIR = 'model_comparison_results'
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

REPORT_COLUMN_NAME = 'event_info_text'
EVENT_ID_COLUMN_NAME = 'file_name'

# --- Models to Compare ---
# Define the Ollama models you want to test.
# Ensure these models are pulled locally (e.g., ollama pull llama3.1:8b)
OLLAMA_MODELS_TO_TEST = [
    {'name': 'llama3.1:8b', 'hf_name': 'meta-llama/Meta-Llama-3.1-8B-Instruct'},
    {'name': 'llama3:8b', 'hf_name': 'meta-llama/Meta-Llama-3-8B-Instruct'},
    {'name': 'gemma:7b', 'hf_name': 'google/gemma-7b-it'},
    {'name': 'mistral:7b', 'hf_name': 'mistralai/Mistral-7B-Instruct-v0.2'},
    {'name': 'granite:3.2b', 'hf_name': 'ibm-granite/granite-3.2-8b-instruct'},
    {'name': 'phi:3.5', 'hf_name': 'microsoft/Phi-3-mini-128k-instruct'},
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
            tokenizers[hf_model_name] = None # Store None to indicate fallback
    
    tokenizer_for_model = tokenizers[hf_model_name]

    if tokenizer_for_model:
        return len(tokenizer_for_model.encode(text))
    else:
        # Fallback: rough character-based estimation (approx 4 chars per token)
        return len(text) // 4

FIELDS = list(FIELD_VALUE_SCHEMA.keys())

# --- Schema Instructions Generation for Prompt ---
schema_instructions = ""
for field, definition in FIELD_VALUE_SCHEMA.items():
    if field == "event_type":
        schema_instructions += f"""- For `{field}`, classify the primary type of incident from these EXACT options:
            {', '.join(FIELD_VALUE_SCHEMA['event_type'])}.
            
            **Important Rules:**
            1. MUST choose one of these exact values - no variations allowed
            2. If you believe a new category should exist, still choose 'OTHERS' and explain in 'specified_matter'
            3. NEVER use 'NULL' - default to 'OTHERS' if uncertain
            4. Choose the MOST SPECIFIC applicable type\n"""
    elif field == "event_sub_type":
        schema_instructions += f"""- For `{field}`, provide a specific sub-type relevant to the incident:
            **Rules:**
            1. MUST match the selected event_type's sub-types from these options:
                {', '.join(ALL_EVENT_SUB_TYPES)}
            2. If event_type is 'OTHERS', sub-type MUST be 'OTHERS'
            3. If no matching sub-type, use the most generic option for that type
            4. NEVER use 'NULL'\n"""
    elif isinstance(definition, list):
        schema_instructions += f"- For `{field}`, select one value from: {', '.join(definition)}. If not specified, use \"not specified\".\n"
    elif definition == "text_allow_not_specified":
        schema_instructions += f"- For `{field}`, extract the relevant text directly from the report. If none, use \"not specified\".\n"

def validate_classification(extracted_data: dict, original_text: str) -> dict:
    """
    Post-processes the LLM output to enforce schema rules and fix common errors.
    Modified to prioritize event_sub_type for event_type correction.
    """
    # Create a copy to avoid modifying the original dict during iteration if needed later
    data = extracted_data.copy()

    # 1. Ensure event_type is valid based on predefined schema
    if data['event_type'] not in FIELD_VALUE_SCHEMA['event_type']:
        print(f"Invalid event_type '{data['event_type']}'. Forcing to 'OTHERS'.")
        data['event_type'] = 'OTHERS'
        data['event_sub_type'] = 'OTHERS'
        data['specified_matter'] = f"Original classification: {data['event_type']} - {original_text[:100]}..."
    
    # 2. Validate and potentially correct event_sub_type and event_type
    if data['event_type'] == 'OTHERS':
        # If event_type is OTHERS, sub-type must also be OTHERS
        data['event_sub_type'] = 'OTHERS'
    else:
        current_event_type = data['event_type']
        current_event_sub_type = data['event_sub_type']
        
        allowed_subtypes_for_current_type = FIELD_VALUE_SCHEMA['event_sub_type'].get(current_event_type, [])

        potential_parent_types = []
        for etype, subtypes in FIELD_VALUE_SCHEMA['event_sub_type'].items():
            if current_event_sub_type in subtypes:
                potential_parent_types.append(etype)

        if current_event_sub_type not in allowed_subtypes_for_current_type:
            print(f"Mismatch: Sub-type '{current_event_sub_type}' is not valid for Event Type '{current_event_type}'.")

            if potential_parent_types:
                new_event_type = potential_parent_types[0] # Take the first potential parent type
                print(f"Changing event_type from '{current_event_type}' to '{new_event_type}' to match sub-type.")
                data['event_type'] = new_event_type
            else:
                print(f"Sub-type '{current_event_sub_type}' does not belong to any known event_type. Setting both to 'OTHERS'.")
                data['event_type'] = 'OTHERS'
                data['event_sub_type'] = 'OTHERS'
                data['specified_matter'] = (
                    f"Original classification: Event Type '{current_event_type}', Sub-type '{current_event_sub_type}'. "
                    f"Sub-type mismatch and no valid parent event_type found for it."
                )
    
    # 3. Clean text fields (handle 'NULL' and ensure presence of all fields)
    for field in FIELDS:
        # If field is missing or 'NULL', set to 'not specified' or default category
        if field not in data or data.get(field) == 'NULL':
            if field == 'event_type':
                data[field] = 'OTHERS'
            elif field == 'event_sub_type':
                data[field] = 'OTHERS'
            else:
                data[field] = 'not specified'
        # Ensure all fields are strings and strip whitespace
        if isinstance(data.get(field), str):
            data[field] = data[field].strip()
    
    return data

def extract_report_data(report_text: str, ollama_model_name: str, hf_model_name: str) -> dict:
    """
    Extracts structured data from emergency call transcripts using the LLM.
    Strictly follows schema.py definitions.
    """
    if not report_text or pd.isna(report_text):
        print("Skipping empty/invalid report.")
        return {
            field: "OTHERS" if field == "event_type" else ("OTHERS" if field == "event_sub_type" else "not specified")
            for field in FIELDS
        } | {
            '__input_tokens': 0,
            '__output_tokens': 0,
            '__processing_time_sec': 0,
            '__tokens_per_second': 0,
            '__status': 'skipped_empty_report',
            '__error_message': 'Empty or NaN report text'
        }

    prompt_content = f"""EMERGENCY CALL CLASSIFICATION TASK:
You are analyzing 112 emergency call transcripts. Extract structured information with these STRICT RULES:

1. For event_type: MUST choose from these EXACT options: {', '.join(FIELD_VALUE_SCHEMA['event_type'])}
2. For event_sub_type: MUST match the chosen event_type's sub-types
3. If event_type is 'OTHERS', event_sub_type MUST be 'OTHERS' and generate a related event_sub_type
4. NEVER use 'NULL' for any field
5. Extract ONLY information explicitly stated in the CALLER'S statements
6. For text fields: Extract verbatim when possible, otherwise 'not specified'

SCHEMA RULES:
{schema_instructions}

CALL TRANSCRIPT:
\"\"\"{report_text}\"\"\"

Output MUST be in EXACT format (one field per line):
field_name: value"""

    input_tokens = get_token_count(prompt_content, hf_model_name)
    start_time = time.perf_counter()
    llm_output = ""

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

        # Parse LLM output
        extracted_data = {}
        for line in llm_output.strip().split('\n'):
            if ':' in line:
                try:
                    field_name, value = line.split(':', 1)
                    field_name = field_name.strip()
                    # Only accept fields defined in the schema
                    if field_name in FIELDS:
                        extracted_data[field_name] = value.strip()
                except ValueError:
                    print(f"Warning: Could not parse line '{line}' from LLM output.")
                    continue
        
        # Ensure all fields from schema are present, even if not extracted by LLM
        final_data = {}
        for field in FIELDS:
            final_data[field] = extracted_data.get(field, 'not specified') # Default for text fields
            if field in ['event_type', 'event_sub_type']: # Specific defaults for categorical
                if field not in extracted_data or extracted_data[field] == 'NULL':
                    final_data[field] = 'OTHERS'

        # Post-process validation
        final_data = validate_classification(final_data, report_text)
        
        # Add metrics
        final_data.update({
            '__input_tokens': input_tokens,
            '__output_tokens': output_tokens,
            '__processing_time_sec': processing_time,
            '__tokens_per_second': tokens_per_second,
            '__status': 'success',
            '__error_message': ''
        })

        return final_data

    except Exception as e:
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        input_tokens_on_error = get_token_count(prompt_content, hf_model_name)
        output_tokens_on_error = get_token_count(llm_output, hf_model_name)
        tokens_per_second_on_error = (input_tokens_on_error + output_tokens_on_error) / processing_time if processing_time > 0 else 0

        error_message = str(e).replace('\n', ' ')[:200]
        print(f"Error processing report with {ollama_model_name}: {error_message}")
        
        error_data = {
            'event_type': 'OTHERS',
            'event_sub_type': 'OTHERS',
            **{field: 'not specified' for field in FIELDS if field not in ['event_type', 'event_sub_type']},
            '__input_tokens': input_tokens_on_error,
            '__output_tokens': output_tokens_on_error,
            '__processing_time_sec': processing_time,
            '__tokens_per_second': tokens_per_second_on_error,
            '__status': 'error',
            '__error_message': error_message
        }
        return error_data

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
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding ground truth JSON file '{file_path}': {e}")
        return {}
    except Exception as e:
        print(f"Error loading ground truth annotations from '{file_path}': {e}")
        return {}

def calculate_accuracy_metrics(df_predicted: pd.DataFrame, ground_truth: dict) -> dict:
    """
    Calculates accuracy metrics by comparing predicted outputs with ground truth.
    Assumes ground_truth is structured like: {'file_name': {'field1': 'value1', ...}}
    """
    if not ground_truth:
        return {"accuracy_score": np.nan, "completeness_score": np.nan, "field_accuracy": {}}

    total_fields = 0
    correct_fields = 0
    total_expected_fields_found = 0
    
    field_correct_counts = defaultdict(int)
    field_total_counts = defaultdict(int)

    for index, row in df_predicted.iterrows():
        file_name = row[EVENT_ID_COLUMN_NAME]
        if file_name in ground_truth:
            gt_record = ground_truth[file_name]
            
            for field in FIELDS:
                if field in gt_record: # Only evaluate fields that have ground truth
                    total_fields += 1
                    field_total_counts[field] += 1
                    
                    predicted_value = str(row.get(field, 'not specified')).strip().lower()
                    gt_value = str(gt_record.get(field, 'not specified')).strip().lower()

                    if predicted_value == gt_value:
                        correct_fields += 1
                        field_correct_counts[field] += 1
                    
                    if predicted_value != 'not specified' and predicted_value != 'others': # Check if LLM attempted to extract
                         total_expected_fields_found += 1 # A proxy for completeness: how many fields were actually extracted (not 'not specified')

    accuracy_score = correct_fields / total_fields if total_fields > 0 else 0
    # Completeness: Ratio of fields the model *attempted* to fill vs total fields in ground truth
    # This might need refinement based on exact definition of completeness.
    # For now, it's how many non-'not specified'/'others' fields were in predicted data, assuming ground truth was populated.
    completeness_score = total_expected_fields_found / total_fields if total_fields > 0 else 0
    
    field_accuracy = {
        field: field_correct_counts[field] / field_total_counts[field]
        for field in field_correct_counts if field_total_counts[field] > 0
    }
    
    return {
        "overall_accuracy": accuracy_score,
        "overall_completeness": completeness_score,
        "field_accuracy": field_accuracy
    }


def generate_single_model_metrics_report(model_name: str, metrics_df: pd.DataFrame, output_dir: str):
    """Generates and saves summary statistics and plots for a single model."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Generating Metrics Report and Visualizations for {model_name} ---")

    processed_df = metrics_df[metrics_df['__status'] == 'success']

    # Summary statistics
    summary_stats = {
        'Total Reports Attempted': len(metrics_df),
        'Reports Successfully Classified': len(processed_df),
        'Reports Skipped (Empty/Invalid)': metrics_df[metrics_df['__status'] == 'skipped_empty_report'].shape[0],
        'Reports with Errors': metrics_df[metrics_df['__status'] == 'error'].shape[0],
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

    # Write summary to file
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
        
        # Error details
        errored_reports = metrics_df[metrics_df['__status'] == 'error']
        if not errored_reports.empty:
            f.write("\n--- Error Details ---\n")
            error_counts = errored_reports['__error_message'].value_counts()
            f.write(f"Unique Error Messages and Counts:\n{error_counts.to_string()}\n")
            f.write(f"\nExample Errored Reports (first 5):\n")
            for i, row in errored_reports.head(5).iterrows():
                f.write(f"Event ID: {row.get(EVENT_ID_COLUMN_NAME, 'N/A')}, Error: {row['__error_message']}, Report: {str(row.get(REPORT_COLUMN_NAME, ''))[:100]}...\n")
    print(f"Summary statistics saved to: {summary_file}")

    # Generate plots
    sns.set_style("whitegrid")
    
    plot_data_available = not processed_df.empty

    # 1. Processing Time Distribution
    if plot_data_available and not processed_df['__processing_time_sec'].empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(processed_df['__processing_time_sec'], kde=True, bins=20)
        plt.title(f'Distribution of Processing Time per Report ({model_name})')
        plt.xlabel('Processing Time (seconds)')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_")}_processing_time_distribution.png'))
        plt.close()
    else:
        print(f"Not enough data to plot Processing Time Distribution for {model_name}.")

    # 2. Tokens per Second Distribution
    if plot_data_available and not processed_df['__tokens_per_second'].empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(processed_df['__tokens_per_second'], kde=True, bins=20)
        plt.title(f'Distribution of Tokens per Second ({model_name})')
        plt.xlabel('Tokens per Second')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_")}_tokens_per_second_distribution.png'))
        plt.close()
    else:
        print(f"Not enough data to plot Tokens per Second Distribution for {model_name}.")

    # 3. Input Tokens Distribution
    if plot_data_available and not processed_df['__input_tokens'].empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(processed_df['__input_tokens'], kde=True, bins=20)
        plt.title(f'Distribution of Input Token Counts ({model_name})')
        plt.xlabel('Input Tokens')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_")}_input_tokens_distribution.png'))
        plt.close()
    else:
        print(f"Not enough data to plot Input Tokens Distribution for {model_name}.")

    # 4. Output Tokens Distribution
    if plot_data_available and not processed_df['__output_tokens'].empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(processed_df['__output_tokens'], kde=True, bins=20)
        plt.title(f'Distribution of Output Token Counts ({model_name})')
        plt.xlabel('Output Tokens')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_")}_output_tokens_distribution.png'))
        plt.close()
    else:
        print(f"Not enough data to plot Output Tokens Distribution for {model_name}.")

    # 5. Event Type Classification Counts
    if 'event_type' in metrics_df.columns and not metrics_df['event_type'].empty:
        plt.figure(figsize=(12, 8))
        event_type_counts = metrics_df['event_type'].value_counts()
        sns.barplot(x=event_type_counts.index, y=event_type_counts.values)
        plt.title(f'Count of Classified Event Types ({model_name})')
        plt.xlabel('Event Type')
        plt.ylabel('Count')
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name.replace(":", "_")}_event_type_classification_counts.png'))
        plt.close()
    else:
        print(f"No event type data to plot Classification Counts for {model_name}.")

    # 6. Event Sub-Type Classification Counts
    if 'event_sub_type' in metrics_df.columns and not metrics_df['event_sub_type'].empty:
        plt.figure(figsize=(15, 10))
        sub_type_counts = metrics_df[metrics_df['event_sub_type'].str.upper() != 'NULL']['event_sub_type'].value_counts()
        sub_type_counts = sub_type_counts[sub_type_counts.index.str.upper() != 'NOT SPECIFIED']

        if not sub_type_counts.empty:
            sns.barplot(x=sub_type_counts.index, y=sub_type_counts.values)
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


def run_classification_pipeline_for_model(model_config: dict, ground_truth_folder: str, output_base_dir: str, ground_truth_annotations: dict) -> dict:
    """
    Runs the classification pipeline for a single model and returns its results.
    """
    model_name = model_config['name']
    hf_model_name = model_config['hf_name']
    
    model_output_dir = os.path.join(output_base_dir, model_name.replace(":", "_"))
    os.makedirs(model_output_dir, exist_ok=True)

    output_csv_file = os.path.join(model_output_dir, 'classified_reports.csv')
    output_json_file = os.path.join(model_output_dir, 'classified_reports.json')

    print(f"\n--- Running Classification for Model: {model_name} ---")
    print(f"Input Folder: {ground_truth_folder}")
    print(f"Output CSV: {output_csv_file}")
    print(f"Output JSON: {output_json_file}")
    print(f"Schema: {len(FIELDS)} fields with strict validation")

    reports_to_process = load_reports_from_folder(ground_truth_folder) 
    if not reports_to_process:
        print(f"No reports to process in '{ground_truth_folder}'. Skipping this model.")
        return {'model_name': model_name, 'df_results': pd.DataFrame(), 'accuracy_metrics': {}}

    df_to_process = pd.DataFrame(reports_to_process)
    all_extracted_records = []

    output_columns_order = [
        EVENT_ID_COLUMN_NAME,
        REPORT_COLUMN_NAME
    ] + FIELDS + [
        '__input_tokens',
        '__output_tokens',
        '__processing_time_sec',
        '__tokens_per_second',
        '__status',
        '__error_message'
    ]

    print(f"\nStarting classification for {len(df_to_process)} calls from '{ground_truth_folder}' using {model_name}...")
    for index, row in df_to_process.iterrows():
        event_id = row[EVENT_ID_COLUMN_NAME]
        original_report = row[REPORT_COLUMN_NAME]
        
        print(f"Processing call {index + 1}/{len(df_to_process)} (ID: {event_id}) with {model_name}")
        
        extracted_record = extract_report_data(original_report, model_name, hf_model_name)
        final_record = {
            EVENT_ID_COLUMN_NAME: event_id,
            REPORT_COLUMN_NAME: original_report,
            **extracted_record
        }
        all_extracted_records.append(final_record)

    # Save results for this model
    df_model_output = pd.DataFrame(all_extracted_records, columns=output_columns_order)
    try:
        df_model_output.to_csv(output_csv_file, index=False, encoding='utf-8')
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(df_model_output.to_dict(orient='records'), f, indent=4, ensure_ascii=False)
        print(f"\nSuccessfully saved {len(all_extracted_records)} classified calls for {model_name} to CSV and JSON.")
    except Exception as e:
        print(f"Error saving results for {model_name}: {e}")

    # Generate single model metrics report and plots
    metrics_df_for_plots = df_model_output[[col for col in df_model_output.columns if col.startswith('__') or col in FIELDS + [EVENT_ID_COLUMN_NAME, REPORT_COLUMN_NAME]]]
    generate_single_model_metrics_report(model_name, metrics_df_for_plots, model_output_dir)

    # Calculate and return accuracy metrics if ground truth is available
    accuracy_metrics = calculate_accuracy_metrics(df_model_output, ground_truth_annotations)

    return {'model_name': model_name, 'df_results': df_model_output, 'accuracy_metrics': accuracy_metrics}


def generate_comparative_plots(all_model_results: dict, output_dir: str):
    """Generates comparison plots across all models."""
    print("\n--- Generating Cross-Model Comparison Plots ---")
    
    comparison_data = []
    for model_name, res in all_model_results.items():
        if not res['df_results'].empty:
            processed_df = res['df_results'][res['df_results']['__status'] == 'success']
            if not processed_df.empty:
                comparison_data.append({
                    'Model': model_name,
                    'Average Processing Time (s)': processed_df['__processing_time_sec'].mean(),
                    'Average Tokens per Second': processed_df['__tokens_per_second'].mean(),
                    'Average Input Tokens': processed_df['__input_tokens'].mean(),
                    'Average Output Tokens': processed_df['__output_tokens'].mean(),
                    'Overall Accuracy': res['accuracy_metrics'].get('overall_accuracy', np.nan),
                    'Overall Completeness': res['accuracy_metrics'].get('overall_completeness', np.nan),
                    'Successful Classifications (%)': (len(processed_df) / len(res['df_results'])) * 100
                })
    
    if not comparison_data:
        print("No sufficient data to generate comparative plots.")
        return

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison_melted = df_comparison.melt(id_vars='Model', 
                                              value_vars=['Average Processing Time (s)', 'Average Tokens per Second', 
                                                          'Average Input Tokens', 'Average Output Tokens',
                                                          'Overall Accuracy', 'Overall Completeness', 
                                                          'Successful Classifications (%)'],
                                              var_name='Metric', value_name='Value')

    sns.set_style("whitegrid")
    
    # 1. Bar plot for Average Processing Time
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Model', y='Average Processing Time (s)', data=df_comparison)
    plt.title('Average Processing Time per Report Across Models')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_avg_processing_time.png'))
    plt.close()

    # 2. Bar plot for Average Tokens per Second
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Model', y='Average Tokens per Second', data=df_comparison)
    plt.title('Average Tokens per Second Across Models (Inference Speed)')
    plt.ylabel('Tokens/Second')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_avg_tokens_per_second.png'))
    plt.close()

    # 3. Bar plot for Average Input Tokens
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Model', y='Average Input Tokens', data=df_comparison)
    plt.title('Average Input Token Count Across Models')
    plt.ylabel('Tokens')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_avg_input_tokens.png'))
    plt.close()

    # 4. Bar plot for Average Output Tokens
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Model', y='Average Output Tokens', data=df_comparison)
    plt.title('Average Output Token Count Across Models')
    plt.ylabel('Tokens')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_avg_output_tokens.png'))
    plt.close()

    # 5. Bar plot for Overall Accuracy (if ground truth available)
    if 'Overall Accuracy' in df_comparison.columns and not df_comparison['Overall Accuracy'].isnull().all():
        plt.figure(figsize=(12, 7))
        sns.barplot(x='Model', y='Overall Accuracy', data=df_comparison)
        plt.title('Overall Accuracy Across Models (Higher is Better)')
        plt.ylabel('Accuracy Score')
        plt.ylim(0, 1) # Accuracy is between 0 and 1
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_overall_accuracy.png'))
        plt.close()
    else:
        print("Skipping Overall Accuracy plot: Ground truth not available or no valid data.")

    # 6. Bar plot for Overall Completeness (if ground truth available)
    if 'Overall Completeness' in df_comparison.columns and not df_comparison['Overall Completeness'].isnull().all():
        plt.figure(figsize=(12, 7))
        sns.barplot(x='Model', y='Overall Completeness', data=df_comparison)
        plt.title('Overall Completeness Across Models (Higher is Better)')
        plt.ylabel('Completeness Score')
        plt.ylim(0, 1) # Completeness is between 0 and 1
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_overall_completeness.png'))
        plt.close()
    else:
        print("Skipping Overall Completeness plot: Ground truth not available or no valid data.")

    # 7. Bar plot for Successful Classifications Percentage
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Model', y='Successful Classifications (%)', data=df_comparison)
    plt.title('Percentage of Successfully Classified Reports Across Models')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_successful_classifications.png'))
    plt.close()
    
    print(f"All comparative plots saved to: {output_dir}")

    # Optional: Save comparison data to CSV
    df_comparison.to_csv(os.path.join(output_dir, 'model_comparison_summary.csv'), index=False)
    print(f"Model comparison summary saved to: {os.path.join(output_dir, 'model_comparison_summary.csv')}")


def main():
    """Main function to orchestrate the multi-model comparison."""
    print("--- Starting Multi-Model Emergency Call Classification Benchmark ---")

    # Load ground truth annotations once if available
    ground_truth_annotations = load_ground_truth_annotations(GROUND_TRUTH_ANNOTATIONS_FILE)

    all_model_overall_results = {} # To store df_results and accuracy_metrics for each model

    for model_config in OLLAMA_MODELS_TO_TEST:
        model_name = model_config['name']
        print(f"\n{'='*60}\nRunning benchmark for model: {model_name}\n{'='*60}")
        
        try:
            results_for_model = run_classification_pipeline_for_model(
                model_config, 
                GROUND_TRUTH_FOLDER, 
                BASE_OUTPUT_DIR,
                ground_truth_annotations # Pass ground truth for accuracy calculation
            )
            all_model_overall_results[model_name] = results_for_model
            print(f"Finished benchmark for model: {model_name}")
        except Exception as e:
            print(f"Critical error running benchmark for model {model_name}: {e}")
            all_model_overall_results[model_name] = {
                'model_name': model_name,
                'df_results': pd.DataFrame(), # Empty DataFrame on error
                'accuracy_metrics': {"overall_accuracy": np.nan, "completeness_score": np.nan, "field_accuracy": {}}
            }

    print(f"\n{'='*60}\nAll model benchmarks completed.\n{'='*60}")

    # Generate overall comparative plots
    generate_comparative_plots(all_model_overall_results, BASE_OUTPUT_DIR)

    print("\n--- Multi-Model Comparison Report Generation Complete ---")

if __name__ == "__main__":
    main()