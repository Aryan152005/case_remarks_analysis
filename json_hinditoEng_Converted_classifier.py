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

# Assuming schema.py is in the same directory or accessible via PYTHONPATH
from schema import FIELD_VALUE_SCHEMA, ALL_EVENT_SUB_TYPES, ALL_EVENT_TYPES_AND_SUBTYPES_FLATTENED # New import for better sub-type mapping

GROUND_TRUTH_FOLDER = 'data/ground_truth_eng/' # This path is for loading reports to classify, not ground truth for comparison
OUTPUT_CSV_FILE = 'checkoutput/classified_reports.csv'
OUTPUT_JSON_FILE = 'checkoutput/classified_reports.json'
REPORT_COLUMN_NAME = 'event_info_text'
EVENT_ID_COLUMN_NAME = 'file_name'

# Model Selection
OLLAMA_MODEL = 'llama3.1:8b' # Ollama model name, can be changed to any other available model
HUGGINGFACE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct" # Used for tokenizer

# Output directory for metrics and plots
RUN_OUTPUT_DIR = 'run_output'
os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
METRICS_SUMMARY_FILE = os.path.join(RUN_OUTPUT_DIR, 'classification_metrics_summary.txt')
PROCESSING_TIME_PLOT = os.path.join(RUN_OUTPUT_DIR, 'processing_time_distribution.png')
TOKENS_PER_SECOND_PLOT = os.path.join(RUN_OUTPUT_DIR, 'tokens_per_second_distribution.png')
INPUT_TOKENS_PLOT = os.path.join(RUN_OUTPUT_DIR, 'input_tokens_distribution.png')
OUTPUT_TOKENS_PLOT = os.path.join(RUN_OUTPUT_DIR, 'output_tokens_distribution.png')
EVENT_TYPE_COUNTS_PLOT = os.path.join(RUN_OUTPUT_DIR, 'event_type_classification_counts.png')
EVENT_SUB_TYPE_COUNTS_PLOT = os.path.join(RUN_OUTPUT_DIR, 'event_sub_type_classification_counts.png')

# Initialize tokenizer
print(f"Initializing Hugging Face Tokenizer for '{HUGGINGFACE_MODEL_NAME}'...")
try:
    tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_NAME)
    print("Tokenizer initialized successfully.")
except Exception as e:
    print(f"Error initializing Hugging Face Tokenizer: {e}")
    print("Proceeding with a rough character-based token estimation as a fallback.")
    tokenizer = None

def get_token_count(text: str) -> int:
    """Estimates token count using the Hugging Face tokenizer or a fallback."""
    if not text:
        return 0
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        # Rough estimation: 1 token ~ 4 characters for English
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
        # UPDATED: Use the ALL_EVENT_SUB_TYPES directly as options for the LLM
        # The validation function will handle the mapping to parent types.
        schema_instructions += f"""- For `{field}`, provide a specific sub-type relevant to the incident.
            **Rules:**
            1. MUST choose from the list of ALL possible sub-types:
                {', '.join(ALL_EVENT_SUB_TYPES)}
            2. If 'event_type' is 'OTHERS', then 'event_sub_type' MUST also be 'OTHERS'.
            3. If no suitable sub-type is found, use 'OTHERS'.
            4. NEVER use 'NULL'.\n"""
    elif isinstance(definition, list):
        schema_instructions += f"- For `{field}`, select one value from: {', '.join(definition)}. If not specified, use \"not specified\".\n"
    elif definition == "text_allow_not_specified":
        schema_instructions += f"- For `{field}`, extract the relevant text directly from the report. If none, use \"not specified\".\n"

# --- NEW: Helper function to find parent event type for a sub-type ---
def get_parent_event_type(sub_type: str) -> str:
    """
    Given an event_sub_type, returns its corresponding event_type.
    Uses the ALL_EVENT_TYPES_AND_SUBTYPES_FLATTENED mapping from schema.py.
    Returns 'OTHERS' if no direct parent is found.
    """
    return ALL_EVENT_TYPES_AND_SUBTYPES_FLATTENED.get(sub_type.lower(), 'others').upper()


def validate_classification(extracted_data: dict, original_text: str) -> dict:
    """
    Post-processes the LLM output to enforce schema rules and fix common errors.
    Modified to prioritize event_sub_type for event_type correction.
    """
    # Clean text fields (existing logic, no change needed)
    for field in FIELDS:
        # Ensure values are stripped and lowercased for comparison, but keep original case for final output
        # Handle 'NULL' specifically, map it to 'not specified' or 'OTHERS'
        current_value = str(extracted_data.get(field, '')).strip()
        if current_value.lower() == 'null':
            if field in ['event_type', 'event_sub_type']:
                extracted_data[field] = 'OTHERS'
            else:
                extracted_data[field] = 'not specified'
        else:
            extracted_data[field] = current_value # Keep the casing from LLM for non-special values

    # Convert event_type and event_sub_type to consistent casing for internal validation
    extracted_data['event_type'] = extracted_data.get('event_type', 'OTHERS').upper()
    extracted_data['event_sub_type'] = extracted_data.get('event_sub_type', 'OTHERS').upper()

    # 1. Ensure event_type is valid based on predefined schema
    if extracted_data['event_type'] not in FIELD_VALUE_SCHEMA['event_type']:
        print(f"Invalid event_type '{extracted_data['event_type']}' from LLM. Forcing to 'OTHERS'.")
        extracted_data['event_type'] = 'OTHERS'
        extracted_data['event_sub_type'] = 'OTHERS'
        extracted_data['specified_matter'] = f"Original event_type classification was invalid: {extracted_data['event_type']} - {original_text[:100]}..."

    # 2. Validate and potentially correct event_sub_type and event_type based on the schema
    current_event_type = extracted_data['event_type']
    current_event_sub_type = extracted_data['event_sub_type']
    
    # If the LLM put 'OTHERS' for event_type, force sub-type to 'OTHERS' as per rule
    if current_event_type == 'OTHERS':
        if current_event_sub_type != 'OTHERS':
            print(f"Event type is 'OTHERS' but sub-type is '{current_event_sub_type}'. Forcing sub-type to 'OTHERS'.")
        extracted_data['event_sub_type'] = 'OTHERS'
        return extracted_data # No further validation needed for OTHERS/OTHERS

    # Check if the extracted sub-type is globally valid
    if current_event_sub_type not in ALL_EVENT_SUB_TYPES:
        print(f"Invalid event_sub_type '{current_event_sub_type}' from LLM. Attempting correction.")
        # If sub-type is invalid, default to 'OTHERS' and log
        extracted_data['event_sub_type'] = 'OTHERS'
        if extracted_data['event_type'] != 'OTHERS': # If event_type is not OTHERS, it's still valid
            print(f"Sub-type '{current_event_sub_type}' is unknown. Keeping event_type '{extracted_data['event_type']}' and setting sub-type to 'OTHERS'.")
            # If the original event_type was valid, it's fine.
        else: # Event_type was already OTHERS, so this is consistent
             pass

    # If the sub-type is now 'OTHERS' (either set by LLM or corrected above), it's fine for any event_type
    if extracted_data['event_sub_type'] == 'OTHERS':
        return extracted_data

    # Now, check if the current event_sub_type is valid for the current event_type
    allowed_subtypes_for_current_type = FIELD_VALUE_SCHEMA['event_sub_type'].get(current_event_type, [])
    
    if current_event_sub_type not in allowed_subtypes_for_current_type:
        print(f"Mismatch: Sub-type '{current_event_sub_type}' is not valid for Event Type '{current_event_type}'. Attempting to re-classify event_type based on sub-type.")
        
        # Try to find a new event_type that matches the sub_type
        new_event_type_from_subtype = get_parent_event_type(current_event_sub_type)
        
        if new_event_type_from_subtype.upper() in FIELD_VALUE_SCHEMA['event_type'] and new_event_type_from_subtype.upper() != 'OTHERS':
            print(f"Found a matching event_type '{new_event_type_from_subtype.upper()}' for sub-type '{current_event_sub_type}'. Changing event_type.")
            extracted_data['event_type'] = new_event_type_from_subtype.upper()
        else:
            # If no suitable parent event_type is found for the sub-type, or it's 'OTHERS',
            # then set both to 'OTHERS' to ensure consistency.
            print(f"Could not find a valid parent event_type for sub-type '{current_event_sub_type}'. Setting both to 'OTHERS'.")
            extracted_data['event_type'] = 'OTHERS'
            extracted_data['event_sub_type'] = 'OTHERS'
            extracted_data['specified_matter'] = (
                f"Original classification had a sub-type '{current_event_sub_type}' not matching event type '{current_event_type}', and no clear re-classification possible. "
                f"Original Text: {original_text[:100]}..."
            )
    
    return extracted_data

def extract_report_data(report_text: str) -> dict:
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

    # Updated Prompt: Removed the specific event_sub_type list from prompt as LLM will get ALL_EVENT_SUB_TYPES
    # and validation will handle the type-subtype consistency.
    prompt_content = f"""EMERGENCY CALL CLASSIFICATION TASK:
You are analyzing 112 emergency call transcripts. Extract structured information with these STRICT RULES:

1. For event_type: MUST choose from these EXACT options: {', '.join(FIELD_VALUE_SCHEMA['event_type'])}
2. For event_sub_type: MUST choose from ALL possible sub-types: {', '.join(ALL_EVENT_SUB_TYPES)}.
3. If event_type is 'OTHERS', event_sub_type MUST be 'OTHERS'.
4. NEVER use 'NULL' for any field, use 'not specified' or 'OTHERS' as appropriate.
5. Extract ONLY information explicitly stated in the CALLER'S statements.
6. For text fields: Extract verbatim when possible, otherwise 'not specified'.

SCHEMA RULES:
{schema_instructions}

CALL TRANSCRIPT:
\"\"\"{report_text}\"\"\"

Output MUST be in EXACT format (one field per line):
field_name: value"""

    input_tokens = get_token_count(prompt_content)
    start_time = time.perf_counter()
    llm_output = ""

    try:
        response = chat(
            model=OLLAMA_MODEL,
            messages=[{
                "role": "user",
                "content": prompt_content
            }],
            options={'temperature': 0.2}
        )
        
        llm_output = response['message']['content']
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        output_tokens = get_token_count(llm_output)
        tokens_per_second = (input_tokens + output_tokens) / processing_time if processing_time > 0 else 0

        # Parse LLM output
        extracted_data = {}
        for line in llm_output.strip().split('\n'):
            if ':' in line:
                try:
                    field_name, value = line.split(':', 1)
                    field_name = field_name.strip()
                    if field_name in FIELDS:
                        extracted_data[field_name] = value.strip()
                except ValueError:
                    print(f"Warning: Could not parse line '{line}' from LLM output.")
                    continue
        
        final_data = {}
        for field in FIELDS:
            # Ensure all fields are present, even if LLM missed them
            final_data[field] = extracted_data.get(field, 'not specified')
        
        # Post-process validation - This is crucial for enforcing schema rules
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
        input_tokens_on_error = get_token_count(prompt_content)
        output_tokens_on_error = get_token_count(llm_output) # Attempt to get tokens from partial output
        tokens_per_second_on_error = (input_tokens_on_error + output_tokens_on_error) / processing_time if processing_time > 0 else 0

        error_message = str(e).replace('\n', ' ')[:200]
        print(f"Error processing report: {error_message}")
        
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


def generate_metrics_report(metrics_df: pd.DataFrame, output_dir: str):
    """Generates and saves summary statistics and plots for classification metrics."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Generating Metrics Report and Visualizations ---")

    processed_df = metrics_df[metrics_df['__status'] == 'success']

    if processed_df.empty:
        print("No successful reports to generate detailed performance metrics and plots. Skipping.")
        with open(os.path.join(output_dir, 'classification_metrics_summary.txt'), 'w') as f:
            f.write("No successful reports processed to generate detailed metrics.\n")
            f.write(f"Total reports attempted: {len(metrics_df)}\n")
            f.write(f"Skipped reports: {metrics_df[metrics_df['__status'] == 'skipped_empty_report'].shape[0]}\n")
            f.write(f"Errored reports: {metrics_df[metrics_df['__status'] == 'error'].shape[0]}\n")
        return

    # Calculate summary statistics
    summary_stats = {
        'Total Reports Attempted': len(metrics_df),
        'Reports Successfully Classified': len(processed_df),
        'Reports Skipped (Empty/Invalid)': metrics_df[metrics_df['__status'] == 'skipped_empty_report'].shape[0],
        'Reports with Errors': metrics_df[metrics_df['__status'] == 'error'].shape[0],
        '--- Processing Time (seconds) ---': '',
        'Min Processing Time': processed_df['__processing_time_sec'].min(),
        'Max Processing Time': processed_df['__processing_time_sec'].max(),
        'Average Processing Time': processed_df['__processing_time_sec'].mean(),
        'Median Processing Time': processed_df['__processing_time_sec'].median(),
        'Std Dev Processing Time': processed_df['__processing_time_sec'].std(),
        '--- Tokens per Second ---': '',
        'Min Tokens per Second': processed_df['__tokens_per_second'].min(),
        'Max Tokens per Second': processed_df['__tokens_per_second'].max(),
        'Average Tokens per Second': processed_df['__tokens_per_second'].mean(),
        'Median Tokens per Second': processed_df['__tokens_per_second'].median(),
        'Std Dev Tokens per Second': processed_df['__tokens_per_second'].std(),
        '--- Input Tokens ---': '',
        'Min Input Tokens': processed_df['__input_tokens'].min(),
        'Max Input Tokens': processed_df['__input_tokens'].max(),
        'Average Input Tokens': processed_df['__input_tokens'].mean(),
        'Median Input Tokens': processed_df['__input_tokens'].median(),
        'Std Dev Input Tokens': processed_df['__input_tokens'].std(),
        '--- Output Tokens ---': '',
        'Min Output Tokens': processed_df['__output_tokens'].min(),
        'Max Output Tokens': processed_df['__output_tokens'].max(),
        'Average Output Tokens': processed_df['__output_tokens'].mean(),
        'Median Output Tokens': processed_df['__output_tokens'].median(),
        'Std Dev Output Tokens': processed_df['__output_tokens'].std(),
    }
    

    with open(METRICS_SUMMARY_FILE, 'w') as f:
        f.write(f"Ollama LLM Report Classification Summary for Model: {OLLAMA_MODEL}\n")
        f.write(f"Date of Report: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n")
        for key, value in summary_stats.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}{value}\n")
        f.write("-" * 50 + "\n")
    
        errored_reports = metrics_df[metrics_df['__status'] == 'error']
        if not errored_reports.empty:
            f.write("\n--- Error Details ---\n")
            error_counts = errored_reports['__error_message'].value_counts()
            f.write(f"Unique Error Messages and Counts:\n{error_counts.to_string()}\n")
            f.write(f"\nExample Errored Reports (first 5):\n")
            for i, row in errored_reports.head(5).iterrows():
                f.write(f"Event ID: {row[EVENT_ID_COLUMN_NAME]}, Error: {row['__error_message']}, Report: {str(row[REPORT_COLUMN_NAME])[:100]}...\n")

    print(f"Summary statistics saved to: {METRICS_SUMMARY_FILE}")

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    # 1. Processing Time Distribution
    if not processed_df['__processing_time_sec'].empty:
        sns.histplot(processed_df['__processing_time_sec'], kde=True, bins=20)
        plt.title('Distribution of Processing Time per Report (seconds)')
        plt.xlabel('Processing Time (seconds)')
        plt.ylabel('Frequency')
        plt.savefig(PROCESSING_TIME_PLOT)
        plt.clf()
        print(f"Processing time plot saved to: {PROCESSING_TIME_PLOT}")
    else:
        print("Not enough data to plot Processing Time Distribution.")

    # 2. Tokens per Second Distribution
    if not processed_df['__tokens_per_second'].empty:
        sns.histplot(processed_df['__tokens_per_second'], kde=True, bins=20)
        plt.title('Distribution of Tokens per Second (Inference Speed)')
        plt.xlabel('Tokens per Second')
        plt.ylabel('Frequency')
        plt.savefig(TOKENS_PER_SECOND_PLOT)
        plt.clf()
        print(f"Tokens per second plot saved to: {TOKENS_PER_SECOND_PLOT}")
    else:
        print("Not enough data to plot Tokens per Second Distribution.")

    # 3. Input Tokens Distribution
    if not processed_df['__input_tokens'].empty:
        sns.histplot(processed_df['__input_tokens'], kde=True, bins=20)
        plt.title('Distribution of Input Token Counts')
        plt.xlabel('Input Tokens')
        plt.ylabel('Frequency')
        plt.savefig(INPUT_TOKENS_PLOT)
        plt.clf()
        print(f"Input tokens plot saved to: {INPUT_TOKENS_PLOT}")
    else:
        print("Not enough data to plot Input Tokens Distribution.")

    # 4. Output Tokens Distribution
    if not processed_df['__output_tokens'].empty:
        sns.histplot(processed_df['__output_tokens'], kde=True, bins=20)
        plt.title('Distribution of Output Token Counts')
        plt.xlabel('Output Tokens')
        plt.ylabel('Frequency')
        plt.savefig(OUTPUT_TOKENS_PLOT)
        plt.clf()
        print(f"Output tokens plot saved to: {OUTPUT_TOKENS_PLOT}")
    else:
        print("Not enough data to plot Output Tokens Distribution.")

    # 5. Event Type Classification Counts
    if 'event_type' in metrics_df.columns and not metrics_df['event_type'].empty:
        plt.figure(figsize=(12, 8))
        event_type_counts = metrics_df['event_type'].value_counts()
        sns.barplot(x=event_type_counts.index, y=event_type_counts.values)
        plt.title('Count of Classified Event Types')
        plt.xlabel('Event Type')
        plt.ylabel('Count')
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()
        plt.savefig(EVENT_TYPE_COUNTS_PLOT)
        plt.clf()
        print(f"Event type counts plot saved to: {EVENT_TYPE_COUNTS_PLOT}")
    else:
        print("No event type data to plot Classification Counts.")

    # 6. Event Sub-Type Classification Counts (New Plot)
    if 'event_sub_type' in metrics_df.columns and not metrics_df['event_sub_type'].empty:
        plt.figure(figsize=(15, 10)) # Larger figure for more sub-types
        # Filter out 'NULL' and 'not specified' for better visualization if desired
        sub_type_counts = metrics_df[metrics_df['event_sub_type'].str.upper() != 'NULL']['event_sub_type'].value_counts()
        sub_type_counts = sub_type_counts[sub_type_counts.index.str.upper() != 'NOT SPECIFIED']

        if not sub_type_counts.empty:
            sns.barplot(x=sub_type_counts.index, y=sub_type_counts.values)
            plt.title('Count of Classified Event Sub-Types (Excluding NULL/Not Specified)')
            plt.xlabel('Event Sub-Type')
            plt.ylabel('Count')
            plt.xticks(rotation=90, ha='right', fontsize=8) # Adjust font size if too many labels
            plt.tight_layout()
            plt.savefig(EVENT_SUB_TYPE_COUNTS_PLOT)
            plt.clf()
            print(f"Event sub-type counts plot saved to: {EVENT_SUB_TYPE_COUNTS_PLOT}")
        else:
            print("No valid event sub-type data to plot Classification Counts.")
    else:
        print("No event sub-type data to plot Classification Counts.")


# Defined a function to run the classification
def run_classification_pipeline(input_folder: str, batch_save_interval: int = 2):
    print(f"--- Emergency Call Classification System ---")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Input Folder: {input_folder}") # Use the provided input_folder
    print(f"Output CSV: {OUTPUT_CSV_FILE}")
    print(f"Output JSON: {OUTPUT_JSON_FILE}")
    print(f"Schema: {len(FIELDS)} fields with strict validation")

    # Ensure directories exist
    os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True)
    
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
    ] + FIELDS + [
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
        
        extracted_record = extract_report_data(original_report)
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
                df_current_output.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
                # Save to JSON
                with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
                    json.dump(df_current_output.to_dict(orient='records'), f, indent=4, ensure_ascii=False)
                print(f"Saved {len(all_extracted_records)} classified calls to CSV and JSON.")
            except Exception as e:
                print(f"Error saving results at iteration {index+1}: {e}")

    print("\nProcessing complete.")

    # Generate final metrics report using all collected records
    if all_extracted_records:
        metrics_df_final = pd.DataFrame(all_extracted_records)
        generate_metrics_report(metrics_df_final, RUN_OUTPUT_DIR)


if __name__ == "__main__":
    # If to run it standalone, you can do:
    run_classification_pipeline('data/split_conversations_english', batch_save_interval=2) # Added batch_save_interval
    print("This script is intended to be imported and run by main.py for the full pipeline.")
    print("To run standalone, uncomment the line above and specify an input folder.")