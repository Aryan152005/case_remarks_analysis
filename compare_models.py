import pandas as pd
import ollama 
import json
import sys
import os
import time
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import the schema from the separate file
from schema import FIELD_VALUE_SCHEMA

# --- Configuration ---
GROUND_TRUTH_FOLDER = 'data/ground_truth/' # Input folder for text files
# Output base directory for all model runs
BASE_OUTPUT_DIR = 'model_comparison_runs'

REPORT_COLUMN_NAME = 'event_info' # Column name for the report text in the internal dataframe
EVENT_ID_COLUMN_NAME = 'file_name' # Using file name as event ID for reports from folder

# List of models to test and their corresponding Hugging Face tokenizer names
MODELS_TO_TEST = [
    {
        'ollama_name': 'llama3.1:8b',
        'hf_tokenizer_name': "meta-llama/Llama-3.1-8B-Instruct",
        'run_tag': 'llama3_1_8b' # Used for output directory/file naming
    },
    {
        'ollama_name': 'mistral:7b-instruct',
        'hf_tokenizer_name': "mistralai/Mistral-7B-Instruct-v0.2",
        'run_tag': 'mistral_7b_instruct'
    },
    {
        'ollama_name': 'gemma:7b',
        'hf_tokenizer_name': "google/gemma-7b-it",
        'run_tag': 'gemma_7b'
    }
]

# Dynamically get the list of fields from the imported schema
FIELDS = list(FIELD_VALUE_SCHEMA.keys())

# --- Tokenizer Initialization (moved inside main loop for dynamic loading) ---
# We'll load the tokenizer for each model as we iterate through them.

def get_token_count(text: str, current_tokenizer) -> int:
    """Estimates token count using the Hugging Face tokenizer or a fallback."""
    if not text:
        return 0
    if current_tokenizer:
        return len(current_tokenizer.encode(text))
    else:
        # Fallback for rough estimation if tokenizer fails or is not available
        return len(text) // 4

# --- Schema Instructions Generation for Prompt ---
schema_instructions_template = ""
for field, definition in FIELD_VALUE_SCHEMA.items():
    if field == "event_type":
        schema_instructions_template += f"- For `{field}`, select one value from: {', '.join(json.dumps(val) for val in definition)}. If the event type cannot be determined from the report, use \"NULL\".\n"
    elif field == "event_sub_type":
        schema_instructions_template += f"- For `{field}`, select one value from the sub-types associated with the chosen `event_type`. If `event_type` is 'OTHERS', then `event_sub_type` MUST be 'OTHERS'. In this specific case (and ONLY this case), if you can identify a *possible specific event type* in the report that is *not* in our predefined list of `event_type`s, append it in parentheses to the `event_sub_type` like this: `event_sub_type: OTHERS (Possible event type: [inferred type])`. Otherwise, `event_sub_type` remains just 'OTHERS'. If the `event_sub_type` cannot be determined from the report or does not match a valid sub-type for the `event_type`, use \"NULL\".\n"
        schema_instructions_template += "  Detailed sub-types per event type (for your reference in classification):\n"
        for etype, subtypes in FIELD_VALUE_SCHEMA["event_sub_type"].items():
            schema_instructions_template += f"  - {json.dumps(etype)}: {', '.join(json.dumps(val) for val in subtypes)}\n"
    elif isinstance(definition, list): # For categorical fields like state_of_victim, victim_gender
        schema_instructions_template += f"- For `{field}`, select one value from: {', '.join(json.dumps(val) for val in definition)}. If not explicitly stated, use \"not specified\".\n"
    elif definition == "text_allow_not_specified": # For free-form text fields
        schema_instructions_template += f"- For `{field}`, extract the relevant text directly from the report. Keep it concise, one line or one word. If no relevant text is present, use \"not specified\". Do NOT infer or assume beyond the provided text.\n"

def extract_report_data(report_text: str, ollama_model: str, current_tokenizer) -> dict:
    """
    Extracts structured data from a single emergency report using the LLM.
    Handles empty/invalid reports and errors during LLM inference.
    """
    if not report_text or pd.isna(report_text):
        print(f"Skipping empty or invalid report.")
        return {
            field: "NULL" if field in ["event_type", "event_sub_type"] else "not specified"
            for field in FIELDS
        } | {
            '__input_tokens': 0,
            '__output_tokens': 0,
            '__processing_time_sec': 0,
            '__tokens_per_second': 0,
            '__status': 'skipped_empty_report',
            '__error_message': 'Empty or NaN report text'
        }

    prompt_content = f"""You are an extremely meticulous and precise emergency response classifier. Your task is to extract structured information from the provided emergency report by assigning values to ALL of the predefined fields.

    **Crucial Instructions for Accurate and Non-Hallucinatory Extraction:**
    1.  **Extract ONLY information explicitly stated in the emergency report text.** Do NOT infer, guess, or add external knowledge. If a piece of information for a field is not directly present, use the specified default value ("NULL" or "not specified").
    2.  **Strictly adhere to the provided field names and their allowed values/extraction rules.**
    3.  **Output Format:** The output MUST be in the format `field_name: value` for each requested field, one field per line. Ensure ALL {len(FIELDS)} fields are present in your output.
    4.  **Special Handling for `event_type` and `event_sub_type`:**
        * For `event_type`: If it cannot be explicitly determined from the report, its value MUST be "NULL".
        * For `event_sub_type`:
            * If `event_type` is determined to be "OTHERS", then `event_sub_type` MUST also be "OTHERS". In this specific case (and ONLY this case), if you can identify a *possible specific event type* mentioned in the report that is *not* in our predefined list of `event_type`s, append it in parentheses to the `event_sub_type` like this: `event_sub_type: OTHERS (Possible event type: [inferred type])`. Otherwise, `event_sub_type` remains just 'OTHERS'.
            * For all other `event_type`s (that are not "NULL" or "OTHERS"), `event_sub_type` must be selected *only* from the sub-types strictly associated with that chosen `event_type` as per the schema. If no specific sub-type from the allowed list is explicitly clear in the report, use "NULL".
            * If `event_type` itself is "NULL", then `event_sub_type` must also be "NULL".
    5.  **For fields marked "not specified":** If the information for a field is not explicitly present in the emergency report text, its value MUST be "not specified". This applies to categorical fields (like `state_of_victim` when not found) and text fields (like `specified_matter` when not found).
    6.  **Keep values concise and factual.** For text extraction fields, provide the smallest understandable snippet (one line, one word). Avoid explanations, elaborate sentences, or extraneous text.

    **Schema for Field Values:**
    Please adhere to the following schema for allowed values and extraction rules for each field. This is critical for accurate classification.
    {schema_instructions_template}

    Now, analyze the following emergency report carefully:
    \"\"\"{report_text}\"\"\"

    Extract the information below based strictly on the report content and the provided schema.

    Output:
    """

    input_tokens = get_token_count(prompt_content, current_tokenizer)
    start_time = time.perf_counter()
    llm_output = "" # Initialize llm_output

    try:
        response = ollama.chat(model=ollama_model, messages=[{ # <--- Use ollama.chat here
            "role": "user",
            "content": prompt_content
        }], options={
            'temperature': 0.0, # Ensures deterministic output for extraction
            'num_ctx': 8000 # Increased context window
        })
        llm_output = response['message']['content']
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        output_tokens = get_token_count(llm_output, current_tokenizer)
        tokens_per_second = (input_tokens + output_tokens) / processing_time if processing_time > 0 else 0

        extracted_data = {}
        for line in llm_output.strip().split('\n'):
            if ':' in line:
                try:
                    field_name, value = line.split(':', 1)
                    extracted_data[field_name.strip()] = value.strip()
                except ValueError:
                    print(f"Warning: Could not parse line '{line}' from LLM output. Skipping this line.")
                    continue
        
        final_data = {}
        # Post-processing / Validation for all fields
        for field in FIELDS:
            predicted_value = extracted_data.get(field) # Get value from LLM output

            if field == "event_type":
                # Ensure event_type is one of the allowed values or "NULL"
                if predicted_value not in FIELD_VALUE_SCHEMA["event_type"]:
                    final_data[field] = "NULL"
                    if predicted_value is not None:
                        print(f"Warning: LLM returned invalid event_type '{predicted_value}' for {ollama_model}. Forcing to 'NULL'.")
                else:
                    final_data[field] = predicted_value
            
            elif field == "event_sub_type":
                predicted_event_type = final_data.get('event_type') # Use the (potentially validated) event_type
                
                # Rule 1: If event_type is NULL, event_sub_type must be NULL
                if predicted_event_type == "NULL":
                    final_data[field] = "NULL"
                    if predicted_value not in ["NULL", "not specified", None]:
                        print(f"Warning: event_type is NULL for {ollama_model}, but LLM returned event_sub_type '{predicted_value}'. Forcing to 'NULL'.")
                
                # Rule 2: If event_type is OTHERS, event_sub_type must start with 'OTHERS'
                elif predicted_event_type == 'OTHERS':
                    if predicted_value is None or not predicted_value.startswith('OTHERS'):
                        final_data[field] = 'OTHERS' # Default to plain OTHERS
                        if predicted_value is not None:
                            print(f"Warning: event_type is 'OTHERS' for {ollama_model} but event_sub_type '{predicted_value}' is not 'OTHERS'. Forcing to 'OTHERS'.")
                    else:
                        final_data[field] = predicted_value # Keep 'OTHERS (Possible type)' if LLM provided it
                
                # Rule 3: For other valid event_types, validate against its specific sub-types
                elif predicted_event_type in FIELD_VALUE_SCHEMA["event_sub_type"]:
                    allowed_sub_types = FIELD_VALUE_SCHEMA["event_sub_type"][predicted_event_type]
                    if predicted_value not in allowed_sub_types:
                        final_data[field] = "NULL" # If not in allowed list, default to NULL
                        if predicted_value is not None:
                            print(f"Warning: event_sub_type '{predicted_value}' is not valid for event_type '{predicted_event_type}' for {ollama_model}. Forcing to 'NULL'.")
                    else:
                        final_data[field] = predicted_value
                else: # Fallback if event_type somehow invalid but not caught as NULL yet
                    final_data[field] = "NULL"
                    if predicted_value not in ["NULL", "not specified", None]:
                        print(f"Warning: Unhandled event_type '{predicted_event_type}' for {ollama_model}, forcing event_sub_type '{predicted_value}' to 'NULL'.")

            elif isinstance(FIELD_VALUE_SCHEMA[field], list): # Categorical fields like state_of_victim, victim_gender
                if predicted_value not in FIELD_VALUE_SCHEMA[field]:
                    final_data[field] = "not specified"
                    if predicted_value is not None:
                        print(f"Warning: LLM returned invalid value '{predicted_value}' for '{field}' for {ollama_model}. Forcing to 'not specified'.")
                else:
                    final_data[field] = predicted_value
            
            elif FIELD_VALUE_SCHEMA[field] == "text_allow_not_specified": # Free-form text fields
                if predicted_value is None or predicted_value.strip() == "":
                    final_data[field] = "not specified"
                else:
                    final_data[field] = predicted_value.strip()
            else: # Should not happen if schema is well-defined
                final_data[field] = predicted_value if predicted_value is not None else "not specified"


        final_data['__input_tokens'] = input_tokens
        final_data['__output_tokens'] = output_tokens
        final_data['__processing_time_sec'] = processing_time
        final_data['__tokens_per_second'] = tokens_per_second
        final_data['__status'] = 'success'
        final_data['__error_message'] = '' # No error

        return final_data

    except Exception as e:
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        input_tokens_on_error = get_token_count(prompt_content, current_tokenizer)
        output_tokens_on_error = get_token_count(llm_output, current_tokenizer)
        tokens_per_second_on_error = (input_tokens_on_error + output_tokens_on_error) / processing_time if processing_time > 0 else 0

        cleaned_report_preview_on_error = str(report_text)[:70].replace('\n', ' ')
        print(f"An error occurred while processing report for {ollama_model}: '{cleaned_report_preview_on_error}...': {e}")
        
        error_message = str(e).replace('\n', ' ')[:200]

        error_data = {}
        for field in FIELDS:
            if field in ["event_type", "event_sub_type"]:
                error_data[field] = "NULL"
            else:
                error_data[field] = "not specified"

        error_data['__input_tokens'] = input_tokens_on_error
        error_data['__output_tokens'] = output_tokens_on_error
        error_data['__processing_time_sec'] = processing_time
        error_data['__tokens_per_second'] = tokens_per_second_on_error
        error_data['__status'] = 'error'
        error_data['__error_message'] = error_message
        return error_data

def generate_metrics_report(metrics_df: pd.DataFrame, output_dir: str, model_name: str):
    """Generates and saves summary statistics and plots for classification metrics for a specific model."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Generating Metrics Report and Visualizations for {model_name} ---")

    processed_df = metrics_df[metrics_df['__status'] == 'success']

    metrics_summary_file = os.path.join(output_dir, f'{model_name.replace(":", "_")}_classification_metrics_summary.txt')
    processing_time_plot = os.path.join(output_dir, f'{model_name.replace(":", "_")}_processing_time_distribution.png')
    tokens_per_second_plot = os.path.join(output_dir, f'{model_name.replace(":", "_")}_tokens_per_second_distribution.png')
    input_tokens_plot = os.path.join(output_dir, f'{model_name.replace(":", "_")}_input_tokens_distribution.png')
    output_tokens_plot = os.path.join(output_dir, f'{model_name.replace(":", "_")}_output_tokens_distribution.png')
    event_type_counts_plot = os.path.join(output_dir, f'{model_name.replace(":", "_")}_event_type_classification_counts.png')
    event_sub_type_counts_plot = os.path.join(output_dir, f'{model_name.replace(":", "_")}_event_sub_type_classification_counts.png')


    if processed_df.empty:
        print(f"No successful reports for {model_name} to generate detailed performance metrics and plots. Skipping.")
        with open(metrics_summary_file, 'w', encoding='utf-8') as f:
            f.write(f"No successful reports processed for {model_name} to generate detailed metrics.\n")
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
    
    # Save summary statistics to a text file
    with open(metrics_summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Ollama LLM Report Classification Summary for Model: {model_name}\n")
        f.write(f"Date of Report: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n")
        for key, value in summary_stats.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}{value}\n")
        f.write("-" * 50 + "\n")
        
        # Add error details if any
        errored_reports = metrics_df[metrics_df['__status'] == 'error']
        if not errored_reports.empty:
            f.write("\n--- Error Details ---\n")
            error_counts = errored_reports['__error_message'].value_counts()
            f.write(f"Unique Error Messages and Counts:\n{error_counts.to_string()}\n")
            f.write(f"\nExample Errored Reports (first 5):\n")
            for i, row in errored_reports.head(5).iterrows():
                display_report = str(row[REPORT_COLUMN_NAME])[:100].replace('\n', ' ')
                f.write(f"Event ID: {row[EVENT_ID_COLUMN_NAME]}, Error: {row['__error_message']}, Report: {display_report}...\n")

    print(f"Summary statistics saved to: {metrics_summary_file}")

    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    # 1. Processing Time Distribution
    if not processed_df['__processing_time_sec'].empty:
        sns.histplot(processed_df['__processing_time_sec'], kde=True, bins=20)
        plt.title(f'Distribution of Processing Time per Report ({model_name} - seconds)')
        plt.xlabel('Processing Time (seconds)')
        plt.ylabel('Frequency')
        plt.savefig(processing_time_plot)
        plt.clf()
        print(f"Processing time plot saved to: {processing_time_plot}")
    else:
        print("Not enough data to plot Processing Time Distribution.")

    # 2. Tokens per Second Distribution
    if not processed_df['__tokens_per_second'].empty:
        sns.histplot(processed_df['__tokens_per_second'], kde=True, bins=20)
        plt.title(f'Distribution of Tokens per Second ({model_name} - Inference Speed)')
        plt.xlabel('Tokens per Second')
        plt.ylabel('Frequency')
        plt.savefig(tokens_per_second_plot)
        plt.clf()
        print(f"Tokens per second plot saved to: {tokens_per_second_plot}")
    else:
        print("Not enough data to plot Tokens per Second Distribution.")

    # 3. Input Tokens Distribution
    if not processed_df['__input_tokens'].empty:
        sns.histplot(processed_df['__input_tokens'], kde=True, bins=20)
        plt.title(f'Distribution of Input Token Counts ({model_name})')
        plt.xlabel('Input Tokens')
        plt.ylabel('Frequency')
        plt.savefig(input_tokens_plot)
        plt.clf()
        print(f"Input tokens plot saved to: {input_tokens_plot}")
    else:
        print("Not enough data to plot Input Tokens Distribution.")

    # 4. Output Tokens Distribution
    if not processed_df['__output_tokens'].empty:
        sns.histplot(processed_df['__output_tokens'], kde=True, bins=20)
        plt.title(f'Distribution of Output Token Counts ({model_name})')
        plt.xlabel('Output Tokens')
        plt.ylabel('Frequency')
        plt.savefig(output_tokens_plot)
        plt.clf()
        print(f"Output tokens plot saved to: {output_tokens_plot}")
    else:
        print("Not enough data to plot Output Tokens Distribution.")

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
        plt.savefig(event_type_counts_plot)
        plt.clf()
        print(f"Event type counts plot saved to: {event_type_counts_plot}")
    else:
        print("No event type data to plot Classification Counts.")

    # 6. Event Sub-Type Classification Counts (New Plot)
    if 'event_sub_type' in metrics_df.columns and not metrics_df['event_sub_type'].empty:
        plt.figure(figsize=(15, 10))
        normalized_sub_types = metrics_df['event_sub_type'].apply(lambda x: 'OTHERS' if isinstance(x, str) and x.startswith('OTHERS (') else x)
        sub_type_counts = normalized_sub_types[normalized_sub_types.str.upper() != 'NULL']
        sub_type_counts = sub_type_counts[sub_type_counts.index.str.upper() != 'NOT SPECIFIED']

        if not sub_type_counts.empty:
            sub_type_counts = sub_type_counts.value_counts()
            sns.barplot(x=sub_type_counts.index, y=sub_type_counts.values)
            plt.title(f'Count of Classified Event Sub-Types ({model_name} - Excluding NULL/Not Specified)')
            plt.xlabel('Event Sub-Type')
            plt.ylabel('Count')
            plt.xticks(rotation=90, ha='right', fontsize=8)
            plt.tight_layout()
            plt.savefig(event_sub_type_counts_plot)
            plt.clf()
            print(f"Event sub-type counts plot saved to: {event_sub_type_counts_plot}")
        else:
            print("No valid event sub-type data to plot Classification Counts.")
    else:
        print("No event sub-type data to plot Classification Counts.")

def load_reports_from_folder(folder_path: str):
    """
    Loads text reports from a specified folder.
    Each .txt file is treated as a single report.
    """
    reports_data = []
    if not os.path.isdir(folder_path):
        print(f"Error: Input folder '{folder_path}' not found. Please create it and place .txt files inside.")
        sys.exit(1)

    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    if not txt_files:
        print(f"Warning: No .txt files found in '{folder_path}'. Please ensure files are present.")
        return []

    print(f"Found {len(txt_files)} text files in '{folder_path}'.")
    for i, filename in enumerate(txt_files):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                reports_data.append({
                    EVENT_ID_COLUMN_NAME: os.path.splitext(filename)[0],
                    REPORT_COLUMN_NAME: content
                })
        except Exception as e:
            print(f"Error reading file '{file_path}': {e}. Skipping this file.")
    return reports_data


if __name__ == "__main__":
    print(f"--- Ollama LLM Multi-Model Report Classification Comparison ---")
    print(f"Input Folder: {GROUND_TRUTH_FOLDER}")
    print(f"Base Output Directory: {BASE_OUTPUT_DIR}")
    print(f"Report Column Name (Internal): '{REPORT_COLUMN_NAME}'")
    print(f"Event ID Column Name (Internal): '{EVENT_ID_COLUMN_NAME}'")
    print(f"Target Fields: {', '.join(FIELDS)}")

    # Ensure base output directory exists
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(GROUND_TRUTH_FOLDER, exist_ok=True) # Ensure input data directory exists

    # Load reports once
    reports_to_process = load_reports_from_folder(GROUND_TRUTH_FOLDER)
    
    if not reports_to_process:
        print("No reports to process. Exiting.")
        sys.exit(0)

    df_to_process = pd.DataFrame(reports_to_process)

    # Define the order of columns for the output CSV, ensuring all new fields are included
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

    for model_info in MODELS_TO_TEST:
        ollama_model_name = model_info['ollama_name']
        hf_tokenizer_name = model_info['hf_tokenizer_name']
        run_tag = model_info['run_tag']

        current_run_output_dir = os.path.join(BASE_OUTPUT_DIR, run_tag)
        os.makedirs(current_run_output_dir, exist_ok=True)
        
        current_output_csv_file = os.path.join(current_run_output_dir, f'classified_reports_{run_tag}.csv')
        current_output_json_file = os.path.join(current_run_output_dir, f'classified_reports_{run_tag}.json')

        print(f"\n\n=========================================================")
        print(f"--- Starting Processing for Model: {ollama_model_name} ---")
        print(f"  Output CSV: {current_output_csv_file}")
        print(f"  Output JSON: {current_output_json_file}")
        print(f"  Metrics and Plots Directory: {current_run_output_dir}")
        print(f"=========================================================\n")

        # Initialize tokenizer for the current model
        current_tokenizer = None
        print(f"Initializing Hugging Face Tokenizer for '{hf_tokenizer_name}'...")
        try:
            current_tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)
            print("Tokenizer initialized successfully.")
        except Exception as e:
            print(f"Error initializing Hugging Face Tokenizer for {hf_tokenizer_name}: {e}")
            print("Proceeding with a rough character-based token estimation as a fallback.")
            current_tokenizer = None

        # Check if Ollama model is available, pull if not
        try:
            ollama.show(ollama_model_name)
        except ollama.ResponseError:
            print(f"Model '{ollama_model_name}' not found locally. Attempting to pull...")
            try:
                ollama.pull(ollama_model_name)
                print(f"Model '{ollama_model_name}' pulled successfully.")
            except Exception as e:
                print(f"Error pulling model '{ollama_model_name}': {e}. Skipping this model run.")
                continue # Skip to the next model in the list

        all_extracted_records = []
        all_metrics_for_report = []

        print(f"\nStarting classification process for {len(df_to_process)} reports using {ollama_model_name}...")
        for index, row in df_to_process.iterrows():
            event_id = row[EVENT_ID_COLUMN_NAME]
            original_report = row[REPORT_COLUMN_NAME]
            
            cleaned_report_preview = str(original_report)[:70].replace('\n', ' ')
            print(f"\nProcessing report {index + 1}/{len(df_to_process)} (Event ID: {event_id}): '{cleaned_report_preview}...'")

            # Pass the current model and tokenizer to the extraction function
            extracted_record = extract_report_data(original_report, ollama_model_name, current_tokenizer)

            final_record = {
                EVENT_ID_COLUMN_NAME: event_id,
                REPORT_COLUMN_NAME: original_report,
                **extracted_record 
            }
            all_extracted_records.append(final_record)
            all_metrics_for_report.append(extracted_record)

            print(f"  --- Extracted Data for '{event_id}' by {ollama_model_name} ---")
            for field in FIELDS:
                print(f"  {field.replace('_', ' ').title()}: {extracted_record.get(field, 'N/A')}")
            print(f"  Status: {extracted_record.get('__status', 'unknown')}")
            print(f"  Input Tokens (HF): {extracted_record.get('__input_tokens', 0)}")
            print(f"  Output Tokens (HF): {extracted_record.get('__output_tokens', 0)}")
            print(f"  Processing Time: {extracted_record.get('__processing_time_sec', 0):.2f} seconds")
            print(f"  Tokens per Second: {extracted_record.get('__tokens_per_second', 0):.2f}")
            if extracted_record.get('__status') == 'error':
                print(f"  Error Message: {extracted_record.get('__error_message', 'No error message provided.')}")
            print(f"  -------------------------------------")

        # --- Final Save to CSV and JSON for the current model ---
        if all_extracted_records:
            df_final_output = pd.DataFrame(all_extracted_records, columns=output_columns_order)
            
            try:
                df_final_output.to_csv(current_output_csv_file, index=False, encoding='utf-8')
                print(f"\nAll classified reports for {ollama_model_name} saved to CSV: '{current_output_csv_file}'.")
            except Exception as e:
                print(f"Error saving all reports for {ollama_model_name} to CSV: {e}")

            try:
                with open(current_output_json_file, 'w', encoding='utf-8') as f:
                    json.dump(all_extracted_records, f, indent=4, ensure_ascii=False)
                print(f"All classified reports for {ollama_model_name} saved to JSON: '{current_output_json_file}'.")
            except Exception as e:
                print(f"Error saving all reports for {ollama_model_name} to JSON: {e}")
        else:
            print(f"\nNo reports were successfully processed by {ollama_model_name} to save to CSV/JSON.")

        # Generate final metrics report for the current model
        if all_metrics_for_report:
            metrics_df_final = pd.DataFrame(all_metrics_for_report)
            generate_metrics_report(metrics_df_final, current_run_output_dir, ollama_model_name)
        else:
            print(f"\nNo metrics collected for {ollama_model_name}. Skipping final metrics report generation.")

    print(f"\n\n--- Ollama LLM Multi-Model Report Classification Comparison Complete ---")
    print(f"Results for all models are organized in subdirectories under: {BASE_OUTPUT_DIR}")