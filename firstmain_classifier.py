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

# Import the schema from the separate file
from schema import FIELD_VALUE_SCHEMA, ALL_EVENT_SUB_TYPES

# --- Configuration ---
INPUT_CSV_FILE = 'data/combined_event_data.csv'
OUTPUT_CSV_FILE = 'output/granite_structured_reports_v2.csv' # Changed output filename to indicate new version
REPORT_COLUMN_NAME = 'event_info' # Updated column name as per your instruction
EVENT_ID_COLUMN_NAME = 'event_id' # Assuming this column still exists
OLLAMA_MODEL = 'granite3.2:8b'
HUGGINGFACE_MODEL_NAME = "ibm-granite/granite-3.2-8b-instruct"

# Output directory for metrics and plots
RUN_OUTPUT_DIR = 'run_output_v2' # Changed output directory to indicate new version
METRICS_SUMMARY_FILE = os.path.join(RUN_OUTPUT_DIR, 'classification_metrics_summary.txt')
PROCESSING_TIME_PLOT = os.path.join(RUN_OUTPUT_DIR, 'processing_time_distribution.png')
TOKENS_PER_SECOND_PLOT = os.path.join(RUN_OUTPUT_DIR, 'tokens_per_second_distribution.png')
INPUT_TOKENS_PLOT = os.path.join(RUN_OUTPUT_DIR, 'input_tokens_distribution.png')
OUTPUT_TOKENS_PLOT = os.path.join(RUN_OUTPUT_DIR, 'output_tokens_distribution.png')
EVENT_TYPE_COUNTS_PLOT = os.path.join(RUN_OUTPUT_DIR, 'event_type_classification_counts.png')
EVENT_SUB_TYPE_COUNTS_PLOT = os.path.join(RUN_OUTPUT_DIR, 'event_sub_type_classification_counts.png') # New plot for sub-types


print(f"Initializing Hugging Face Tokenizer for '{HUGGINGFACE_MODEL_NAME}'...")
try:
    tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_NAME)
    print("Tokenizer initialized successfully.")
except Exception as e:
    print(f"Error initializing Hugging Face Tokenizer: {e}")
    print("Proceeding with a rough character-based token estimation as a fallback.")
    tokenizer = None

def get_token_count(text: str) -> int:
    if not text:
        return 0
    if tokenizer:
        return len(tokenizer.encode(text))
    else:
        # Fallback for rough estimation if tokenizer fails
        return len(text) // 4

# Dynamically get the list of fields from the imported schema
FIELDS = list(FIELD_VALUE_SCHEMA.keys())

# --- Schema Instructions Generation for Prompt ---
schema_instructions = ""
for field, definition in FIELD_VALUE_SCHEMA.items():
    if field == "event_type":
        schema_instructions += f"- For `{field}`, select one value from: {', '.join(json.dumps(val) for val in definition)}. If the event type cannot be determined from the report, use \"NULL\".\n"
    elif field == "event_sub_type":
        schema_instructions += f"- For `{field}`, select one value from the sub-types associated with the chosen `event_type`. Example: If `event_type` is 'VIOLENT CRIME', `event_sub_type` could be 'ASSAULT', 'KIDNAPPING', etc. If `event_type` is 'OTHERS', then `event_sub_type` MUST be 'OTHERS'. If the `event_sub_type` cannot be determined, use \"NULL\".\n"
        schema_instructions += "  Detailed sub-types per event type:\n"
        for etype, subtypes in FIELD_VALUE_SCHEMA["event_sub_type"].items():
            schema_instructions += f"  - {json.dumps(etype)}: {', '.join(json.dumps(val) for val in subtypes)}\n"
    elif isinstance(definition, list): # For categorical fields like state_of_victim, victim_gender
        schema_instructions += f"- For `{field}`, select one value from: {', '.join(json.dumps(val) for val in definition)}. If not explicitly stated, use \"not specified\".\n"
    elif definition == "text_allow_not_specified": # For free-form text fields
        schema_instructions += f"- For `{field}`, extract the relevant text directly from the report. If no relevant text is present, use \"not specified\". Do NOT infer or assume.\n"


def extract_report_data(report_text: str) -> dict:
    if not report_text or pd.isna(report_text):
        print(f"Skipping empty or invalid report.")
        # Provide default "NULL" or "not specified" for all fields
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

    prompt_content = f"""You are an extremely meticulous and precise emergency response classifier. Your task is to extract structured information from the provided emergency report (`event_info` column) by assigning values to predefined fields.

    **Crucial Instructions for Accurate and Non-Hallucinatory Extraction:**
    1.  **Extract ONLY information explicitly stated in the `event_info` text.** Do NOT infer, guess, or add external knowledge. If a piece of information is not directly present, use the specified default value.
    2.  **Strictly adhere to the provided field names and their allowed values/extraction rules.**
    3.  **Output Format:** The output MUST be in the format `field_name: value` for each requested field, one field per line.
    4.  **Special Handling for `event_type` and `event_sub_type`:**
        * If `event_type` cannot be determined, its value MUST be "NULL".
        * If `event_type` is determined to be "OTHERS", then `event_sub_type` MUST also be "OTHERS". In this specific case (and ONLY this case), if you can identify a *possible specific event type* in the `event_info` that is *not* in our predefined list of `event_type`s, append it in parentheses to the `event_sub_type` like this: `event_sub_type: OTHERS (Possible event type: [inferred type])`. Otherwise, `event_sub_type` remains just 'OTHERS'.
        * For all other `event_type`s, `event_sub_type` must be selected from the sub-types strictly associated with that `event_type`. If no specific sub-type from the list is clear, use "NULL".
    5.  **For fields marked "not specified":** If the information for a field is not explicitly present in the `event_info` text, its value MUST be "not specified". This applies to categorical and text fields.
    6.  **Keep values concise and factual.** Avoid explanations or extraneous text.

    **Schema for Field Values:**
    Please adhere to the following schema for allowed values and extraction rules for each field. This is critical for accurate classification.
    {schema_instructions}

    Now, analyze the following emergency report carefully from the `event_info` column:
    \"\"\"{report_text}\"\"\"

    Extract the information below based strictly on the report content and the provided schema.

    Output:"""

    input_tokens = get_token_count(prompt_content)
    start_time = time.perf_counter()
    llm_output = "" # Initialize llm_output

    try:
        response = chat(model=OLLAMA_MODEL, messages=[{
            "role": "user",
            "content": prompt_content
        }])
        llm_output = response['message']['content']
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        output_tokens = get_token_count(llm_output)
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
        for field in FIELDS:
            # Default to "NULL" for event_type/sub_type, "not specified" for others
            default_value = "NULL" if field in ["event_type", "event_sub_type"] else "not specified"
            final_data[field] = extracted_data.get(field, default_value)

            # Post-processing / Validation for event_type and event_sub_type
            if field == "event_type":
                # Ensure event_type is one of the allowed values or "NULL"
                if final_data[field] not in FIELD_VALUE_SCHEMA["event_type"] and final_data[field] != "NULL":
                    print(f"Warning: LLM returned invalid event_type '{final_data[field]}'. Forcing to 'NULL'.")
                    final_data[field] = "NULL"
            elif field == "event_sub_type":
                predicted_event_type = final_data.get('event_type')
                predicted_sub_type = final_data[field]
                
                # Check if event_type is OTHERS and handle sub_type accordingly
                if predicted_event_type == 'OTHERS':
                    # Allow 'OTHERS' as event_sub_type, possibly with an inferred type in parentheses
                    if not predicted_sub_type.startswith('OTHERS'):
                        print(f"Warning: event_type is 'OTHERS' but event_sub_type '{predicted_sub_type}' is not 'OTHERS'. Forcing to 'OTHERS'.")
                        final_data[field] = 'OTHERS'
                elif predicted_event_type != "NULL" and predicted_event_type in FIELD_VALUE_SCHEMA["event_sub_type"]:
                    allowed_sub_types = FIELD_VALUE_SCHEMA["event_sub_type"][predicted_event_type]
                    # If the predicted sub_type is not in the allowed list for the predicted event_type, default to NULL/not specified
                    if predicted_sub_type not in allowed_sub_types:
                        print(f"Warning: event_sub_type '{predicted_sub_type}' is not valid for event_type '{predicted_event_type}'. Forcing to 'NULL'.")
                        final_data[field] = "NULL"
                else: # event_type is NULL or not a valid type, so sub_type should also be NULL
                    if predicted_sub_type not in ["NULL", "not specified"]:
                        print(f"Warning: event_type '{predicted_event_type}' invalid/NULL, forcing event_sub_type '{predicted_sub_type}' to 'NULL'.")
                        final_data[field] = "NULL"


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
        input_tokens_on_error = get_token_count(prompt_content)
        output_tokens_on_error = get_token_count(llm_output)
        tokens_per_second_on_error = (input_tokens_on_error + output_tokens_on_error) / processing_time if processing_time > 0 else 0

        # FIXED: Using 'report_text' which is available in this function's scope.
        # Also, perform replace outside the f-string.
        cleaned_report_preview_on_error = str(report_text)[:70].replace('\n', ' ')
        print(f"An error occurred while processing report: '{cleaned_report_preview_on_error}...': {e}")
        
        error_message = str(e).replace('\n', ' ')[:200] # Truncate and clean

        error_data = {
            field: "error_processing" for field in FIELDS
        }
        error_data['event_type'] = "NULL" # Default to NULL on error for event type
        error_data['event_sub_type'] = "NULL" # Default to NULL on error for event sub type
        # Set other fields to "not specified" on error
        for field in FIELDS:
            if field not in ["event_type", "event_sub_type"]:
                error_data[field] = "not specified"


        error_data['__input_tokens'] = input_tokens_on_error
        error_data['__output_tokens'] = output_tokens_on_error
        error_data['__processing_time_sec'] = processing_time
        error_data['__tokens_per_second'] = tokens_per_second_on_error
        error_data['__status'] = 'error'
        error_data['__error_message'] = error_message
        return error_data

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
    
    # Save summary statistics to a text file
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
        
        # Add error details if any
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


if __name__ == "__main__":
    print(f"--- Ollama LLM Report Classification ---")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Input CSV: {INPUT_CSV_FILE}")
    print(f"Output CSV: {OUTPUT_CSV_FILE}")
    print(f"Report Column: '{REPORT_COLUMN_NAME}'")
    print(f"Event ID Column: '{EVENT_ID_COLUMN_NAME}'")
    print(f"Target Fields: {', '.join(FIELDS)}")
    print(f"Metrics and Plots Output Directory: {RUN_OUTPUT_DIR}")

    os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(INPUT_CSV_FILE), exist_ok=True) # Ensure data directory exists
    os.makedirs(os.path.dirname(OUTPUT_CSV_FILE), exist_ok=True) # Ensure output directory exists

    try:
        df_input = pd.read_csv(INPUT_CSV_FILE)
        print(f"Successfully loaded '{INPUT_CSV_FILE}'. Number of reports: {len(df_input)}")
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_CSV_FILE}' not found. Please ensure it exists.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input CSV '{INPUT_CSV_FILE}': {e}")
        sys.exit(1)

    if REPORT_COLUMN_NAME not in df_input.columns:
        print(f"Error: Column '{REPORT_COLUMN_NAME}' not found in '{INPUT_CSV_FILE}'.")
        print(f"Available columns: {df_input.columns.tolist()}")
        sys.exit(1)

    if EVENT_ID_COLUMN_NAME not in df_input.columns:
        print(f"Error: Column '{EVENT_ID_COLUMN_NAME}' not found in '{INPUT_CSV_FILE}'.")
        print(f"Available columns: {df_input.columns.tolist()}")
        sys.exit(1)

    # Process all rows in the dataframe. Changed to first 5 rows for sample testing.
    df_to_process = df_input.iloc[:5]


    BATCH_SIZE = 1 # Keep batching for efficiency and intermediate saving


    output_file_exists = os.path.exists(OUTPUT_CSV_FILE)
    write_mode = 'a' if output_file_exists else 'w'
    header_written = output_file_exists 

    all_extracted_records_batch = [] 
    all_metrics_for_report = [] 

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

    print(f"\nStarting classification process for {len(df_to_process)} reports...")
    for index, row in df_to_process.iterrows():
        event_id = row[EVENT_ID_COLUMN_NAME]
        original_report = row[REPORT_COLUMN_NAME]
        
        # FIXED: Perform replace outside the f-string
        cleaned_report_preview = str(original_report)[:70].replace('\n', ' ')
        print(f"\nProcessing report {index + 1}/{len(df_to_process)} (Event ID: {event_id}): '{cleaned_report_preview}...'")

        extracted_record = extract_report_data(original_report)

        # Combine original identifiers with extracted data for CSV
        final_record_for_csv = {
            EVENT_ID_COLUMN_NAME: event_id,
            REPORT_COLUMN_NAME: original_report,
            **extracted_record 
        }
        all_extracted_records_batch.append(final_record_for_csv)
        
        # Also store metrics separately for overall report generation
        all_metrics_for_report.append(extracted_record)

        # Print extracted data for immediate feedback
        for field in FIELDS:
            print(f"  {field.replace('_', ' ').title()}: {extracted_record.get(field, 'N/A')}")
        print(f"  Status: {extracted_record.get('__status', 'unknown')}")
        print(f"  Input Tokens (HF): {extracted_record.get('__input_tokens', 0)}")
        print(f"  Output Tokens (HF): {extracted_record.get('__output_tokens', 0)}")
        print(f"  Processing Time: {extracted_record.get('__processing_time_sec', 0):.2f} seconds")
        print(f"  Tokens per Second: {extracted_record.get('__tokens_per_second', 0):.2f}")
        if extracted_record.get('__status') == 'error':
            print(f"  Error Message: {extracted_record.get('__error_message', 'No error message provided.')}")

        # Save batch to CSV
        if (index + 1) % BATCH_SIZE == 0 or (index + 1) == len(df_to_process):
            if all_extracted_records_batch: 
                df_batch = pd.DataFrame(all_extracted_records_batch, columns=output_columns_order)
                try:
                    # 'header=not header_written' ensures header is written only once
                    df_batch.to_csv(OUTPUT_CSV_FILE, mode=write_mode, header=not header_written, index=False)
                    print(f"\nSaved batch of {len(all_extracted_records_batch)} reports to '{OUTPUT_CSV_FILE}'.")
                    header_written = True # Mark header as written after first save
                    write_mode = 'a' # Subsequent writes will append
                    all_extracted_records_batch = [] # Clear batch for next set
                except Exception as e:
                    print(f"Error saving batch to CSV: {e}")
            else:
                print(f"Batch {index // BATCH_SIZE + 1} empty, skipping save.")

    # Generate final metrics report after all processing
    if all_metrics_for_report:
        metrics_df_final = pd.DataFrame(all_metrics_for_report)
        generate_metrics_report(metrics_df_final, RUN_OUTPUT_DIR)
    else:
        print("\nNo reports processed. Skipping final metrics report generation.")

    print(f"\n--- Ollama LLM Report Classification Complete ---")
    print(f"All classified reports saved to: {OUTPUT_CSV_FILE}")
    print(f"Performance metrics summary and plots saved to: {RUN_OUTPUT_DIR}")