
import os
import sys
# Import the functions from your translation script
from translator_hindi_to_eng import process_hindi_reports_in_folder, ENGLISH_OUTPUT_FOLDER, HINDI_INPUT_FOLDER

# Import the function from your classification script
# Assuming your classification script is named 'classification_pipeline.py'
# If it's your previous file, name it accordingly, e.g., 'llm_classifier.py'
# Let's assume you've renamed it for clarity and added the run_classification_pipeline function
from json_hinditoEng_Converted_classifier import run_classification_pipeline

# --- Define Pipeline Input/Output Folders ---
# These should match the settings in your translator and classifier scripts if they are hardcoded
# Or, better, pass these as arguments to the functions if you make them configurable.
HINDI_REPORTS_DIR = 'data/ground_truth_hin'
ENGLISH_TRANSLATED_REPORTS_DIR = 'output/data/english_translated_reports/'
CLASSIFICATION_OUTPUT_DIR = 'output/' # Your classifier saves here
METRICS_OUTPUT_DIR = 'run_output/' # Your classifier saves metrics here

def main_pipeline():
    print("--- Starting Full Language Processing Pipeline ---")

    # Step 1: Ensure input directories exist
    os.makedirs(HINDI_REPORTS_DIR, exist_ok=True)
    os.makedirs(ENGLISH_TRANSLATED_REPORTS_DIR, exist_ok=True)
    os.makedirs(CLASSIFICATION_OUTPUT_DIR, exist_ok=True)
    os.makedirs(METRICS_OUTPUT_DIR, exist_ok=True)

    print(f"\nPhase 1: Translating Hindi reports from '{HINDI_REPORTS_DIR}' to English...")
    translated_files = process_hindi_reports_in_folder()

    if not translated_files:
        print("\nNo English reports were generated. Aborting classification phase.")
        sys.exit(0) # Exit if no files were translated

    print(f"\nPhase 2: Classifying English reports from '{ENGLISH_TRANSLATED_REPORTS_DIR}'...")
    # Pass the folder containing the translated English files to the classification pipeline
    run_classification_pipeline(ENGLISH_TRANSLATED_REPORTS_DIR)

    print("\n--- Full Pipeline Execution Complete ---")

if __name__ == "__main__":
    main_pipeline()