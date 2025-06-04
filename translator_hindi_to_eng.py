import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
import os
import sys
import time
import re
import json

# --- Configuration for Translator ---
# Automatically detect if CUDA (GPU) is available, otherwise use CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Source and Target Languages for IndicTrans2
SRC_LANG = "hin_Deva"  # Hindi in Devanagari script
TGT_LANG = "eng_Latn"  # English in Latin script

# Model Name: AI4Bharat IndicTrans2 Indic-to-English 1B parameter model
# This is the most capable open-source model for your task.
MODEL_NAME = "ai4bharat/indictrans2-indic-en-1B" 

# Folder paths for input Hindi reports and output English translations
# Ensure these match the paths used in your main.py if integrating
HINDI_INPUT_FOLDER = 'data/ground_truth_hin'
ENGLISH_OUTPUT_FOLDER = 'output/data/english_translated_reports/'

# Directory for storing detailed translation performance metrics
TRANSLATOR_METRICS_OUTPUT_DIR = 'run_output/translator_metrics/' 

# Create output directories if they don't exist
os.makedirs(ENGLISH_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TRANSLATOR_METRICS_OUTPUT_DIR, exist_ok=True)

# --- Batching Configuration ---
# Adjust this value based on your GPU memory and typical report length.
# Start with a small batch size (e.g., 1 or 2) and increase if there's room,
# especially if you have 8GB VRAM with the 1B parameter model.
TRANSLATION_BATCH_SIZE = 1 

# --- Global Model and Processor Variables ---
# These will be initialized once and reused across all translations
tokenizer = None
model = None
ip = None

# --- Model Initialization Function ---
def initialize_translator_model():
    """
    Initializes the AI4Bharat IndicTrans2 model, tokenizer, and IndicProcessor.
    Handles CUDA/CPU device selection and Flash Attention 2 setup.
    """
    global tokenizer, model, ip
    
    # Check if model is already initialized to prevent redundant loading
    if tokenizer is not None and model is not None and ip is not None:
        print("Translator model already initialized.")
        return

    print(f"Initializing AI4Bharat IndicTrans2 model '{MODEL_NAME}' on device: {DEVICE}...")
    try:
        start_time = time.time()
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        # Determine attention implementation (Flash Attention 2 for CUDA if supported)
        attn_impl = "flash_attention_2" if DEVICE == "cuda" and torch.cuda.is_available() else None
        
        if attn_impl == "flash_attention_2":
            try:
                import flash_attn # Attempt to import flash_attn module
                # Check for Flash Attention 2 specific GPU compute capability (Ampere+ -> 8.0+)
                if torch.cuda.get_device_properties(0).major < 8:
                    print(f"GPU (Compute Capability {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}) does not support Flash Attention 2 (requires 8.0+). Falling back to default attention.")
                    attn_impl = None
                else:
                    print("Using flash_attention_2 for optimized performance.")
            except ImportError:
                print("Flash Attention 2 module `flash_attn` not found. Please install it for better performance if encountering OOM errors.")
                print("  Install command: `pip install flash-attn --no-build-isolation` (ensure CUDA toolkit is correctly set up and matches PyTorch's CUDA version).")
                attn_impl = None
        elif DEVICE == "cuda" and torch.cuda.is_available():
            print("Running on CUDA but not using flash_attention_2. Consider installing `flash_attn` for better performance if encountering OOM errors.")

        # Load the model with appropriate dtype and attention implementation
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32, # Use float16 for GPU, float32 for CPU
            attn_implementation=attn_impl 
        ).to(DEVICE)
        model.eval() # Set model to evaluation mode for inference (disables dropout, etc.)

        # Initialize IndicProcessor for pre- and post-processing
        ip = IndicProcessor(inference=True)
        
        end_time = time.time()
        print(f"AI4Bharat IndicTrans2 model and tokenizer initialized successfully in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error initializing AI4Bharat IndicTrans2 model '{MODEL_NAME}': {e}")
        print("Common causes: model not found on Hugging Face (check model name/internet), or insufficient VRAM for loading.")
        print(f"  If 'CUDA out of memory' during initialization, try reducing the model size, increasing GPU VRAM, or setting DEVICE='cpu'.")
        sys.exit(1) # Exit if model initialization fails

# --- Dynamic Code-Mixing Pre-processing ---
# This regex aims to capture sequences of Latin script characters (a-zA-Z),
# numbers (0-9), and some common punctuation/symbols that are typically
# part of code-mixed English words/phrases within Hindi text.
# It is designed to be dynamic, not relying on a fixed list of English words.
LATIN_MIXED_REGEX = r'\b[a-zA-Z0-9]+(?:[\s.,!?-]*[a-zA-Z0-9]+)*\b|[\(\)\[\]\{\}\+\=\-\_\*&%@#\$`~]'

def identify_and_replace_mixed_segments(text: str) -> tuple[str, dict]:
    """
    Identifies segments in the text that are predominantly Latin script (English words, numbers)
    or specific symbols, and replaces them with unique placeholders.
    This preserves these segments during translation.
    """
    segments = []
    placeholders = {}
    last_idx = 0
    
    # Iterate through all non-overlapping matches of the regex in the text
    for i, match in enumerate(re.finditer(LATIN_MIXED_REGEX, text)):
        # Add the Hindi part before the current matched segment
        if match.start() > last_idx:
            segments.append(text[last_idx:match.start()])
            
        segment_text = match.group(0)
        
        # Only create a placeholder if the segment is not just whitespace or
        # pure punctuation (e.g., if it contains letters or numbers)
        # This prevents replacing meaningful Hindi punctuation with placeholders unnecessarily.
        if segment_text.strip() and not re.fullmatch(r'[^a-zA-Z0-9\s]+', segment_text.strip()):
            placeholder = f"__ENG_PH_{len(placeholders)}__" # Unique placeholder name
            placeholders[placeholder] = segment_text
            segments.append(placeholder)
        else:
            segments.append(segment_text) # Keep original if it's pure punctuation/whitespace
            
        last_idx = match.end()
    
    # Add any remaining text after the last match
    if last_idx < len(text):
        segments.append(text[last_idx:])
        
    processed_text = "".join(segments)
    
    return processed_text, placeholders

def restore_mixed_segments(translated_text: str, placeholders: dict) -> str:
    """
    Restores the original Latin script segments (English words, numbers) back into
    the translated text using the stored placeholders.
    """
    restored_text = translated_text
    # Restore in reverse order of placeholder generation to handle potential overlaps safely
    # (though with unique placeholders, order usually doesn't matter much)
    for placeholder, original_segment in sorted(placeholders.items(), key=lambda item: int(item[0].split('_')[-2]), reverse=True):
        restored_text = restored_text.replace(placeholder, original_segment)
    return restored_text

# --- Main Translation Function ---
def translate_hindi_to_english_batch(hindi_texts: list[str]) -> tuple[list[str], dict]:
    """
    Translates a list of Hindi texts to English using the loaded AI4Bharat model.
    Includes dynamic pre-processing for code-mixing and robust post-processing.
    Records time taken for each sub-step.
    """
    if not hindi_texts:
        return [], {}

    # Ensure model is initialized before starting translation
    if tokenizer is None or model is None or ip is None:
        initialize_translator_model()

    processed_hindi_texts = []
    all_placeholders = [] # Store placeholders unique to each text in the batch

    # Step 1: Pre-processing for Code-Mixing (Time Tracking)
    preprocess_start_time = time.time()
    for text in hindi_texts:
        processed_text, placeholders = identify_and_replace_mixed_segments(text)
        processed_hindi_texts.append(processed_text)
        all_placeholders.append(placeholders)
    preprocess_end_time = time.time()
    
    # Step 2: Tokenization (Time Tracking)
    tokenize_start_time = time.time()
    # IndicProcessor preprocesses the Hindi text for the model (e.g., normalizes script, etc.)
    batch_preprocessed = ip.preprocess_batch(
        processed_hindi_texts, # Use the texts with placeholders
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
    )
    
    # Tokenize input. `truncation=True` will cut off inputs longer than the model's max input token limit.
    # This is a critical point for "100% accuracy" if your source reports are very long.
    inputs = tokenizer(
        batch_preprocessed,
        truncation=True, # Truncates input if it exceeds model's max input tokens (e.g., 512)
        padding="longest", # Pads shorter sequences to the length of the longest in the batch
        return_tensors="pt", # Returns PyTorch tensors
        return_attention_mask=True, # Returns attention mask for proper padding handling by model
    ).to(DEVICE)
    tokenize_end_time = time.time()

    # Step 3: Generate Translations (Model Inference - Time Tracking)
    generate_start_time = time.time()
    with torch.no_grad(): # Disable gradient calculation for inference to save memory and speed
        generated_tokens = model.generate(
            **inputs,
            use_cache=True, # Improves performance by caching attention outputs
            min_length=0,   # Minimum output length (0 allows empty translations if model decides)
            max_length=2048, # Maximum output length. Set generously to avoid truncating translations.
                             # Consider increasing to 4096 if reports are extremely long.
            num_beams=5,    # Number of beams for beam search. Higher values generally mean better quality
                            # but slower inference. 5 is a good balance.
            num_return_sequences=1, # Return only the single best translation
            # no_repeat_ngram_size=3, # Optional: helps prevent repetitive phrases (uncomment and test if needed)
        )
    generate_end_time = time.time()

    # Step 4: Decode and Post-process (Time Tracking)
    decode_postprocess_start_time = time.time()
    # Decode the generated tokens back into human-readable text
    generated_tokens_decoded = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True, # Remove special tokens like padding and EOS
        clean_up_tokenization_spaces=True, # Clean up extra spaces introduced by tokenization
    )
    # IndicProcessor postprocesses the translated English text (e.g., correct casing, spacing, etc.)
    translations = ip.postprocess_batch(generated_tokens_decoded, lang=TGT_LANG)

    # Final Post-processing: Restore original mixed segments
    final_translations = []
    for i, translated_text in enumerate(translations):
        final_translations.append(restore_mixed_segments(translated_text, all_placeholders[i]))
    decode_postprocess_end_time = time.time()

    # Collect detailed metrics for the current batch
    batch_metrics = {
        'preprocess_time_sec': preprocess_end_time - preprocess_start_time,
        'tokenize_time_sec': tokenize_end_time - tokenize_start_time,
        'generate_time_sec': generate_end_time - generate_start_time,
        'decode_postprocess_time_sec': decode_postprocess_end_time - decode_postprocess_start_time,
        'total_batch_time_sec': (decode_postprocess_end_time - preprocess_start_time),
        'num_texts_in_batch': len(hindi_texts),
        'input_tokens_in_batch': inputs['input_ids'].numel(), # Total tokens in the input batch
        'output_tokens_in_batch': generated_tokens.numel() # Total tokens in the generated output batch
    }
    
    return final_translations, batch_metrics

# --- Main Processing Loop for Folder ---
def process_hindi_reports_in_folder() -> list[str]:
    """
    Reads Hindi text files from HINDI_INPUT_FOLDER, translates them to English in batches,
    and saves each English translation individually to ENGLISH_OUTPUT_FOLDER.
    Collects and saves detailed performance metrics for analysis.
    """
    initialize_translator_model() # Ensure model is loaded and ready before processing files

    # Get a sorted list of all Hindi text files in the input folder
    hindi_files = sorted([f for f in os.listdir(HINDI_INPUT_FOLDER) if f.endswith('.txt')])
    total_files = len(hindi_files)
    
    if not hindi_files:
        print(f"No Hindi text files found in '{HINDI_INPUT_FOLDER}'. Please place your .txt files there to begin translation.")
        return []

    print(f"\nFound {total_files} Hindi reports in '{HINDI_INPUT_FOLDER}'.")
    print(f"Translating in batches of {TRANSLATION_BATCH_SIZE}...")

    all_translated_file_paths = [] # To keep track of successfully translated files
    all_batch_metrics = []       # To store metrics for each processed batch
    
    # Iterate through files in batches
    for i in range(0, total_files, TRANSLATION_BATCH_SIZE):
        batch_files = hindi_files[i : i + TRANSLATION_BATCH_SIZE]
        
        hindi_texts_batch = []
        original_filenames_batch = []

        # Read content for the current batch of files
        for filename in batch_files:
            filepath = os.path.join(HINDI_INPUT_FOLDER, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    hindi_texts_batch.append(f.read())
                original_filenames_batch.append(filename)
            except Exception as e:
                print(f"Error reading file '{filename}': {e}. Skipping this file in batch.")
                continue # Skip to the next file in the batch if reading fails

        if not hindi_texts_batch: # If all files in the batch failed to read or batch was empty
            print(f"Skipping empty or failed batch {i//TRANSLATION_BATCH_SIZE + 1}.")
            continue

        print(f"Processing batch {i//TRANSLATION_BATCH_SIZE + 1} of {len(hindi_files) // TRANSLATION_BATCH_SIZE + (1 if len(hindi_files) % TRANSLATION_BATCH_SIZE > 0 else 0)} ({len(hindi_texts_batch)} files)...")

        try:
            # Call the translation function for the current batch
            english_translations_batch, batch_metrics = translate_hindi_to_english_batch(hindi_texts_batch)
            
            # Add batch-specific metadata to metrics
            batch_metrics['batch_index'] = i // TRANSLATION_BATCH_SIZE
            batch_metrics['start_file_index'] = i
            batch_metrics['file_names_in_batch'] = original_filenames_batch
            all_batch_metrics.append(batch_metrics)

            # Save translated texts for the current batch to output folder
            for j, translation in enumerate(english_translations_batch):
                original_filename = original_filenames_batch[j]
                output_filepath = os.path.join(ENGLISH_OUTPUT_FOLDER, original_filename)
                try:
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        f.write(translation)
                    all_translated_file_paths.append(output_filepath)
                except Exception as e:
                    print(f"Error saving translated file '{original_filename}': {e}")
        
        except RuntimeError as e: # Catch specific runtime errors like CUDA OOM
            if "CUDA out of memory" in str(e):
                print(f"CUDA Out of Memory error encountered for batch starting with file '{original_filenames_batch[0]}'.")
                print(f"Recommendation: Try reducing 'TRANSLATION_BATCH_SIZE' ({TRANSLATION_BATCH_SIZE}) or processing fewer files at once.")
                torch.cuda.empty_cache() # Attempt to free up cached memory
            else:
                print(f"An unexpected runtime error occurred during batch translation: {e}")
            break # Stop processing further batches after a critical error
        except Exception as e: # Catch any other unexpected errors during batch processing
            print(f"An unexpected error occurred during batch translation: {e}")
            break # Stop processing further batches

    print(f"\nTranslation phase complete. Successfully translated and saved {len(all_translated_file_paths)} files to '{ENGLISH_OUTPUT_FOLDER}'.")
    
    # Save cumulative metrics for the entire translation run
    if all_batch_metrics:
        metrics_filepath = os.path.join(TRANSLATOR_METRICS_OUTPUT_DIR, f"translator_metrics_{time.strftime('%Y%m%d-%H%M%S')}.json")
        try:
            with open(metrics_filepath, 'w', encoding='utf-8') as f:
                json.dump(all_batch_metrics, f, indent=4, ensure_ascii=False) # Use indent for readability
            print(f"Translator batch metrics saved to: {metrics_filepath}")
            
            # Calculate and print overall summary statistics
            total_preprocess_time = sum(m['preprocess_time_sec'] for m in all_batch_metrics)
            total_tokenize_time = sum(m['tokenize_time_sec'] for m in all_batch_metrics)
            total_generate_time = sum(m['generate_time_sec'] for m in all_batch_metrics)
            total_decode_postprocess_time = sum(m['decode_postprocess_time_sec'] for m in all_batch_metrics)
            total_overall_time = sum(m['total_batch_time_sec'] for m in all_batch_metrics)
            total_input_tokens = sum(m['input_tokens_in_batch'] for m in all_batch_metrics)
            total_output_tokens = sum(m['output_tokens_in_batch'] for m in all_batch_metrics)
            
            print("\n--- Translation Phase Summary Metrics ---")
            print(f"Total files attempted for translation: {total_files}")
            print(f"Total files successfully translated and saved: {len(all_translated_file_paths)}")
            print(f"Total pre-processing time: {total_preprocess_time:.4f} seconds")
            print(f"Total tokenization time: {total_tokenize_time:.4f} seconds")
            print(f"Total generation (translation) time: {total_generate_time:.4f} seconds")
            print(f"Total decoding & post-processing time: {total_decode_postprocess_time:.4f} seconds")
            print(f"Total overall translation time (wall clock): {total_overall_time:.4f} seconds")
            
            # Calculate and print average tokens per second for generation (most computationally intensive part)
            if total_generate_time > 0:
                 print(f"Average input tokens per second (generation only): {total_input_tokens / total_generate_time:.2f} tokens/sec")
                 print(f"Average output tokens per second (generation only): {total_output_tokens / total_generate_time:.2f} tokens/sec")
            else:
                 print("No generation time recorded (no successful batches).")
            
        except Exception as e:
            print(f"Error saving or calculating translator metrics: {e}")

    return all_translated_file_paths

# --- Entry point for the script ---
if __name__ == "__main__":
    print("--- AI4Bharat Hindi to English Translator (Standalone Execution) ---")
    print(f"Input Hindi reports expected in: '{HINDI_INPUT_FOLDER}'")
    print(f"English translations will be saved to: '{ENGLISH_OUTPUT_FOLDER}'")
    
    # Guide user to place files if the input folder is empty
    if not os.path.exists(HINDI_INPUT_FOLDER) or not os.listdir(HINDI_INPUT_FOLDER):
        print(f"\nNote: The input directory '{HINDI_INPUT_FOLDER}' is empty or does not exist.")
        print(f"Please create this directory and place your Hindi .txt files into it for translation.")
    
    # Start the translation process
    processed_files = process_hindi_reports_in_folder()
    
    # Final confirmation message
    if processed_files:
        print(f"\nProcess completed. Check '{ENGLISH_OUTPUT_FOLDER}' for translated files and '{TRANSLATOR_METRICS_OUTPUT_DIR}' for performance metrics.")
    else:
        print("\nTranslation process finished with no files translated. Check logs for errors or ensure input files are present.")