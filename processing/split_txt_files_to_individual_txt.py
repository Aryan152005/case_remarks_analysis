import os

def split_conversations(input_file, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Read the full content of the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split content by double newlines or newlines
    conversations = [c.strip() for c in content.strip().split('\n') if c.strip()]

    for convo in conversations:
        # Assume the first word (before first space) is the ID
        parts = convo.strip().split(' ', 1)
        if len(parts) != 2:
            continue  # Skip malformed lines

        caller_id, conversation_text = parts[0], parts[1]

        # Create output file name
        output_filename = f"1_{caller_id}.txt"
        output_path = os.path.join(output_folder, output_filename)

        # Write only the conversation text
        with open(output_path, 'w', encoding='utf-8') as out_f:
            out_f.write(conversation_text)

    print(f"✅ Done! {len(conversations)} files saved in: {output_folder}")

# Example usage:
input_txt_file = "rnnt_predictions_100.txt"  # Replace with your actual input file
output_dir = "split_conversations"
split_conversations(input_txt_file, output_dir)
