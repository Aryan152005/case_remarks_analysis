import pandas as pd
import numpy as np # For checking for all NaN values

# Path to the Excel file
xlsx_file = 'KL_Event_Data_2024 (1).xlsx'

# Get all sheet names from the Excel file
sheet_names = pd.ExcelFile(xlsx_file).sheet_names
print(f"Found sheets: {sheet_names}")

# List to hold DataFrames
df_list = []

# To keep track of column names for standardization
all_column_names = set()

for sheet in sheet_names:
    print(f"\n--- Processing sheet: {sheet} ---")

    # Read the sheet initially without assuming a header, to inspect its structure
    # We read a few extra rows to find the actual header if it's not the very first row
    temp_df = pd.read_excel(xlsx_file, sheet_name=sheet, header=None, nrows=10) # Read first 10 rows

    # Identify the row that looks like the actual header
    # We'll look for the first row that is not entirely NaN (empty)
    # and has at least two non-NaN values (to avoid picking up a single entry as header)
    header_row_index = 0
    for r_idx in range(temp_df.shape[0]):
        row_data = temp_df.iloc[r_idx, :]
        # Check if row is not all NaN and has at least 2 non-NaN values (heuristic for header)
        if not row_data.isna().all() and row_data.count() >= 2:
            header_row_index = r_idx
            print(f"Detected header row for '{sheet}' at index: {header_row_index}")
            break
    else:
        print(f"Warning: Could not reliably detect header for '{sheet}'. Assuming first non-empty row.")


    # Now read the sheet again, this time specifying the detected header row
    # This automatically skips rows before the header.
    df = pd.read_excel(xlsx_file, sheet_name=sheet, header=header_row_index)

    # --- Step 1: Handle the entirely empty first column (Column A) ---
    # Check if the first column is entirely NaN
    if df.shape[1] > 0 and df.iloc[:, 0].isna().all():
        print(f"First column of '{sheet}' (original index 0) is entirely empty. Dropping it.")
        df = df.iloc[:, 1:] # Drop the first column
    else:
        print(f"First column of '{sheet}' contains data or is not entirely empty.")

    # --- Step 2: Handle empty rows after header and leading/trailing whitespace in column names ---
    # Drop rows where all values are NaN (might occur after reading with header)
    df.dropna(how='all', inplace=True)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # --- Step 3: Collect unique column names for standardization later ---
    all_column_names.update(df.columns)

    print(f"Shape of '{sheet}' after processing: {df.shape}")
    print(f"First 5 rows of '{sheet}' after processing:")
    print(df.head())
    print(f"Columns of '{sheet}': {list(df.columns)}")

    # Append the processed DataFrame to the list
    df_list.append(df)

# --- Standardization before concatenation ---
# This step is crucial if column names are not exactly the same across all sheets
# due to slight variations (e.g., 'Date ' vs 'Date').
# If you expect completely identical column names and just need to align them,
# this part helps by reindexing to a common set of columns.

# Create a master list of sorted unique column names
master_columns = sorted(list(all_column_names))
print(f"\nMaster list of columns for concatenation: {master_columns}")

final_df_list = []
for i, df_item in enumerate(df_list):
    sheet_name_for_debug = sheet_names[i]
    # Reindex each DataFrame to the master_columns, filling missing columns with NaN
    # This aligns all DataFrames for concatenation
    aligned_df = df_item.reindex(columns=master_columns)
    final_df_list.append(aligned_df)
    print(f"DataFrame from '{sheet_name_for_debug}' reindexed. New shape: {aligned_df.shape}")


# Concatenate all DataFrames
print("\n--- Concatenating aligned DataFrames ---")
combined_df = pd.concat(final_df_list, ignore_index=True)
print(f"Shape of combined_df after concatenation: {combined_df.shape}")
print(f"First 5 rows of combined_df:")
print(combined_df.head())


# Final check for any entirely empty columns that might have appeared due to reindexing
# (e.g., if a column only existed in one sheet but was NaN elsewhere)
combined_df.dropna(axis=1, how='all', inplace=True)
print(f"Shape of combined_df after dropping any new all-NaN columns: {combined_df.shape}")


# Export to CSV
csv_output_file = 'combined_event_data.csv'
combined_df.to_csv(csv_output_file, index=False)

print(f"\n✅ All sheets combined and saved to '{csv_output_file}'")