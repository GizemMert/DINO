import os
import re

# Define the path to the directory containing the patient folders
data_directory = '/lustre/groups/labs/marr/qscd01/datasets/230824_MLL_BELUGA/RawImages'

# Expected filename pattern
expected_pattern = re.compile(r'Gal-\d{6}\.RGB\.TIF$', re.IGNORECASE)

# List to store patient folders with mismatched filenames
folders_with_mismatched_files = []

# Iterate over each patient folder
for folder_patient in os.listdir(data_directory):
    folder_patient_path = os.path.join(data_directory, folder_patient)

    # Check if the path is a directory
    if os.path.isdir(folder_patient_path):
        has_mismatch = False

        # Iterate over each file in the patient folder
        for file_name in os.listdir(folder_patient_path):
            file_path = os.path.join(folder_patient_path, file_name)

            # Check if it's a file and ignore .npy and .db files
            if os.path.isfile(file_path) and not (file_name.endswith('.npy') or file_name.endswith('.db')):
                # Check if the file name does not match the expected pattern
                if not expected_pattern.match(file_name):
                    print(f'Mismatched file: {file_path}')  # Print out the mismatched file path
                    has_mismatch = True

        # If any mismatched files were found in this folder, add it to the list
        if has_mismatch:
            folders_with_mismatched_files.append(folder_patient)

# Print the results
print(f'\nNumber of patient folders with mismatched filenames: {len(folders_with_mismatched_files)}')
print('List of patient folders with mismatched filenames:')
for folder in folders_with_mismatched_files:
    print(folder)
