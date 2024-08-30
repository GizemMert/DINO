import os

# Define the path to the RawImages directory
PATH_TO_IMAGES = '/lustre/groups/labs/marr/qscd01/datasets/230824_MLL_BELUGA/RawImages'

# List all entries in the base directory and filter out only directories (patient folders)
patient_folders = [os.path.join(PATH_TO_IMAGES, patient_folder)
                   for patient_folder in os.listdir(PATH_TO_IMAGES)
                   if os.path.isdir(os.path.join(PATH_TO_IMAGES, patient_folder))]

# Count the total number of patient folders
num_patient_folders = len(patient_folders)

# Initialize a counter for folders that already have the output file
num_folders_with_output = 0

# Loop through patient folders to check for the existence of the .npy file
for folder_patient in patient_folders:
    output_npy_file = os.path.join(folder_patient, 'single_cell_probabilities.npy')
    if os.path.exists(output_npy_file):
        num_folders_with_output += 1

# Print the counts
print(f"Total number of patient folders in RawImages: {num_patient_folders}")
print(f"Number of patient folders with 'single_cell_probabilities.npy' file: {num_folders_with_output}")