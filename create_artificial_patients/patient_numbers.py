import os

# Define the path to the RawImages directory
PATH_TO_IMAGES = '/lustre/groups/labs/marr/qscd01/datasets/230824_MLL_BELUGA/RawImages'

# List all entries in the base directory and filter out only directories (patient folders)
patient_folders = [os.path.join(PATH_TO_IMAGES, patient_folder)
                   for patient_folder in os.listdir(PATH_TO_IMAGES)
                   if os.path.isdir(os.path.join(PATH_TO_IMAGES, patient_folder))]

# Count the number of patient folders
num_patient_folders = len(patient_folders)

print(f"Number of patient folders in RawImages: {num_patient_folders}")