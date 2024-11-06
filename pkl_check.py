import pickle

with open('/home/aih/gizem.mert/Dino/DINO/fold0/mixed_uncertain/max_10_percent/file_paths.pkl', 'rb') as f:
    mixed_data_filepaths = pickle.load(f)

for key, value in mixed_data_filepaths.items():
    print(f"Key: {key}, Value: {value}, Type: {type(value)}")
