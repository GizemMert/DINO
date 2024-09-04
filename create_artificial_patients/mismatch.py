import pandas as pd

# Load the single cell results CSV file
single_cell_results_path = "/home/aih/gizem.mert/Dino/DINO/fold0/train/single_cell_results.csv"

# Read the CSV file into a DataFrame
df_sc_res = pd.read_csv(single_cell_results_path)

# Display unique AML subtypes in the single cell results data
unique_subtypes = df_sc_res["AML_subtype"].unique()
print("Unique AML subtypes in single cell results data:")
print(unique_subtypes)

# Check specifically for the presence of "MDS / MPN"
if "MDS / MPN" in unique_subtypes:
    print("\n'MDS / MPN' is present in the single cell results data.")
else:
    print("\n'MDS / MPN' is NOT present in the single cell results data.")

# Optional: Count the number of entries for each AML subtype
subtype_counts = df_sc_res["AML_subtype"].value_counts()
print("\nNumber of entries for each AML subtype:")
print(subtype_counts)

# Optional: Provide a summary of entries related to "MDS / MPN"
mds_mpn_entries = df_sc_res[df_sc_res["AML_subtype"] == "MDS / MPN"]
print("\nSummary of entries for 'MDS / MPN':")
print(mds_mpn_entries.describe(include='all'))

# Optional: Check if there are any missing values in the relevant columns
print("\nMissing values in 'AML_subtype' column:")
print(df_sc_res["AML_subtype"].isnull().sum())

