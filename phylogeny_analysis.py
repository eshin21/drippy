import pandas as pd
import os

# %%
# Define file paths
input_file = 'CollecTF_Output_Reports/40_Threshold_OUTPUT_REPORT/40_raw_report_data.xlsx'
output_dir = 'CollecTF_Output_Reports/40_Threshold_OUTPUT_REPORT'
output_file = os.path.join(output_dir, '40_best_result_all_fields.xlsx')

# # Ensure the output directory exists
# os.makedirs(output_dir, exist_ok=True)

# Load the data
df = pd.read_excel(input_file)

# Sort by p_value to ensure the first record is the lowest
df_sorted = df.sort_values(by='p_value', ascending=True)

# Group by the specified columns and pick the first record (lowest p-value)
# reset_index() keeps the dataframe structure clean
best_results = df_sorted.groupby(['Species Protein', 'UniProt ID']).first().reset_index()

# Save to Excel
best_results.to_excel(output_file, index=False)

print(f"Successfully saved best results to {output_file}")



# Assuming 'best_results' is the dataframe from the previous step
# Group by 'Prospective Pattern', select the 'Alignment' column, and count occurrences
alignment_tally = best_results.groupby('Prospective Pattern')['Alignment'].value_counts()

# To display or save the result
print(alignment_tally)


# Tally the Alignment field across the groups of Prospective Pattern
# normalize=True gives the relative frequency (percentage as a decimal)
# Multiplying by 100 converts it to a standard percentage
alignment_tally_pct = best_results.groupby('Prospective Pattern')['Alignment'].value_counts(normalize=True) * 100

# Convert to DataFrame for easier viewing and saving
alignment_tally_df = alignment_tally_pct.reset_index(name='Percentage')


# %%
######################################################################
############## CONFUSION MATRIX ANALYSIS
######################################################################


error_type = 'FP'
direction = 'IR'

# Define file paths
conclusion_file = 'CollecTF_Output_Reports/40_Threshold_OUTPUT_REPORT/40_best_conclusion_pipeline_data.xlsx'

result_file = 'CollecTF_Output_Reports/40_Threshold_OUTPUT_REPORT/40_best_result_all_fields.xlsx'

# Load the datasets
df_conclusion = pd.read_excel(conclusion_file)
df_result = pd.read_excel(result_file)


## EDA

# Drop duplicates based on the specified unique identifiers
unique_entries = df_result.drop_duplicates(subset=['Species Protein', 'UniProt ID'])

# Compute the average of 'Num Sequences' from the unique entries
average_num_sequences = unique_entries['Num Sequences'].mean()

print(f"The average of 'Num Sequences' across unique Species/UniProtID entries is: {average_num_sequences}")



# 1. Filter the conclusion dataset
filtered_conclusion = df_conclusion[df_conclusion[f'Sys_{direction}_Eval (p<=0.05)'] == error_type]

# Select unique keys for filtering
filter_keys = filtered_conclusion[['Species Protein', 'UniProt ID', 'Prospective Pattern']].drop_duplicates()

# 2. Filter the result dataset using an inner merge
# This creates a data frame containing only rows present in both the keys and the result file
search_results = pd.merge(
    df_result, 
    filter_keys, 
    on=['Species Protein', 'UniProt ID', 'Prospective Pattern'], 
    how='inner'
)



search_results.to_excel(f'AlignmentAnalysis/{direction}_{error_type}_CollecTF_40.xlsx')

search_results['Num Sequences'].mean()

# %%
############################################################
#### MEME confusion matrix analysis
############################################################


# %%
# Define file paths
input_file =  'MEME_Output_Reports/MEME_70_Threshold_OUTPUT_REPORT/70_raw_report_data.xlsx'
output_dir =   'MEME_Output_Reports/MEME_70_Threshold_OUTPUT_REPORT/'

output_file = os.path.join(output_dir, '70_best_result_all_fields.xlsx')

# # Ensure the output directory exists
# os.makedirs(output_dir, exist_ok=True)

# Load the data
df = pd.read_excel(input_file)

# Sort by p_value to ensure the first record is the lowest
df_sorted = df.sort_values(by='p_value', ascending=True)

# Group by the specified columns and pick the first record (lowest p-value)
# reset_index() keeps the dataframe structure clean
best_results = df_sorted.groupby(['MEME Folder', 'Motif Number']).first().reset_index()

# Save to Excel
best_results.to_excel(output_file, index=False)

print(f"Successfully saved best results to {output_file}")



# Assuming 'best_results' is the dataframe from the previous step
# Group by 'ManualAnnotation', select the 'Alignment' column, and count occurrences
alignment_tally = best_results.groupby('ManualAnnotation')['Alignment'].value_counts()

# To display or save the result
print(alignment_tally)


# Tally the Alignment field across the groups of ManualAnnotation
# normalize=True gives the relative frequency (percentage as a decimal)
# Multiplying by 100 converts it to a standard percentage
alignment_tally_pct = best_results.groupby('ManualAnnotation')['Alignment'].value_counts(normalize=True) * 100

# Convert to DataFrame for easier viewing and saving
alignment_tally_df = alignment_tally_pct.reset_index(name='Percentage')




#%%


#############################################################


    error_type = 'FP'
    direction = 'IR'

    # Define file paths
    conclusion_file = 'MEME_Output_Reports/MEME_70_Threshold_OUTPUT_REPORT/70_best_conclusion_pipeline_data.xlsx'

    result_file = 'MEME_Output_Reports/MEME_70_Threshold_OUTPUT_REPORT/70_best_result_all_fields.xlsx'

    # Load the datasets
    df_conclusion = pd.read_excel(conclusion_file)
    df_result = pd.read_excel(result_file)


    ## EDA

    # Drop duplicates based on the specified unique identifiers
    unique_entries = df_result.drop_duplicates(subset=['MEME Folder', 'Motif Number'])

    # Compute the average of 'Num Sequences' from the unique entries
    average_num_sequences = unique_entries['Num Sequences'].mean()

    print(f"The average of 'Num Sequences' across unique Species/UniProtID entries is: {average_num_sequences}")



    # 1. Filter the conclusion dataset
    filtered_conclusion = df_conclusion[df_conclusion[f'Sys_{direction}_Eval (p<=0.05)'] == error_type]

    # Select unique keys for filtering
    filter_keys = filtered_conclusion[['MEME Folder', 'Motif Number', 'ManualAnnotation']].drop_duplicates()

    # 2. Filter the result dataset using an inner merge
    # This creates a data frame containing only rows present in both the keys and the result file
    search_results = pd.merge(
        df_result, 
        filter_keys, 
        on=['MEME Folder', 'Motif Number', 'ManualAnnotation'], 
        how='inner'
    )



    search_results.to_excel(f'AlignmentAnalysis/{direction}_{error_type}_MEME_70.xlsx')

    search_results['Num Sequences'].mean()

    # %%
