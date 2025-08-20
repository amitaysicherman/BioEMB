import os
import pandas as pd
from bioemb.data_manager import dataset_to_task_type



def read_csv_with_bug(file_name):
    first_row_data = pd.read_csv(
        file_name,
        header=0,
        nrows=2  # Read only the first row.
    )

    df = pd.read_csv(
        file_name,
        header=None,     # Tell pandas the file has no header row to interpret.
        skiprows=2,      # Skip the first line of the file.
        names=["?"]+list(first_row_data.columns)
    )
    return first_row_data, df



all_tasks = os.listdir("results")
for task_name in all_tasks:
    if not os.path.isdir(os.path.join('results', task_name)):
        continue
    res_file = os.path.join('results', task_name, "evaluation_log.csv")
    first_row, other_rows = read_csv_with_bug(res_file)
    data = pd.read_csv(res_file)
    pre_trained_score = first_row["first_row_data"]
    bioemb_scores = other_rows["first_row_data"]  # TODO: fix the bug in the name.
    if dataset_to_task_type[task_name] == "classification":
        best_score = bioemb_scores.max()
    elif dataset_to_task_type[task_name] == "regression":
        best_score = bioemb_scores.min()
    print(f"Task: {task_name}, Pre-trained Score: {pre_trained_score:.4f}, BioEmb Best Score: {best_score:.4f}")
