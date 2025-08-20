import os
import pandas as pd
from bioemb.data_manager import dataset_to_task_type


def read_csv_with_bug(file_name):
    first_row_data = pd.read_csv(
        file_name,
        header=0,
        nrows=1  # Read only the first row.
    )

    df = pd.read_csv(
        file_name,
        header=None,  # Tell pandas the file has no header row to interpret.
        skiprows=2,  # Skip the first line of the file.
        names=["?"] + list(first_row_data.columns)
    )
    return first_row_data, df


all_tasks = os.listdir("results")
for dir_name in all_tasks:
    if not os.path.isdir(os.path.join('results', dir_name)):
        continue
    res_file = os.path.join('results', dir_name, "evaluation_log.csv")
    if not os.path.exists(res_file):
        print(f"Results file for {dir_name} does not exist.")
        continue
    first_row, other_rows = read_csv_with_bug(res_file)
    pre_trained_score = first_row["eval_downstream_auc"].values[0]
    bioemb_scores_test = other_rows["eval_downstream_auc"]
    bioemb_scores_valid = other_rows["eval_downstream_auc_valid"]
    task_name = dir_name.split("~")[0]
    task_type = dataset_to_task_type[task_name]
    if task_type == "classification":
        best_index = bioemb_scores_valid.idxmax()
        best_score = bioemb_scores_test[best_index]
    elif task_type == "regression":
        best_index = bioemb_scores_valid.idxmin()
        best_score = bioemb_scores_test[best_index]

    print(
        f"Task: {dir_name}[{task_type}], Pre-trained Score: {pre_trained_score:.4f}, BioEmb Best Score: {best_score:.4f}")
