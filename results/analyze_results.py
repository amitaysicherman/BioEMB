import os
import pandas as pd
from bioemb.data_manager import dataset_to_task_type

all_tasks = os.listdir("results")
for task_name in all_tasks:
    if not os.path.isdir(os.path.join('results', task_name)):
        continue
    res_file = os.path.join('results', task_name, "evaluation_log.csv")
    data = pd.read_csv(res_file)
    first_row = data.iloc[0]
    pre_trained_score = first_row["eval_downstream_auc"]
    other_rows = data.iloc[1:]
    bioemb_scores = other_rows["eval_loss"]  # TODO: fix the bug in the name.
    if dataset_to_task_type[task_name] == "classification":
        best_score = bioemb_scores.max()
    elif dataset_to_task_type[task_name] == "regression":
        best_score = bioemb_scores.min()
    print(f"Task: {task_name}, Pre-trained Score: {pre_trained_score:.4f}, BioEmb Best Score: {best_score:.4f}")
