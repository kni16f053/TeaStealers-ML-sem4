import argparse

import mlflow

from product_classifier.params import Settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train_and_evaluate")
    parser.add_argument("commit", nargs=1, type=str)
    commit = parser.parse_args().commit[0]

    with open(Settings.tracking_commit_file, "r") as f:
        run_id = f.read()

    mlflow.set_tracking_uri(Settings.mlflow_uri)
    mlflow.start_run(run_id=run_id, nested=True)
    mlflow.log_param("commit", commit)
