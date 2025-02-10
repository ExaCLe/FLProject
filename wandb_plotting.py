import wandb
import matplotlib.pyplot as plt


def analyze_sweep_runs(sweep_path, metric_name, num_epochs):
    """
    Analyzes runs from a wandb sweep.

    Parameters:
        sweep_path (str): The full sweep path, e.g. "entity/project/sweep_id".
        metric_name (str): The name of the metric to plot and evaluate.
        num_epochs (int): The number of epochs to consider (plots values up to this epoch and evaluates best value at the last logged epoch ≤ num_epochs).

    Returns:
        best_run (wandb.apis.public.Run): The run object with the best metric value at the cutoff epoch.
    """

    # Initialize the wandb API.
    api = wandb.Api()

    # Retrieve the sweep object.
    try:
        sweep = api.sweep(sweep_path)
    except Exception as e:
        print(f"Error retrieving sweep {sweep_path}: {e}")
        return None

    # Get all runs in the sweep.
    runs = sweep.runs
    if not runs:
        print("No runs found in the sweep.")
        return None

    # Set up the plot.
    plt.figure(figsize=(10, 6))

    best_run = None
    best_metric_value = None

    # Iterate over each run in the sweep.
    for run in runs:
        # Get run history as a pandas DataFrame.
        try:
            df = run.history(pandas=True)
        except Exception as e:
            print(f"Could not retrieve history for run {run.id}: {e}")
            continue

        # Ensure the run logged the 'epoch' column.
        if "round" not in df.columns:
            print(f"Run {run.id} does not have an 'round' column; skipping.")
            continue

        # Filter to only include logs up to the desired epoch.
        df_cut = df[df["round"] <= num_epochs]
        if df_cut.empty:
            print(f"Run {run.id} has no logs up to epoch {num_epochs}; skipping.")
            continue

        # Ensure that the target metric is present.
        if metric_name not in df_cut.columns:
            print(f"Run {run.id} does not log the metric '{metric_name}'; skipping.")
            continue

        # Plot the metric vs. epoch for this run.
        plt.plot(df_cut["round"], df_cut[metric_name], label=run.id)
        print(df_cut.head())

        # Identify the last logged epoch (closest to num_epochs) and its metric value.
        last_epoch = df_cut["round"].max()
        # It’s possible that multiple rows have the same epoch value; take the last one.
        final_value = df_cut.loc[df_cut["round"] == last_epoch, metric_name].iloc[-1]

        # For this example, "best" means highest metric value.
        if best_metric_value is None or final_value > best_metric_value:
            best_metric_value = final_value
            best_run = run

    # Finalize and show the plot.
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} over epochs (up to epoch {num_epochs})")
    plt.legend(loc="best", fontsize="small", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Report the best run.
    if best_run is not None:
        print(f"\nBest run: {best_run.id}")
        print(f"{metric_name} = {best_metric_value} at epoch (≤ {num_epochs})")
    else:
        print("No run with valid metric data was found.")

    return best_run


# Example usage:
if __name__ == "__main__":
    # Replace with your own values:
    sweep_id = (
        "exacle-/federated-xnli/itu0jt6g"  # e.g., "myusername/myproject/abc12345"
    )
    metric = "aggregated/eval_accuracy"  # Change to your metric name
    epochs = 10  # Number of epochs to consider

    best_run = analyze_sweep_runs(sweep_id, metric, epochs)
