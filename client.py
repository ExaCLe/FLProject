import wandb
from flwr.client import NumPyClient
import torch
from model import train, test
import os
import json


# Flower client for federated learning
class GPT2FLClient(NumPyClient):
    def __init__(
        self,
        model,
        trainloader,
        testloader,
        device,
        client_id=None,
        wandb_group=None,
        experiment_name="federated_xnli",
    ):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.client_id = client_id
        self.wandb_group = wandb_group
        self.experiment_name = experiment_name
        self.wandb_run = None
        self.wandb_enabled = client_id is not None and wandb_group is not None
        self.run_id = None

        # Ensure model is on correct device
        self.model = self.model.to(device)

        # Create a directory to store run IDs if it doesn't exist
        os.makedirs("./.wandb_runs", exist_ok=True)
        self.run_id_file = f"./.wandb_runs/{client_id}.json" if client_id else None

        if self.wandb_enabled:
            # Try to load existing run ID
            if self.run_id_file and os.path.exists(self.run_id_file):
                with open(self.run_id_file, "r") as f:
                    self.run_id = json.load(f)["run_id"]

            # Initialize wandb with existing run ID or create new run
            self.wandb_run = wandb.init(
                project="federated-xnli",
                name=f"client_{client_id}",
                group=wandb_group,
                id=self.run_id,  # Use existing run ID if available
                settings=wandb.Settings(start_method="thread"),
                reinit=True,
                resume="allow",
            )

            # Save the run ID for future use
            if self.run_id_file and not self.run_id:
                with open(self.run_id_file, "w") as f:
                    json.dump({"run_id": wandb.run.id}, f)  # type: ignore
                self.run_id = wandb.run.id  # type: ignore

    def get_parameters(self, config):
        """Get parameters from the model, ensuring they're on CPU."""
        return [
            val.cpu().detach().numpy() for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters, config):
        """Set parameters in the model, handling device placement carefully."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device="cpu") for k, v in params_dict}

        # Load state dict with explicit device mapping
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in state_dict.items()}, strict=True
        )

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        round_num = config.get("round_num", 0)

        metrics = train(self.model, self.trainloader, epochs=1, device=self.device)

        if self.wandb_enabled:
            wandb.log(
                {
                    "train/loss": metrics["train_loss"],
                    "train/accuracy": metrics["train_accuracy"],
                    "round": round_num,
                }
            )

        return self.get_parameters(config), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        loss, accuracy = test(self.model, self.testloader, device=self.device)

        if self.wandb_enabled:
            round_num = config.get("round_num", 0)
            wandb.log(
                {
                    "eval/loss": loss,
                    "eval/accuracy": accuracy,
                    "round": round_num,
                }
            )

        return loss, len(self.testloader.dataset), {"accuracy": accuracy}

    def __del__(self):
        """Cleanup wandb run when the client is destroyed"""
        if hasattr(self, "wandb_run") and self.wandb_run:
            self.wandb_run.finish()
            # Don't delete the run ID file as it needs to persist across runs
