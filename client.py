from flwr.client import NumPyClient
import torch
from model import train, test


class GPT2FLClient(NumPyClient):
    def __init__(
        self,
        model,
        trainloader,
        testloader,
        device,
        client_id=None,
    ):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.client_id = client_id

        # Ensure model is on correct device
        self.model = self.model.to(device)

    def get_parameters(self, config):
        return [
            val.cpu().detach().numpy() for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters, config):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device="cpu") for k, v in params_dict}
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in state_dict.items()}, strict=True
        )

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        metrics = train(self.model, self.trainloader, epochs=1, device=self.device)

        # Return parameters, number of datapoints, and metrics
        return (
            self.get_parameters(config),
            len(self.trainloader.dataset),
            {
                "client_id": self.client_id,
                "train_loss": metrics["train_loss"],
                "train_accuracy": metrics["train_accuracy"],
            },
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        loss, accuracy = test(self.model, self.testloader, device=self.device)

        # Return loss, number of datapoints, and metrics
        return (
            loss,
            len(self.testloader.dataset),
            {
                "client_id": self.client_id,
                "accuracy": accuracy,
            },
        )
