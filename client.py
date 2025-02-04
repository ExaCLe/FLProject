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
        client_id: str,
        sa_interval: float,
        sa_epochs: int,
        sa_samples: int,
        language: str,
        tokenizer,
    ):
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.client_id = client_id
        self.sa_interval = sa_interval
        self.sa_epochs = sa_epochs
        self.sa_samples = sa_samples
        self.language = language
        self.tokenizer = tokenizer

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
        lr = config.get("learning_rate", 5e-5)

        # Use the train function with all necessary parameters
        metrics = train(
            self.model,
            self.trainloader,
            epochs=1,
            device=self.device,
            learning_rate=lr,  # type: ignore
            sa_interval=self.sa_interval,
            sa_samples=self.sa_samples,
            sa_epochs=self.sa_epochs,
            language=self.language,
            tokenizer=self.tokenizer,
        )

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
        return (
            loss,
            len(self.testloader.dataset),
            {
                "client_id": self.client_id,
                "accuracy": accuracy,
            },
        )
