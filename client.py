from flwr.client import NumPyClient
import torch
from model import train, test


class FLClient(NumPyClient):
    """A Flower client implementing federated learning for LLMs.

    This client handles the training and evaluation of LLM in a federated learning setup,
    supporting self-attention analysis during training.

    Args:
        model: The LLM to be trained
        trainloader: DataLoader for training data
        testloader: DataLoader for test data
        device: Device to run the model on (cuda/cpu)
        client_id (str): Unique identifier for the client
        sa_interval (float): Interval for self-attention analysis
        sa_epochs (int): Number of epochs for self-attention analysis
        sa_samples (int): Number of samples for self-attention analysis
        language (str): Language of the training data
        tokenizer: Tokenizer for processing text data
        epochs_per_client (int): Number of training epochs per federated round
    """

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
        epochs_per_client: int,
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
        self.epochs_per_client = epochs_per_client

    def get_parameters(self, config):
        """Get the model's parameters as a list of NumPy arrays.

        Args:
            config: Configuration parameters (unused)

        Returns:
            list: Model parameters as NumPy arrays
        """
        return [
            val.cpu().detach().numpy() for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters, config):
        """Set the model's parameters from a list of NumPy arrays.

        Args:
            parameters: List of model parameters as NumPy arrays
            config: Configuration parameters (unused)
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v, device="cpu") for k, v in params_dict}
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in state_dict.items()}, strict=True
        )

    def fit(self, parameters, config):
        """Train the model using the provided parameters.

        Args:
            parameters: List of model parameters for training
            config: Configuration including learning rate

        Returns:
            tuple: (Updated model parameters,
                   Number of training examples used,
                   Dict containing training metrics)
        """
        self.set_parameters(parameters, config)
        lr = config.get("learning_rate", 5e-5)

        metrics = train(
            self.model,
            self.trainloader,
            epochs=self.epochs_per_client,
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
        """Evaluate the model using the provided parameters.

        Args:
            parameters: List of model parameters for evaluation
            config: Configuration parameters (unused)

        Returns:
            tuple: (Loss value,
                   Number of test examples used,
                   Dict containing evaluation metrics)
        """
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
