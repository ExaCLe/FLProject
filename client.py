from flwr.client import NumPyClient
import torch
from model import train, test


# Flower client for federated learning
class GPT2FLClient(NumPyClient):
    def __init__(self, model, trainloader, testloader, device):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters, config):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        train(self.model, self.trainloader, epochs=1, device=self.device)
        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        loss, accuracy = test(self.model, self.testloader, device=self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}
