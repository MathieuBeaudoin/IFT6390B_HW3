import random
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, NamedTuple
from tqdm import tqdm
import torchvision
from torchvision import transforms

# Seed all random number generators
np.random.seed(197331)
torch.manual_seed(197331)
random.seed(197331)


class NetworkConfiguration(NamedTuple):
    n_channels: Tuple[int, ...] = (16, 32, 48)
    kernel_sizes: Tuple[int, ...] = (3, 3, 3)
    strides: Tuple[int, ...] = (1, 1, 1)
    dense_hiddens: Tuple[int, ...] = (256, 256)

class ActivationAsModule(nn.Module):
    """Basically inspired from the example in pytorch documentation at
    https://pytorch.org/docs/stable/generated/torch.nn.Module.html"""
    def __init__(self, activation):
        super().__init__()
        self.activation = activation
    def forward(self, x):
        return self.activation(x)


class Trainer:

    def __init__(self,
                 network_type: str = "mlp",
                 net_config: NetworkConfiguration = NetworkConfiguration(),
                 lr: float = 0.001,
                 batch_size: int = 128,
                 activation_name: str = "relu"):
        self.lr = lr
        self.batch_size = batch_size
        self.train, self.test = self.load_dataset(self)
        dataiter = iter(self.train)
        images, labels = next(dataiter)
        input_dim = images.shape[1:]
        self.network_type = network_type
        activation_function = self.create_activation_function(activation_name)
        if network_type == "mlp":
            self.network = self.create_mlp(
                input_dim[0]*input_dim[1]*input_dim[2], 
                net_config,
                activation_function
            )
        elif network_type == "cnn":
            self.network = self.create_cnn(
                input_dim[0], 
                net_config, 
                activation_function
            )
        else:
            raise ValueError("Network type not supported")
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.train_logs = {'train_loss': [], 'test_loss': [],
                           'train_mae': [], 'test_mae': []}

    @staticmethod
    def load_dataset(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                shuffle=True)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                shuffle=False)

        return trainloader, testloader

    @staticmethod
    def append_dense(model: torch.nn.Module,
                     input_dim: int,
                     net_config: NetworkConfiguration,
                     activation: torch.nn.Module,
                     activate_last: bool = True) -> None:
        model.append(nn.Flatten())
        n_layers = len(net_config.dense_hiddens)
        for i, layer_size in enumerate(net_config.dense_hiddens):
            model.append(nn.Linear(input_dim, layer_size))
            if activate_last or i < n_layers - 1:
                model.append(activation)
            input_dim = layer_size
        model.append(nn.Linear(input_dim, 1))

    @staticmethod
    def create_mlp(input_dim: int, 
                   net_config: NetworkConfiguration,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a multi-layer perceptron (MLP) network.

        :param net_config: a NetworkConfiguration named tuple. 
            Only the field 'dense_hiddens' will be used.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the MLP.
        """
        model = nn.Sequential()
        Trainer.append_dense(
            model = model,
            input_dim = input_dim,
            net_config = net_config,
            activation = activation
        )
        return model

    @staticmethod
    def create_cnn(in_channels: int, 
                   net_config: NetworkConfiguration,
                   activation: torch.nn.Module) -> torch.nn.Module:
        """
        Create a convolutional network.

        :param in_channels: The number of channels in the input image.
        :param net_config: a NetworkConfiguration specifying the architecture of the CNN.
        :param activation: The activation function to use.
        :return: A PyTorch model implementing the CNN.
        """
        final_pool_shape = (4, 4)
        model = nn.Sequential()
        # Convolutional layers
        _iterator = zip(
            net_config.n_channels,
            net_config.kernel_sizes,
            net_config.strides
        )
        n_conv_layers = len(net_config.n_channels)
        for i, (out_channels, ksize, stride) in enumerate(_iterator):
            model.append(nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = ksize,
                stride = stride
            ))
            model.append(activation)
            in_channels = out_channels
            if i < n_conv_layers - 1:
                pooling = nn.MaxPool2d(kernel_size=2)
            else:
                pooling = nn.AdaptiveMaxPool2d(final_pool_shape)
            model.append(pooling)
        # Dense layers
        Trainer.append_dense(
            model = model,
            input_dim = np.prod(final_pool_shape) * net_config.n_channels[-1],
            net_config = net_config,
            activation = activation,
            activate_last = False
        )
        return model

    @staticmethod
    def create_activation_function(activation_str: str) -> torch.nn.Module:
        functions = {
            "relu": nn.functional.relu,
            "tanh": nn.functional.tanh,
            "sigmoid": nn.functional.sigmoid,
        }
        if activation_str not in functions:
            raise ValueError(f"'{activation_str}' is not a valid argument")
        return ActivationAsModule(functions[activation_str])

    def compute_loss_and_mae(self, 
                             X: torch.Tensor, 
                             y: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        predicted = self.network.forward(X)
        return (
            nn.MSELoss()(predicted, y),
            nn.L1Loss()(predicted, y)
        )

    def training_step(self, 
                      X_batch: torch.Tensor, 
                      y_batch: torch.Tensor):
        # Partially reusing: https://pytorch.org/tutorials/beginner
        # /introyt/trainingyt.html#the-training-loop
        self.optimizer.zero_grad()
        loss, mae = self.compute_loss_and_mae(X_batch, y_batch)
        loss.backward()
        self.optimizer.step()
        return loss, mae

    def train_loop(self, n_epochs: int) -> dict:
        N = len(self.train)
        for epoch in tqdm(range(n_epochs)):
            train_loss = 0.0
            train_mae = 0.0
            for i, data in enumerate(self.train):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                loss, mae = self.training_step(inputs, labels)
                train_loss += loss
                train_mae += mae
            # Log data every epoch
            self.train_logs['train_mae'].append(train_mae / N)
            self.train_logs['train_loss'].append(train_loss / N)
            self.evaluation_loop()    
        return self.train_logs

    def evaluation_loop(self) -> None:
        self.network.eval() # Disables gradient computation
        N = len(self.test)
        with torch.inference_mode():
            test_loss = 0.0
            test_mae = 0.0
            for data in self.test:
                inputs, labels = data
                loss, mae = self.compute_loss_and_mae(inputs, labels)
                test_loss += loss.item()
                test_mae += mae.item()
        self.train_logs['test_mae'].append(test_mae / N)
        self.train_logs['test_loss'].append(test_loss / N)

    def evaluate(self, 
                 X: torch.Tensor, 
                 y: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.network.eval()
        with torch.inference_mode():
            loss, mae = self.compute_loss_and_mae(X, y)
        return torch.mean(loss), torch.mean(mae)
