import os

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optimizers
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

SEED = 42


class AutoEncoder(nn.Module):
    def __init__(self, device='cpu') -> None:
        super().__init__()
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(200, 784),
            nn.Sigmoid()
        )

    def forward(self, x):

        # encoding
        h = self.encoder(x)
        # decoding
        return self.decoder(h)


if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(
        'mps') if torch.backends.mps.is_available() else 'cpu'
    device = 'cpu'

    root = os.path.join(os.path.dirname(__file__),
                        '.', 'data', 'fashion_mnist')
    transform = transforms.Compose([transforms.ToTensor(),
                                    lambda x: x.view(-1)])
    mnist_train = torchvision.datasets.FashionMNIST(
        root=root,
        download=True,
        train=True,
        transform=transform
    )

    mnist_test = torchvision.datasets.FashionMNIST(
        root=root,
        download=True,
        train=False,
        transform=transform
    )
    train_dataloader = DataLoader(mnist_train,
                                  batch_size=100,
                                  shuffle=True)
    test_dataloader = DataLoader(mnist_test,
                                 batch_size=1,
                                 shuffle=False)

    model = AutoEncoder(device=device)

    criterion = nn.BCELoss()
    optimizer = optimizers.Adam(model.parameters())

    def compute_loss(x, preds):
        return criterion(preds, x)

    def train_step(x):
        model.train()
        preds = model(x)
        loss = compute_loss(x, preds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    epochs = 10
    for epoch in range(epochs):
        train_loss = 0.0
        for (x, _) in train_dataloader:
            x = x.to(device)
            loss = train_step(x)
            train_loss += loss
        train_loss /= len(train_dataloader)
        print('Epoch: {}, Cost: {:.3f}'.format(
            epoch+1,
            train_loss
        ))

    x, _ = next(iter(test_dataloader))
    noise = torch.bernoulli(0.8 * torch.ones(x.size())).to(device)
    x_noise = x * noise
    x_reconstructed = model(x_noise)

    plt.figure(figsize=(18, 6))
    for i, image in enumerate([x, x_noise, x_reconstructed]):
        image = image.view(28, 28).detach().cpu().numpy()
        plt.subplot(1, 3, i+1)
        plt.imshow(image, cmap='binary_r')
        plt.axis('off')
    plt.show()
