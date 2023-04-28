import os
from torch import optim, nn, utils, Tensor
import lightning.pytorch as pl
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.utils.data as data
import torch

import torch.nn.functional as F

pl.seed_everything(0, workers=True)

# define the LightningModule
class LitModel(pl.LightningModule):
    def __init__(self, input_size=28*28, num_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
                        nn.Linear(28 * 28, 128), nn.ReLU(),
                        nn.Linear(128, 128), nn.ReLU(),
                        nn.Linear(128, 128), nn.ReLU(),
                        nn.Linear(128, 128), nn.ReLU(),
                    )
        self.decoder = nn.Sequential( nn.Linear(128, num_classes), nn.Softmax(dim=1))

        self.num_classes = num_classes

    def get_loss(self, x, y):
        return F.cross_entropy(x, y)
    
    def get_batch(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        y = F.one_hot(y, num_classes=self.num_classes).float()
        return x, y

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = self.get_batch(batch)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.get_loss(x_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = self.get_batch(batch)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss =self.get_loss(x_hat, y)
        self.log("val_loss", val_loss, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = self.get_batch(batch)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = self.get_loss(x_hat, y)
        self.log("test_loss", test_loss)

        # calculate classification accuracy
        #print(x_hat.shape, y.shape)
        pred_cls = torch.argmax(x_hat, dim=1)
        gt_cls = torch.argmax(y, dim=1)
        acc = torch.sum(pred_cls == gt_cls) / float(len(gt_cls))
        self.log("test_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)
        return optimizer


# init the autoencoder
autoencoder = LitModel()

# setup data
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
test_set = MNIST(os.getcwd(), download=True, train=False, transform=ToTensor())
# use 20% of training data for validation
train_set_size = int(len(dataset) * 0.8)
valid_set_size = len(dataset) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)

BATCH_SIZE = 1024
train_loader = utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


# train the model
trainer = pl.Trainer(devices=1, max_epochs=50)
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)

# test 
trainer.test(model=autoencoder, dataloaders=test_loader)