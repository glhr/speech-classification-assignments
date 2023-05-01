# ###################################
# Group ID : 530
# Members : Rahabu Mwang’amba, Yurii Iotov, Galadrielle Humblot-Renaux
# Date : 28/4/2023
# Lecture: 1 Feedforward neural networks
# Dependencies: Pytorch Lightning 2.0.2, Torchvision 0.15.1, Pandas, Matplotlib, Numpy
# Python version: 3.9.15
# Functionality: Training a fully connected neural network to classify audio samples into three classes.
# Script includes a calculation of the number of parameters in the model, plotting of training curves,
# and performance evaluation on the test set.
# ###################################

import os
from torch import optim, nn, utils
import lightning.pytorch as pl
import torch.utils.data as data
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

pl.seed_everything(0, workers=True) # set random seed for reproducibility

# define the LightningModule
class LitModel(pl.LightningModule):
    def __init__(self, input_size=101*40, num_classes=3):
        super().__init__()

        ## define the model architecture
        ## the encoder extracts features from the input data
        ## the decoder classifies the features (output layer)
        self.encoder = nn.Sequential(
                        nn.Linear(input_size, 128), nn.ReLU(),
                        nn.Linear(128, 128), nn.ReLU(),
                        nn.Linear(128, 128), nn.ReLU(),
                        nn.Linear(128, 128), nn.ReLU(),
                    )
        self.decoder = nn.Sequential( nn.Linear(128, num_classes))

        self.num_classes = num_classes

    def get_loss(self, x, y):
        # loss between the prediction x and the one-hot encoded target y
        return F.cross_entropy(x, y)
    
    def get_batch(self, batch):
        # extract the input sample and the target from a batch of data
        x, y = batch
        x = x.view(x.size(0), -1) # flatten the input into a 1D vector
        return x, y
    
    def calculate_accuracy(self, x_hat, y):
        pred_cls = torch.argmax(x_hat, dim=1) # get the index of the class with the highest probability
        gt_cls = torch.argmax(y, dim=1) # ground truth class
        acc = torch.sum(pred_cls == gt_cls) / float(len(gt_cls))
        return acc

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop
        x, y = self.get_batch(batch)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.get_loss(x_hat, y)
        # Logging to CSV file
        self.log("train_loss", loss, on_epoch=True, on_step=True)
        
        # calculate classification accuracy
        acc = self.calculate_accuracy(x_hat, y)
        self.log("train_acc", acc, on_epoch=True, on_step=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop (executed after every training epoch)
        x, y = self.get_batch(batch)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss =self.get_loss(x_hat, y)
        self.log("val_loss", val_loss, on_epoch=True, on_step=False)

        # calculate classification accuracy
        acc = self.calculate_accuracy(x_hat, y)
        self.log("val_acc", acc, on_epoch=True, on_step=False)
    
    def test_step(self, batch, batch_idx):
        # this is the test loop (used only after training has been completed and we want to evaluate the model on the test set)
        x, y = self.get_batch(batch)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = self.get_loss(x_hat, y)
        self.log("test_loss", test_loss)

        # calculate classification accuracy
        acc = self.calculate_accuracy(x_hat, y)
        self.log("test_acc", acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        # this is the optimizer used to update weights (Stochastic Gradient Descent)
        optimizer = optim.SGD(self.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)
        return optimizer


# initialize the model
model = LitModel()

# calculate the number of parameters in the model and check that it is the same as the one computed in the report
manually_calculated_no_params = 567171
def get_no_params(model):
 nop = 0
 for param in list(model.parameters()):
    nn = 1
    for s in list(param.size()):
        nn = nn * s
    nop += nn
 return nop
nop = get_no_params(model)
assert nop == manually_calculated_no_params, f"We messed up (calculated {manually_calculated_no_params} parameters, but it should be {nop})"

## load samples (X) and labels (y) for train, validation and test sets
X_train = np.load(f"DL_Data/X_train.p", allow_pickle=True)
y_train = np.load(f"DL_Data/Y_train.p", allow_pickle=True)

X_valid = np.load(f"DL_Data/X_valid.p", allow_pickle=True)
y_valid = np.load(f"DL_Data/Y_valid.p", allow_pickle=True)

X_test = np.load(f"DL_Data/X_test.p", allow_pickle=True)
y_test = np.load(f"DL_Data/Y_test.p", allow_pickle=True)

## display a sample from the training set, uncomment to run
# sample = X_train[0]
# plt.imshow(sample.numpy(), cmap="gray")
# plt.show()

## compute training data statistics and use them to normalize the data
train_mean = np.mean(X_train)
train_std = np.std(X_train)
print("Training data - mean: {}, std: {}".format(train_mean, train_std))

def normalize_input_data(data, mean, std):
    return (data - mean) / std

X_train = normalize_input_data(X_train, train_mean, train_std)
X_valid = normalize_input_data(X_valid, train_mean, train_std)
X_test = normalize_input_data(X_test, train_mean, train_std)

## create PyTorch datasets from the input and label data
# see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html for more information
data_transform = transforms.Compose([transforms.ToTensor()])

class OurDataset(data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index].astype(np.float32)
        y = self.y[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
train_set = OurDataset(X_train, y_train, transform=data_transform)
valid_set = OurDataset(X_valid, y_valid, transform=data_transform)
test_set = OurDataset(X_test, y_test, transform=data_transform)

## setup data loaders for iterating over the datasets in batches
BATCH_SIZE = 1024
train_loader = utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = utils.data.DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

## train the model
trainer = pl.Trainer(devices=1, max_epochs=50, log_every_n_steps=1) # define the training configuration
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader) # run the training

## load the results file and plot training & validation curves
log_dir = model.logger.experiment.log_dir # note results are stored in this folder (by default "lightning_logs/version_X")
print("log_dir: ", log_dir)

# read csv file metrics.csv
df = pd.read_csv(os.path.join(log_dir, "metrics.csv"))
# select only the columns we need
df_train = df[["step", "train_loss_step", "train_acc_step"]]
df_val = df[["epoch", "val_loss", "val_acc"]]
df_train = df.groupby("step").mean()
df_val = df.groupby("epoch").mean()
df_val["step"] = 10 + df_val.index * 10 # convert epoch to step (10 steps per epoch)
df_val.index = df_val["step"]

# plot train and validation loss in one subplot and the accuracy in another
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
# plot accuracy
ax[0].plot(df_train.index,df_train["train_acc_step"], label="Training accuracy")
ax[0].plot(df_val.index,df_val["val_acc"], label="Validation accuracy")
ax[0].set_title("Accuracy")
ax[0].set_xlabel("Training iteration")
ax[0].legend()

# plot loss
ax[1].plot(df_train.index,df_train["train_loss_step"], label="Training loss")
ax[1].plot(df_val.index,df_val["val_loss"], label="Validation loss")
ax[1].set_title("Loss")
ax[1].set_xlabel("Training iteration")
ax[1].legend()
plt.savefig(os.path.join(log_dir, "loss.png"))
plt.show()

# evaluate the model on the test set to get the test accuracy
trainer.test(model=model, dataloaders=test_loader)