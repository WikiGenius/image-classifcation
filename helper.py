# Imports here
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import time
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score


# TODO: Build the network
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, dropout_p=0.2):
        super().__init__()
        # Build the network
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])])
        self.layers.extend([nn.Linear(n1, n2)
                            for n1, n2 in zip(hidden_sizes, hidden_sizes[1:])])
        self.layers.extend([nn.Linear(hidden_sizes[-1], output_size)])
        # Build dropout
        self.drop_out = nn.Dropout(dropout_p)

    def forward(self, x):
        # iterate each layer
        for i, each in enumerate(self.layers):
            if i != len(self.layers) - 1:
                # get output of layer i
                x = each(x)
                # get acctivation relu
                x = F.relu(x)
                # make drop_out with p
                x = self.drop_out(x)
            else:
                # last layer = output layer
                x = each(x)
                x = F.log_softmax(x, dim=1)
        return x


class EarlyStopping:
    """Save the best model during the trainning and finish trainning 
    if there is decrease for valid accuracy in delay epochs"""

    def __init__(self, delay, checkpoint_save="save.pth"):
        # path save chekpoint for the best model during training
        self.checkpoint_save = checkpoint_save
        # delay in number of epochs
        self.delay = delay
        # count continuous decrease in accuracy
        self.count_down = 0
        # record prev valid accuracy
        self.prev_valid_accuracy = None
        # record the best accuracy to save the best model
        self.best_accuracy = None

    def track(self, valid_accuracy, model, train_loss, valid_loss):
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.valid_accuracy = valid_accuracy

        if self.prev_valid_accuracy != None and valid_accuracy <= self.prev_valid_accuracy:
            print("Warning: there is deacrease in valid accuracy")
            self.count_down += 1
        else:
            self.count_down = 0
        if self.best_accuracy == None or valid_accuracy > self.best_accuracy:
            print("Winning: better model")
            # save the best model
            torch.save(model.state_dict(), self.checkpoint_save)
            # update the best accuracy metric
            self.best_accuracy = valid_accuracy
        # update prev_valid_accuracy
        self.prev_valid_accuracy = valid_accuracy
        if self.count_down == self.delay:
            # Finish training, there is continuous decreasing in accuracy
            return True
        else:
            return False

    def get_the_best_model(self):
        state_dict = torch.load(self.checkpoint_save)
        self.model.load_state_dict(state_dict)
        return self.model

    def measurements(self):
        return self.train_loss, self.valid_loss, self.valid_accuracy

# Get classifier based on feature detectors (pre_trainned CNN model)


def get_classifier(pre_trained_model, hidden_sizes, output_size, dropout_p=0.2):
    # freeze the pre_trained parameters
    for param in pre_trained_model.parameters():
        param.requires_grad = False
    input_size = pre_trained_model.classifier[0].state_dict()[
        'weight'].shape[1]
    print(f"input_size of features to the classifier: {input_size}")
    print(f'hidden_sizes in classifier: {hidden_sizes}')
    print(f"output_size of classes from the classifier: {output_size}")
    print()
    # define The network
    classifier = Network(input_size, output_size, hidden_sizes, dropout_p)
    # transfer learning
    pre_trained_model.classifier = classifier
    return pre_trained_model


# TODO: Get preds
def get_predictions(log_ps):
    with torch.no_grad():
        # get exp of log to get probabilities
        ps = torch.exp(log_ps)
        # get top_p and top_class
        top_p, top_class = ps.topk(1, dim=1)
        return top_class


# TODO: Make validation/test inference function
def validation_test(model, validation_test_loader, criterion, gpu_choice):
    """make validation or test inference based on the data"""
    with torch.no_grad():
        # find what is the existed device
        device = torch.device(
            "cuda:0" if gpu_choice and torch.cuda.is_available() else "cpu")
        model.to(device)
        # intial helping variables
        accum_accuracy = 0
        running_loss = 0
        # iterate over the data
        for images, labels in validation_test_loader:
            labels, images = labels.to(device), images.to(device)
            # forward pass
            log_ps = model(images)
            # get predictions
            preds = get_predictions(log_ps)

            # get loss
            loss = criterion(log_ps, labels)

            running_loss += loss.item()
            accum_accuracy += accuracy_score(labels.cpu(), preds.cpu())
        # get running_loss, accuracy metrics
        return running_loss / len(validation_test_loader), accum_accuracy / len(validation_test_loader)


# TODO: define Train function
def train(model, optimizer, criterion, early_stopping, trainloader, validloader, gpu, epochs=5, print_every=40):
    # find what is the existed device
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    model.to(device)
    # intial helping variables
    train_loss_container = []
    valid_loss_container = []
    steps = 0
    # loop over epochs
    for e in range(epochs):
        # intial helping variables
        running_loss = 0
        # loop over batchs of trainloader
        for images, labels in trainloader:
            model.train()
            steps += 1
            labels, images = labels.to(device), images.to(device)

            # clear gradient
            optimizer.zero_grad()
            # forward pass
            log_ps = model(images)
            # get loss
            loss = criterion(log_ps, labels)
            # backward pass
            loss.backward()
            # update weights by making step for optimizer
            optimizer.step()

            running_loss += loss.item()
            # valid condition every print_every
            if steps % print_every == 0:
                model.eval()
                train_loss = running_loss / print_every
                valid_loss, valid_accuracy = validation_test(
                    model, validloader, criterion, gpu)
                train_loss_container.append(train_loss)
                valid_loss_container.append(valid_loss)
                running_loss = 0
                # print_results
                print(f"{e+1}/{epochs} .. train_loss: {(train_loss) :0.3f}\
.. valid_loss: {(valid_loss) :0.3f} .. valid_accuracy: {(valid_accuracy * 100) :0.3f}%")

        if early_stopping.track(valid_accuracy, model, train_loss, valid_loss):
            print("Early stopping")
            print("Having the best model")
            model = early_stopping.get_the_best_model()
            train_loss, valid_loss, valid_accuracy = early_stopping.measurements()
            print(
                f".. train_loss: {(train_loss) :0.3f}.. valid_loss: {(valid_loss) :0.3f} .. valid_accuracy: {(valid_accuracy * 100) :0.3f}%")
            break

    # plot train_loss and valid_loss
    plt.plot(train_loss_container,  label="Train loss")
    plt.plot(valid_loss_container,  label="Valid loss")
    plt.legend()
    plt.show()
    print("Having the best model")
    model = early_stopping.get_the_best_model()
    train_loss, valid_loss, valid_accuracy = early_stopping.measurements()
    print(
        f".. train_loss: {(train_loss) :0.3f}.. valid_loss: {(valid_loss) :0.3f} .. valid_accuracy: {(valid_accuracy * 100) :0.3f}%")
    return train_loss, valid_loss, valid_accuracy
