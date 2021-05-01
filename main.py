from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.manifold import TSNE

import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''

class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''
    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=1)
        self.b1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.b2 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.b1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # kernel = stride = 2

        x = self.conv2(x)
        x = self.b2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) #kernel = stride = 2

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        # Softmax outputs a probability distribution (sum to 1)
        # Needed for using NLLoss
        output = F.log_softmax(x, dim=1)
        return output

def imshow(img):
    img = (img*0.3081) + 0.1307     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_num += len(data)

    test_loss /= test_num

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))

    return test_loss


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--problem7', action='store_true', default=False)
    parser.add_argument('--problem8', action='store_true', default=False)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    if args.problem8:

        # Taken from https://stackoverflow.com/questions/58589349/pytorch-confusion-matrix-plot
        def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          normalize=False,
                          cmap=plt.cm.Blues):

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')


        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))
        feature_vectors = []
        model.fc1.register_forward_hook(lambda model, input_, output: feature_vectors.extend(output.tolist()))

        # Visualize first layer kernels
        fig, ax = plt.subplots(3, 3, figsize=(12, 12))
        filters = model.conv1.weight

        for i in range(8):
            filter = filters[i][0]
            ax[i // 3, i % 3].imshow(filter.detach().numpy(), cmap='gray')

        plt.savefig("kernels.png")

        # Import test data
        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        model.eval()    # Set the model to inference mode

        # Find mistakes and get all predictions
        mistakes = []
        y_pred = []
        test_images = []
        y_test = []
        with torch.no_grad():   # For the inference step, gradient is not computed
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                y_pred += pred.tolist()
                y_test += target.tolist()
                correct = target.eq(pred.view_as(target))
                for index, result in enumerate(correct):
                    if result == False:
                        mistakes.append([data[index], target[index], pred[index]])

        # Plot T-SNE
        tsne = TSNE().fit_transform(feature_vectors)
        plt.figure()
        plt.scatter(tsne[:,0], tsne[:,1], c=y_pred)
        plt.colorbar()
        plt.savefig("tsne.png")

        # Show nearest T-SNE neighbors
        feature_vectors = [(vector, i) for i,vector in enumerate(feature_vectors)]
        fig, ax = plt.subplots(4,8)
        for i in range(4):
            x = np.array(feature_vectors[i][0])
            closest = sorted(feature_vectors, key=lambda n: (np.linalg.norm(np.array(n[0]) - x)))
            closest8_vec = closest[1:9]
            closest8_img = [test_dataset[v[1]] for v in closest8_vec]
            for j in range(8):
                ax[i,j].imshow(closest8_img[j][0][0].numpy())

        plt.savefig('closest.png')



        # Plot confusion matrix
        plt.figure()
        cm = confusion_matrix(y_pred, y_test)
        plot_confusion_matrix(cm, range(0,10))
        plt.savefig('confusion_matrix.png')

        # Visualize mistakes
        fig, ax = plt.subplots(3, 3, figsize=(12, 12))
        for i,pos in enumerate(random.sample(range(len(mistakes)), 9)):
            img = mistakes[pos][0]
            img = (img*0.3081) + 0.1307     # unnormalize
            img = img.numpy()
            img = np.transpose(img, (1, 2, 0))

            ax[i // 3, i % 3].imshow(img, cmap='gray')
            ax[i // 3, i % 3].set_title(f'True Label: {mistakes[pos][1].item()}, Predicted: {mistakes[pos][2].item()}')
        plt.savefig("mistakes.png")

        return

    if args.problem7:
        proportions = [0.5, 0.25, 0.125, 0.0625]
        test_errors = []
        train_errors = []

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        # Pytorch has default MNIST dataloader which loads data at each iteration
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([       # Data preprocessing
                        transforms.ToTensor(),           # Add data augmentation here
                        transforms.RandomApply([
                            transforms.RandomRotation(30),
                            transforms.GaussianBlur(3, sigma=(0.1,0.5))
                        ], p=0.3),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        for prop in proportions:
            print("Training on {}% of the data".format(100*prop))

            # Get proportion of test data
            subset_indices_train = random.sample(range(len(train_dataset)),
                                                int(len(train_dataset)*prop))

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size,
                sampler=SubsetRandomSampler(subset_indices_train)
            )

            # Load your model [fcNet, ConvNet, Net]
            model = Net().to(device)

            # Try different optimzers here [Adam, SGD, RMSprop]
            optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

            # Set your learning rate scheduler
            scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

            epochs = range(1, args.epochs + 1)

            # Training loop
            for epoch in epochs:
                train(args, model, device, train_loader, optimizer, epoch)
                scheduler.step()    # learning rate scheduler

            torch.save(model.state_dict(), "mnist_model_net_{}.pt".format(str(prop)))

            train_error = test(model, device, train_loader)
            test_error = test(model, device, test_loader)

            train_errors.append(train_error)
            test_errors.append(test_error)

        proportions = len(train_dataset) * np.array(proportions)
        plt.loglog(proportions, train_errors, 'g', label='Training loss')
        plt.loglog(proportions, test_errors, 'b', label='Testing loss')
        plt.title('Training and Testing loss')
        plt.xlabel('proportions')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("loss_plot_proportions.png")

        return

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test(model, device, test_loader)

        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ToTensor(),           # Add data augmentation here
                    transforms.RandomApply([
                        transforms.RandomRotation(30),
                        transforms.GaussianBlur(3, sigma=(0.1,0.5))
                    ], p=0.3),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    validation_dataset = datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]))

    # You can assign indices for training/validation or use a random subset for
    # training by using SubsetRandomSampler. Right now the train and validation
    # sets are built from the same indices - this is bad! Change it so that
    # the training and validation sets are disjoint and have the correct relative sizes.
    subset_indices_train = []
    subset_indices_valid = []

    # Get dict of class -> indices
    class_to_indices = {}
    for index, data in enumerate(train_dataset):
        class_name = data[1]
        if not class_name in class_to_indices:
            class_to_indices[class_name] = []

        class_to_indices[class_name].append(index)

    # Randomly sample 15% from each
    for indices in class_to_indices.values():
        random.Random(args.seed).shuffle(indices)
        num_samples = int(len(indices) * 0.15)
        subset_indices_valid += indices[:num_samples]
        subset_indices_train += indices[num_samples:]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )

    val_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    model = Net().to(device)
    #model.load_state_dict(torch.load(args.load_model))

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    training_loss = []
    validation_loss = []
    epochs = range(1, args.epochs + 1)

    # Training loop
    for epoch in epochs:
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()    # learning rate scheduler
        # Report validation accuracy
        val_loss = test(model, device, val_loader)

        # Report training accuracy
        train_loss = test(model, device, train_loader)

        training_loss.append(train_loss)
        validation_loss.append(val_loss)

        # You may optionally save your model at each epoch here
        torch.save(model.state_dict(), "mnist_model_net.pt")

    plt.plot(epochs, training_loss, 'g', label='Training loss')
    plt.plot(epochs, validation_loss, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss_plot.png")


if __name__ == '__main__':
    main()
