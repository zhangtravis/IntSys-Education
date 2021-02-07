import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
import numpy as np
from data_loader import get_data_loaders
from typing import List, Union, Tuple


class SimpleNeuralNetModel(nn.Module):
    """SimpleNeuralNetModel [summary]
    
    [extended_summary]
    
    :param layer_sizes: Sizes of the input, hidden, and output layers of the NN
    :type layer_sizes: List[int]
    """
    def __init__(self, layer_sizes: List[int]):
        super(SimpleNeuralNetModel, self).__init__()
        # TODO: Set up Neural Network according the to layer sizes
        # The first number represents the input size and the output would be
        # the last number, with the numbers in between representing the
        # hidden layer sizes
        self.seq = torch.nn.Sequential()
        
        self.seq.add_module("l0", nn.Linear(layer_sizes[0],layer_sizes[1]))
        for i in range(len(layer_sizes)-2):
            self.seq.add_module("relu_"+str(i+1), nn.LeakyReLU(0.1))
            self.seq.add_module("l"+str(i+1), nn.Linear(layer_sizes[i+1],layer_sizes[i+2]))
        


    def forward(self, x):
        """forward generates the prediction for the input x.
        
        :param x: Input array of size (Batch,Input_layer_size)
        :type x: np.ndarray
        :return: The prediction of the model
        :rtype: np.ndarray
        """
        #need to reshape the size of the array
        [width,height] = [x.shape[2], x.shape[3]]
        x = x.reshape(x.shape[0], width*height)
        return self.seq(x)


class SimpleConvNetModel(nn.Module):
    """SimpleConvNetModel [summary]
    
    [extended_summary]
    
    :param img_shape: size of input image as (W, H)
    :type img_shape: Tuple[int, int]
    :param output_shape: output shape of the neural net
    :type output_shape: tuple
    """
    def __init__(self, img_shape: Tuple[int, int], output_shape: tuple):
        super(SimpleConvNetModel, self).__init__()
        # TODO: Set up Conv Net of your choosing. You can / should hardcode
        # the sizes and layers of this Neural Net. The img_size tells you what
        # the input size should be and you have to determine the best way to
        # represent the output_shape (tuple of 2 ints, tuple of 1 int, just an
        # int , etc).

        self.convlayer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(64 * 6 * 6, 600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(600, 120)
        self.fc3 = nn.Linear(120, 10)
    
    def forward(self, x):
        """forward generates the prediction for the input x.
        
        :param x: Input array of size (Batch,Input_layer_size)
        :type x: np.ndarray
        :return: The prediction of the model
        :rtype: np.ndarray
        """
        # convolutional layers with ReLU and pooling

        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = x.view(-1, 64 * 6 * 6)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

def trainAndEvaluate(model, path_to_pkl, path_to_label, epochs):
    """ Trains and evaluates the model
    :param model: model to train on
    :type model: CNN Model or NN Model
    :param path_to_file: Path to data
    :type path_to_file: String
    :return: None"""

    train_loader, val_loader, test_loader = get_data_loaders(path_to_pkl, path_to_label)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    print('Training model...')
    for t in range(epochs):
        for i, (input_t, y) in enumerate(train_loader):
            preds = model(input_t)
            loss = criterion(preds, y.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print('Training Completed')

    model.eval()
    print('Evaluating model...')
    correct = 0
    for batch_index, (input_t, y) in enumerate(test_loader):
        preds = model(input_t)
        for i in range(len(preds)):
            y_hat = np.argmax(preds[i].detach().numpy())      
            if y_hat == y[i]:
                correct += 1

    print("Accuracy={}".format(correct/12000))

if __name__ == "__main__":
    
    CNN_model = SimpleConvNetModel((28,28), 10)
    NN_model = SimpleNeuralNetModel([28*28, 128, 64, 10])
    # Train & evaluate model
    trainAndEvaluate(CNN_model, 'data/processedImages.pkl', 'data/processedLabels.pkl', 10)
    trainAndEvaluate(NN_model, 'data/processedImages.pkl', 'data/processedLabels.pkl', 30)
