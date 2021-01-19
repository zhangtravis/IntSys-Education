import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.parameter import Parameter
from data_loader import get_data_loaders


class LinearRegressionModel(nn.Module):
    """LinearRegressionModel is the linear regression regressor.

    This class handles only the standard linear regression task.

    :param num_param: The number of parameters that need to be initialized.
    :type num_param: int
    """

    def __init__(self, input_size, loss_func):
        ## TODO 1: Set up network
        super().__init__()
        self.loss_f = loss_func
        self.fc1 = nn.Linear(input_size, 1)
              

    def forward(self, x):
        """forward generates the predictions for the input
        
        This function does not have to be called explicitly. We can do the
        following 
        
        .. highlight:: python
        .. code-block:: python

            model = LinearRegressionModel(1, mse_loss)
            predictions = model(X)
    
        :param x: Input array of shape (n_samples, n_features) which we want to
            evaluate on
        :type x: typing.Union[np.ndarray, torch.Tensor]
        :return: The predictions on x
        :rtype: torch.Tensor
        """
        
        #y_pred = torch.matmul(x, self.weight) + self.bias.T
              
        return self.fc1(x)


def data_transform(sample):
    ## TODO: Define a transform on a given (x, y) sample. This can be used, for example
    ## for changing the feature representation of your data so that Linear regression works
    ## better.
    
    #Append x^3 to the beginning of input ?
    square = [sample[0]*sample[0]*sample[0]]
    sample2 = np.append(square,sample)
    return sample2  ## You might want to change this


def mse_loss(output, target):
    """Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`output` and target :math:`target`.
    
    The loss can be described as:

    .. math::
        \\ell(x, y) = L = \\operatorname{mean}(\\{l_1,\\dots,l_N\\}^\\top), \\quad
        l_n = \\left( x_n - y_n \\right)^2,

    where :math:`N` is the batch size. 

    :math:`output` and :math:`target` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.
    
    :param output: The output of the model or our predictions
    :type output: torch.Tensor
    :param target: The expected output or our labels
    :type target: typing.Union[torch.Tensor]
    :return: torch.Tensor
    :rtype: torch.Tensor
    """
    ## TODO 3: Implement Mean-Squared Error loss. 
    # Use PyTorch operations to return a PyTorch tensor
    return nn.functional.mse_loss(output, target) 
    
    #diff = output - target
    
    #return torch.sum(diff * diff) / diff.numel()


def mae_loss(output, target):
    """Creates a criterion that measures the mean absolute error (l1 loss)
    between each element in the input :math:`output` and target :math:`target`.
    
    The loss can be described as:

    .. math::
        \\ell(x, y) = L = \\operatorname{mean}(\\{l_1,\\dots,l_N\\}^\\top), \\quad
        l_n = \\left| x_n - y_n \\right|,

    where :math:`N` is the batch size. 

    :math:`output` and :math:`target` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.
    
    :param output: The output of the model or our predictions
    :type output: torch.Tensor
    :param target: The expected output or our labels
    :type target: typing.Union[torch.Tensor]
    :return: torch.Tensor
    :rtype: torch.Tensor
    """
    ## TODO 4: Implement L1 loss. Use PyTorch operations.
    # Use PyTorch operations to return a PyTorch tensor.
    return torch.abs(output - target)/output.numel()


if __name__ == "__main__":
    ## Here you will want to create the relevant dataloaders for the csv files for which 
    ## you think you should use Linear Regression. The syntax for doing this is something like:
    # Eg:
    # train_loader, val_loader, test_loader =\
    #   get_data_loaders(path_to_csv, 
    #                    transform_fn=data_transform  # Can also pass in None here
    #                    train_val_test=[YOUR TRAIN/VAL/TEST SPLIT], 
    #                    batch_size=YOUR BATCH SIZE)

    ## Now you will want to initialise your Linear Regression model, using something like
    # Eg:
    # model = LinearRegressionModel(...)

    ## Then, you will want to define your optimizer (the thing that updates your model weights)
    # Eg:
    # optimizer = optim.[one of PyTorch's optimizers](model.parameters(), lr=0.01)

    ## Now, you can start your training loop:
    # Eg:
    # model.train()
    # for t in range(TOTAL_TIME_STEPS):
    #   for batch_index, (input_t, y) in enumerate(train_loader):
    #     optimizer.zero_grad()
    #
    #     preds = Feed the input to the model
    #
    #     loss = loss_fn(preds, y)  # You might have to change the shape of things here.
    #     
    #     loss.backward() 
    #     optimizer.step()
    #     
    ## Don't worry about loss.backward() for now. Think of it as calculating gradients.

    ## And voila, your model is trained. Now, use something similar to run your model on
    ## the validation and test data loaders:
    # Eg: 
    # model.eval()
    # for batch_index, (input_t, y) in enumerate(val/test_loader):
    #
    #   preds = Feed the input to the model
    #
    #   loss = loss_fn(preds, y)
    #
    ## You don't need to do loss.backward() or optimizer.step() here since you are no
    ## longer training.

    train_loader, val_loader, test_loader = get_data_loaders("data/DS2.csv",
        #transform_fn=data_transform
        )
    
    model = LinearRegressionModel(1, mse_loss)

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for t in range(10):
        
        for i, (input_t, y) in enumerate(train_loader):
            optimizer.zero_grad()
            preds = model(input_t)
            loss = model.loss_f(preds, y)  # You might have to change the shape of things here.
            loss.backward() 
            optimizer.step()

    
    model.eval()
    for batch_index, (input_t, y) in enumerate(test_loader):
    
        preds = model(input_t)
        loss = model.loss_f(preds, y)
        print(loss)
        print(model.parameters())
        
    
    
