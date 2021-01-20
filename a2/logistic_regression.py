import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_loader import get_data_loaders
from plotting import plot_binary_logistic_boundary
import pandas as pd

epochs = 100

class LogisticRegressionModel(nn.Module):
    """LogisticRegressionModel is the logistic regression classifier.

    This class handles only the binary classification task.

    :param num_param: The number of parameters that need to be initialized.
    :type num_param: int
    """
    def __init__(self, num_param, loss_fn):
        ## TODO 1: Set up network
        super().__init__()
        self.model = nn.Sequential(nn.Linear(num_param, 1), nn.Sigmoid())
        self.loss_fn = loss_fn

    def forward(self, x):
        """forward generates the predictions for the input
        
        This function does not have to be called explicitly. We can do the
        following 
        
        .. highlight:: python
        .. code-block:: python

            model = LogisticRegressionModel(1, logistic_loss)
            predictions = model(X)
    
        :param x: Input array of shape (n_samples, n_features) which we want to
            evaluate on
        :type x: typing.Union[torch.Tensor]
        :return: The predictions on x
        :rtype: torch.Tensor
        """

        ## TODO 2: Implement the logistic regression on sample x
        pred = self.model(x)
        return pred


class MultinomialRegressionModel(nn.Module):
    """MultinomialRegressionModel is logistic regression for multiclass prob.

    This model operates under a one-vs-rest (OvR) scheme for its predictions.

    :param num_param: The number of parameters that need to be initialized.
    :type num_param: int
    :param num_classes: Number of classes to predict
    :type num_classes: int
    :param loss_fn: The loss function that is used to calculate "cost"
    :type loss_fn: typing.Callable[[torch.Tensor, torch.Tensor],torch.Tensor]

    .. seealso:: :class:`LogisticRegressionModel`
    """
    def __init__(self, num_param, num_classes, loss_fn):
        super().__init__()
        ## TODO 3: Set up network
        # NOTE: THIS IS A BONUS AND IS NOT EXPECTED FOR YOU TO BE ABLE TO DO
        self.model = nn.Sequential(nn.Linear(num_param, num_classes), nn.Softmax(dim=1))
        self.loss_fn = loss_fn

    def forward(self, x):
        """forward generates the predictions for the input
        
        This function does not have to be called explicitly. We can do the
        following 
        
        .. highlight:: python
        .. code-block:: python

            model = MultinomialRegressionModel(1, cross_entropy_loss)
            predictions = model(X)
    
        :param x: Input array of shape (n_samples, n_features) which we want to
            evaluate on
        :type x: typing.Union[np.ndarray, torch.Tensor]
        :return: The predictions on x
        :rtype: torch.Tensor
        """
        ## TODO 4: Implement the logistic regression on sample x
        # NOTE: THIS IS A BONUS AND IS NOT EXPECTED FOR YOU TO BE ABLE TO DO
        pred = self.model(x)

        return pred


def logistic_loss(output, target):
    """Creates a criterion that measures the Binary Cross Entropy
    between the target and the output:

    The loss can be described as:

    .. math::
        \\ell(x, y) = L = \\operatorname{mean}(\\{l_1,\dots,l_N\\}^\\top), \\quad
        l_n = -y_n \\cdot \\log x_n - (1 - y_n) \\cdot \\log (1 - x_n),

    where :math:`N` is the batch size.

    Note that the targets :math:`target` should be numbers between 0 and 1.

    :param output: The output of the model or our predictions
    :type output: torch.Tensor
    :param target: The expected output or our labels
    :type target: typing.Union[torch.Tensor]
    :return: torch.Tensor
    :rtype: torch.Tensor
    """
    # TODO 2: Implement the logistic loss function from the slides using
    # pytorch operations

    return -torch.sum(target * torch.log(output) + (1 - target)*torch.log(1-output))/output.numel()


def cross_entropy_loss(output, target):
    """Creates a criterion that measures the Cross Entropy
    between the target and the output:
    
    It is useful when training a classification problem with `C` classes.

    :param output: The output of the model or our predictions
    :type output: torch.Tensor
    :param target: The expected output or our labels
    :type target: typing.Union[torch.Tensor]
    :return: torch.Tensor
    :rtype: torch.Tensor
    """
    # NOTE: THIS IS A BONUS AND IS NOT EXPECTED FOR YOU TO BE ABLE TO DO
    classes = int(torch.max(target).detach().item()) + 1

    # One hot encode target values
    one_hot_target = to_one_hot_encode(target, classes)

    return -torch.sum(one_hot_target*torch.log(output))

def to_one_hot_encode(target, classes):
    """Convert target values to one-hot encoded vector
    :param target: The expected output or our labels
    :type target: typing.Union[torch.Tensor]
    :param classes: Number of classes
    :type classes: int
    :return: one hot encoded vectors of target
    :rtype: torch.Tensor"""

    one_hot = torch.zeros(target.shape[0], classes)
    target = target.long()
    for i in range(one_hot.shape[0]):
        one_hot[i, target[i]] = 1
    return one_hot

def trainAndEvaluate(model, path_to_file):
    """ Trains and evaluates the model
    :param model: model to train on
    :type model: LogisticRegressionModel or MultinomialRegressionModel
    :param path_to_file: Path to data
    :type path_to_file: String
    :return: None"""

    train_loader, val_loader, test_loader = get_data_loaders(path_to_file)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    print('Training model...')
    for t in range(epochs):

        for i, (input_t, y) in enumerate(train_loader):
            preds = model(input_t)
            loss = model.loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print('Training Completed')

    model.eval()
    print('Evaluating model...')
    for batch_index, (input_t, y) in enumerate(test_loader):
        preds = model(input_t)
        loss = model.loss_fn(preds, y)
        print(f"Loss: {loss.detach()}")

if __name__ == "__main__":
    # TODO: Run a sample here
    # Look at linear_regression.py for a hint on how you should do this!!

    # Normal Logistic Regression
    logreg = LogisticRegressionModel(2, logistic_loss)

    # Train & evaluate model
    trainAndEvaluate(logreg, 'data/DS3.csv')

    # Plot data3 using trained model
    data3 = np.array(pd.read_csv('data/DS3.csv'))
    x, y = data3[:, :-1], data3[:, -1:]
    plot_binary_logistic_boundary(logreg, x, y, (-5, 5), (-5, 5))

    # Multinomial Regression
    multiReg = MultinomialRegressionModel(2, 3, cross_entropy_loss)

    # Train & Evaluate model
    trainAndEvaluate(multiReg, 'data/DS4.csv')