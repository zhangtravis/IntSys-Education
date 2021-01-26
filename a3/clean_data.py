import pickle
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt


def load_pickle_file(path_to_file):
    """
    Loads the data from a pickle file and returns that object
    """
    ## Look up: https://docs.python.org/3/library/pickle.html
    ## The code should look something like this:
    # with open(path_to_file, 'rb') as f:
    #   obj = pickle....
    ## We will let you figure out which pickle operation to use
    with open(path_to_file, 'rb') as file:
        data = pickle.load(file)

    return data

def processData(images, labels):
    """
    Rotates images and manually assigns missing labels
    :param images: Images to preprocess
    :type images: List(PIL.Image.Image)
    :param labels: Labels corresponding to the images
    :type labels: List(int)
    :return: Preprocessed images
    :rtype: List(tuple[PIL.Image.Image, int])
    """

    # Defined based on manual observations. Key represents rotation value needed and values represent which image needs that specific rotation
    imageRotate = {-90: [0, 35, 38, 39, 43, 44, 47], -45: [6, 8, 9, 10, 15, 16, 17, 20, 21, 23, 28, 32, 33, 36, 40, 49],
                   180: [2, 3, 4, 5, 7, 12, 14, 22, 27, 37, 41, 45, 46, 48, 52, 54, 57],
                   135: [56],
                   90: [1, 11, 13, 18, 19, 24, 25, 26, 29, 30, 31, 34, 42, 50, 51, 53, 55, 58, 59]}

    # TODO: Not sure about image index 4
    # 0: T-shirt/top; 1: Trouser; 2: Pullover; 3: Dress; 4: Coat; 5: Sandal; 6: Shirt; 7: Sneaker; 8: Bag; 9: Ankle boot
    labelAnnotate = {0: [1, 17, 48, 55], 1: [21, 38], 2: [5], 3: [3, 4, 20, 25, 51], 4: [18, 22, 24, 28, 53], 5: [8, 9, 13, 43],
                     6: [29, 37, 39], 7: [6, 14, 41, 46], 8: [23, 35, 57], 9: [11, 15, 36, 42, 44]}

    for i in range(len(images[:60])):
        for j in imageRotate.keys():
            if i in imageRotate[j]:
                images[i] = images[i].rotate(j)
                break

    key = 0
    for values in labelAnnotate.values():
        for value in values:
            labels[value] = key
        key += 1

    return images, labels

def transformImages(images):
    """
    Returns images all with size (28, 28)
    :param images: list of images
    :type images: List(PIL.Image.Image)
    :return: List(PIL.Image.Image)
    """

    
    # TODO: What about resizing images when the original image is tiny (8,8)?
    for i in range(len(images)):
        if images[i].size[0]== 28 and images[i].size[1]!= 28:
            padding = (images[i].size[1] - 28)/2
            images[i] = images[i].crop((0,padding,28,images[i].size[1]-padding))
        elif images[i].size[0]!= 28 and images[i].size[1]== 28:
            padding = (images[i].size[0] - 28)/2
            images[i] = images[i].crop((padding,0,images[i].size[0]-padding,28))
        else:
            images[i] = images[i].resize((28, 28))

    return images



## You should define functions to resize, rotate and crop images
## below. You can perform these operations either on numpy arrays
## or on PIL images (read docs: https://pillow.readthedocs.io/en/stable/reference/Image.html)


## We want you to clean the data, and then create a train and val folder inside
## the data folder (so your data folder in a3/ should look like: )
# data/
#   train/
#   val/

## Inside the train and val folders, you will have to dump the CLEANED images and
## labels. You can dump images/annotations in a pickle file (because our data loader 
## expects the path to a pickle file.)

## Most code written in this file will be DIY. It's important that you get to practice
## cleaning datasets and visualising them, so we purposely won't give you too much starter
## code. It'll be up to you to look up documentation and understand different Python modules.
## That being said, the task shouldn't be too hard, so we won't send you down any rabbit hole.

if __name__ == "__main__":
    ## Running this script should read the input images.pkl and labels.pkl and clean the data
    ## and store cleaned data into the data/train and data/val folders

    ## To correct rotated images and add missing labels, you might want to prompt the terminal
    ## for input, so that you can input the angle and the missing label
    ## Remember, the first 60 images are rotated, and might contain missing labels.

    images = load_pickle_file('data/images.pkl')
    labels = load_pickle_file('data/labels.pkl')

    processedImages, processedLabels = processData(images, labels)
    processedImages = transformImages(processedImages)

    with open('data/processedImages.pkl', 'wb') as f:
        pickle.dump(processedImages, f)

    with open('data/processedLabels.pkl', 'wb') as f:
        pickle.dump(processedLabels, f)

