import matplotlib.pyplot as plt
import numpy as np

# code from https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#

def show_image(img) :
    img = img / 2 + 0.5 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show() 