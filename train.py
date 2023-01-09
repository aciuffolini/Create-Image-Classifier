# Imports here
import argparse
import torch
import numpy as np

from torchvision import datasets, transforms,models
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import time
import random as random
import os
import ast
from torch import __version__


# Imports python modules
from time import time

# Imports print functions that check the lab
#from print_functions_for_lab_checks import *

# Imports functions created for this program

from get_inputs_args import get_input_args
from load_data import load_data
from classifier import classifier
from save_check import save_check

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

def main():
    start_time = time()

    global in_args
    from load_data import load_data
    #data_dir = args.data_dir
    get_input_args()

    in_args = get_input_args()


    train_data, test_data, valid_data, images ,labels = load_data(data_dir, train_dir, valid_dir, in_args)


    # Call the classifier function with the loaded data


 # Function that checks command line arguments using in_arg
    #check_command_line_arguments(args)
    classifier(train_data, test_data, valid_data, in_args, images, labels)

    model = classifier(train_data,test_data, valid_data, in_args, images, labels)#, in_args.gpu, in_args.lr)



    save_check(model, in_args,train_data)

    end_time = time()
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )



if __name__ == "__main__":
    main()
