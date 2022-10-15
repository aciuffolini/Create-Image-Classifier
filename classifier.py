import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__
import torch
import numpy as np
import json
from torchvision import datasets, transforms,models
import matplotlib.pyplot as plt
#import helper
from PIL import Image
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import time
import random as random
import os

resnet = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg19 = models.vgg19(pretrained=True)

#models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg19': vgg19}


#def classifier(model_name):
    
   
    # apply model to input
    #model = models[model_name]

    
    # instead of (default)training mode
    #model = model.train()
    
    #return model

def classifier(train_dir, valid_dir,in_args):
    
    data_dir = 'flowers'
    #train_dir = data_dir + '/train'
    #valid_dir = data_dir + '/valid'
        
    
# TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])

# TODO: Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)

    images,labels = next(iter(dataloader))

    #def process_data(train_dir, test_dir, valid_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean =[0.485, 0.456, 0.406], 
                                                                std = [0.229, 0.224, 0.225])])
                                           #transforms.ToPILImage()]) 

    
        
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean =[0.485, 0.456, 0.406], 
                                                                std = [0.229, 0.224, 0.225])])


#image_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)

# Pass transforms in here, then run the next cell to see how the transforms look
    
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64,shuffle=True)
    
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    images=images.view(images.shape[0], -1)
    
        
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    model = models.vgg19(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(nn.Linear(25088, 512),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(512, 102),
                               nn.LogSoftmax(dim=1))
                        
    
    model.classifier = classifier
    
    model.to(device);
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.0001)


    epochs = 0
    
    
    if (in_args.epochs is None):
        epochs = 0
    else:
        epochs = in_args.epochs
    
    ###############################################
    if (in_args.lr is None):
        learn_rate = 0.0001
    else:
        learn_rate = in_args.lr
    
    if (in_args.arch is None):
        arch_type = 'vgg19'       
    else:
        arch_type = in_args.arch
    if (arch_type == 'resnet18'):
        model = models.densenet121(pretrained=True)
        model.name="resnet18"
        #input_node=1024
        #output_node=102
    elif (arch_type == 'alexnet'):
        model = models.alexnet(pretrained=True)
        model.name="alexnet"
        
    if (in_args.hidden_units is None):
        hidden_units = 512
    else:
        hidden_units = in_args.hidden_units
    
    ##################################################
    steps = 0

    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
        
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                    
                        inputs, labels = inputs.to(device), labels.to(device)
                    
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
            
        
                running_loss = 0
                model.train()
                
    return model

    
