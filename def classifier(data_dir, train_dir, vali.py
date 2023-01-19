def classifier(data_dir, train_dir, valid_dir, in_args):
    global save_check 
        
    
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

    
        
    valid_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean =[0.485, 0.456, 0.406], 
                                                                std = [0.229, 0.224, 0.225])])


    image_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)

# Pass transforms in here, then run the next cell to see how the transforms look
    
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    #train_data = datasets.ImageFolder(train_dir, transform = train_transforms,   class_to_idx=image_datasets.class_to_idx)
    #train_data.class_to_idx = image_datasets.class_to_idx
    #save_check = [train_loader]        
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64,shuffle=True)
    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64,shuffle=True)
    
    #images=images.view(images.shape[0], -1)
    
        
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    model = models.vgg19(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ##############################################################
    learn_rate = in_args.lr if in_args.lr is not None else 0.0001
    arch_type = in_args.arch if in_args.arch is not None else 'vgg19'

    #if arch_type == 'resnet18':
        #model = models.densenet121(pretrained=True)
        #model.name = "resnet18"
    #elif arch_type == 'alexnet':
        #model = models.alexnet(pretrained=True)
        #model.name = "alexnet"
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)   

    #hidden_units = in_args.hidden_units if in_args.hidden_units is not None else 512
    ###################################
    arch_type = in_args.arch if in_args.arch is not None else 'vgg19'
    if arch_type == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        model = models.vgg19(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    learn_rate = in_args.lr if in_args.lr is not None else 0.0001
    
    
    
    
    
    
    
    
    
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
    #images, labels = images.to(device), labels.to(device)
    #images, labels = images.cuda(), labels.cuda()
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr = 0.0001)

    #model.to(device)
    
    #n_epochs = 1
    epochs = 2
    
    
    #if (in_args.epochs is None):
        #epochs = 3
    #else:
        #epochs = in_args.epochs
    
    ###############################################
    #patience = 5

# Initialize a counter for the number of epochs without improvement
    #wait = 0

# Initialize the best validation loss
    #best_val_loss = float('inf')
    ##################################################
    steps = 0 #change this 0

    running_loss = 0
    print_every = 5
    #epochs = epochs + 1
    #current_epoch = 1
    for epochs in range(epochs):
        
        for inputs, labels in trainloader:
            images = images.view(images.shape[0], -1)
            steps += 1
        
            inputs, labels = inputs.to(device), labels.to(device)
            #inputs, labels = inputs.to(device), labels.to(device)
            #inputs, labels = inputs.cuda(), labels.cuda()
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            #model.zero_grad()# ultima modif

            running_loss += loss.item()
        
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
            #inputs.requires_grad = False 
                    #labels.requires_grad = False
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                        
                print(f"Epoch {epochs}/{epochs}.. "
                f"Train loss: {running_loss/print_every:.3f}.. "
                f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                f"Valid accuracy: {accuracy/len(validloader):.3f}")

                running_loss = 0
                                  
                
                model.train()
                epochs + 1       
        
                 
                
    #class_to_idx = train_data.class_to_idx           
    
    return model, optimizer, criterion, train_data
            