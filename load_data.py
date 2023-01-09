def load_data(data_dir, test_dir, valid_dir, in_args):
    data_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])


    # TODO: Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(data_dir, transform=data_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)

    images,labels = next(iter(dataloader))
    # define the transforms for the training, test, and validation datasets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean =[0.485, 0.456, 0.406],
                                                                std = [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean =[0.485, 0.456, 0.406],
                                                               std = [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean =[0.485, 0.456, 0.406],
                                                                std = [0.229, 0.224, 0.225])])

    # define the datasets
    train_data = datasets.ImageFolder(data_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)



    return train_data, test_data, valid_data, images ,labels
