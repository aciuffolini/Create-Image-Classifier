import argparse


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type = str, default = 'flowers')
    parser.add_argument('--arch', type=str, default='vgg19', help = 'model architecture vgg19')
    parser.add_argument('--gpu', type = bool, default = True, help = 'True: gpu, False: cpu')
    #parser.add_argument('data_directory', default = 'vgg', help='data directory (required)')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help = 'save trained model')
    parser.add_argument('--lr', type = float, default = 0.0001, help = 'learning rate')
    parser.add_argument('--hidden_units', type=int, default = 512, help = 'number hidden units')
    parser.add_argument('--epochs', type=int, default = 0, help='number of epochs')
                            
    #in_args = parser.parse_args()
    return parser.parse_args()
    
#print(get_input_args())
