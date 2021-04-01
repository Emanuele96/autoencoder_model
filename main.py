import argparse
import json
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.datasets as datasets
import math

# Tools 
def read_config_from_json(filename):
    with open('configs/' + filename, 'r') as fp:
        cfg = json.load(fp)
    return cfg

def dump_config_to_json(filename):
    with open('configs/' + filename, 'w') as fp:
        json.dump(cfg, fp)

def load_dataset(cfg):
    if str.lower(cfg["dataset"]) == "mnist":
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    elif str.lower(cfg["dataset"]) == "fmnist":
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=None)
        testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=None)
    elif str.lower(cfg["dataset"]) == "emnist":
        trainset = datasets.EMNIST(root='./data', train=True, split = 'letters' , download=True, transform=None)
        testset = datasets.EMNIST(root='./data', train=False, split = 'letters' , download=True, transform=None)
    elif str.lower(cfg["dataset"]) == "cifar":
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)


    train_loader = DataLoader(dataset=trainset, batch_size= cfg["minibatch_size"], shuffle=True)

    D1_length = math.ceil(cfg["D1_D2_split"][0] * len(train_loader))
    D2_length = math.floor(cfg["D1_D2_split"][1] * len(train_loader))
    D1, D2 = torch.utils.data.random_split(dataset=train_loader, lengths= [D1_length, D2_length])

    D2_train_length = math.ceil(cfg["D2_split"][0] * len(D2))
    D2_val_length = math.floor(cfg["D2_split"][1] * len(D2))
    D2_train, D2_val = torch.utils.data.random_split(dataset=D2, lengths=[D2_train_length, D2_val_length])
    D2_test = DataLoader(dataset=testset, batch_size= cfg["minibatch_size"], shuffle=True)
    print("D1 ", len(D1))
    print("D2 train ", len(D2_train))
    print("D2 val ", len(D2_val))
    print("D2 test ", len(D2_test))

    return D1, D2_train, D2_val, D2_train

if __name__ == "__main__":

    # Parse config file of choice
    parser = argparse.ArgumentParser("Autoencoder project")
    parser.add_argument('--config', default="config.json", type=str, help="Select configuration file to load")
    #parser.add_argument('--train', default=False, type=bool, help="Choose whether to train or not")

    args = parser.parse_args()
    cfg = read_config_from_json(args.config)
    print(cfg)

    #load the correct dataset
    D1, D2_train, D2_val, D2_train = load_dataset(cfg)
