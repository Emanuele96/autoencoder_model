import argparse
import json
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.datasets as datasets
import math
import matplotlib.pyplot as plt
import model
import numpy as np

# Tools 

def showTensor(aTensor):
    plt.figure()
    plt.imshow(aTensor.detach().numpy())
    plt.colorbar()
    plt.show()
    
def read_config_from_json(filename):
    with open('configs/' + filename, 'r') as fp:
        cfg = json.load(fp)
    return cfg

def dump_config_to_json(filename):
    with open('configs/' + filename, 'w') as fp:
        json.dump(cfg, fp)

def load_dataset(cfg):
    if str.lower(cfg["dataset"]) == "mnist":
        trainset = datasets.MNIST(root='./data', train=True, download=True,  transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]))
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]))
        outputs_label = 10
    elif str.lower(cfg["dataset"]) == "fmnist":
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True,  transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]))
        testset = datasets.FashionMNIST(root='./data', train=False, download=True,  transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]))
        outputs_label = 10
    elif str.lower(cfg["dataset"]) == "emnist":
        trainset = datasets.EMNIST(root='./data', train=True, split = 'letters' , download=True,  transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]))
        testset = datasets.EMNIST(root='./data', train=False, split = 'letters' , download=True,  transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]))
        outputs_label = 27
    elif str.lower(cfg["dataset"]) == "cifar":
        trainset = datasets.CIFAR10(root='./data', train=True, download=True,  transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]))
        testset = datasets.CIFAR10(root='./data', train=False, download=True,  transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]))
        outputs_label = 10



    D1_length = math.ceil(cfg["D1_D2_split"][0] * len(trainset))
    D2_length = math.floor(cfg["D1_D2_split"][1] * len(trainset))
    D1, D2 = torch.utils.data.random_split(dataset=trainset, lengths= [D1_length, D2_length])

    D1 = DataLoader(dataset=D1, batch_size= cfg["minibatch_size"], shuffle=True)


    D2_train_length = math.ceil(cfg["D2_split"][0] * len(D2))
    D2_val_length = math.floor(cfg["D2_split"][1] * len(D2))
    D2_train, D2_val = torch.utils.data.random_split(dataset=D2, lengths=[D2_train_length, D2_val_length])

    D2_train = DataLoader(dataset=D2_train, batch_size= cfg["minibatch_size"], shuffle=True)
    D2_val = DataLoader(dataset=D2_val, batch_size= cfg["minibatch_size"], shuffle=True)
    D2_test = DataLoader(dataset=testset, batch_size= cfg["minibatch_size"], shuffle=True)

    print("D1 ", len(D1))
    print("D2 train ", len(D2_train))
    print("D2 val ", len(D2_val))
    print("D2 test ", len(D2_test))

    return D1, D2_train, D2_val, D2_train, testset

if __name__ == "__main__":

    # Parse config file of choice
    parser = argparse.ArgumentParser("Autoencoder project")
    parser.add_argument('--config', default="config.json", type=str, help="Select configuration file to load")
    #parser.add_argument('--train', default=False, type=bool, help="Choose whether to train or not")

    args = parser.parse_args()
    cfg = read_config_from_json(args.config)
    print(cfg)

    #load the correct dataset
    D1, D2_train, D2_val, D2_train, testset = load_dataset(cfg)

    #Get the image shape
    tmp = DataLoader(dataset=testset, batch_size= 1, shuffle=False)
    input_shape = None
    label_shape = None
    #max_label = torch.tensor([0])
    for x_batch, y_batch in tmp:
        #input_shape = x_batch.size()
        input_shape = x_batch.view(1, -1).size()
        label_shape = y_batch.size()
        #max_label = torch.max(max_label, y_batch)
        #print(y_batch)
    #print("max ", max_label)

    print("input shape ", input_shape)
    print("label shape ", label_shape)
    
    '''autoencoder = model.Autoencoder(input_shape, cfg["latent_vector_size"])
    
    for epoch in tqdm(range(cfg["epochs_autoencoder"]))
        #For each batch
        for x_batch, y_batch in D1:
            #Flatten and forward pass
            x_batch = x_batch.view(len(x_batch), 1, -1)
            #print("in ", x_batch.size())
            out = autoencoder(x_batch)
            #print("out ", out.size())
            #for i in range(len(x_batch)):
            #    showTensor(x_batch[i].view(1,28,28).permute(1, 2, 0))
            #    showTensor(out[i].view(1,28,28).permute(1, 2, 0))'''

    autoencoder = model.Model(
        "autoencoder",
        input_shape= input_shape,
        latent_vector_size = cfg["latent_vector_size"],
        use_softmax = False,
        optim_name = cfg["optimizer_autoencoder"],
        loss_name = cfg["loss_autoencoder"],
        lr = cfg["lr_autoencoder"],
        classifier_output=None
     )


    losses = autoencoder.fit(D1, cfg["epochs_autoencoder"])

    #plot losses
    time = np.linspace(0, len(losses), num=len(losses))
    plt.plot(time, losses)
    plt.show()


    for x_batch, y_batch in D1:
                #Flatten and forward pass
                x_batch = x_batch.view(len(x_batch), 1, -1)
                #print("in ", x_batch.size())
                out = autoencoder.forward(x_batch)
                #print("out ", out.size())
                for i in range(len(x_batch)):
                    showTensor(x_batch[i].view(1,28,28).permute(1, 2, 0))
                    showTensor(out[i].view(1,28,28).permute(1, 2, 0))


