import argparse
import json
from torch.utils.data import DataLoader
import torch
import torchvision
import torchvision.datasets as datasets
import math
import matplotlib.pyplot as plt
import model as mod
import numpy as np
import pickle
from pathlib import Path


# Tools 
def pickle_file(path, filename, obj):
    path = Path(path)
    filepath = path / filename
    f = open(filepath, 'wb')
    pickle.dump(obj, f, -1)
    f.close()

def unpickle_file(path, filename):
    path = Path(path)
    filepath = path / filename
    if filepath.is_file():
        f = open(filepath, 'rb')
        obj = pickle.load(f)
        f.close()
        return obj
    return None

def showTensor(aTensor, shape):
    aTensor = aTensor.view(shape).permute(1, 2, 0)
    plt.figure()
    plt.imshow(aTensor.cpu().detach().numpy())
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

    #Get the image shape
    tmp = DataLoader(dataset=testset, batch_size= 1, shuffle=False)
    picture_shape = None
    input_shape = None
    label_shape = None

    for x_batch, y_batch in tmp:
        picture_shape = x_batch.size()[1:]
        input_shape = x_batch.view(1, -1).size()
        label_shape = y_batch.size()

    print("picture shape ", picture_shape)
    print("input shape ", input_shape)
    print("label shape ", label_shape)

    return D1, D2_train, D2_val, D2_train, picture_shape, input_shape, label_shape, outputs_label


def initializate_model(cfg, model_name, classifier_output = None, use_softmax = False, suffix = ""):
    model = unpickle_file("models", model_name + "_"+ suffix + "_" + str(cfg["dataset"]) + ".pkl" )
    if model is None:
        model = mod.Model(
            model_type = model_name,
            input_shape= input_shape,
            latent_vector_size = cfg["latent_vector_size"],
            use_softmax = use_softmax,
            optim_name = cfg["optimizer_" + model_name],
            loss_name = cfg["loss_" + model_name],
            lr = cfg["lr_" + model_name],
            classifier_output= classifier_output
        )
        pickle_file("models", model_name + "_" + str(cfg["dataset"]) + ".pkl", model)

    else:
        print(model_name + " loaded successfull. Already trained ", model.get_trained_episode_count(), " epochs.")
    return model

def train_model(cfg, model, dataset, val_dataset, suffix = ""):
    model.fit(dataset, val_dataset, cfg["epochs_" + model.get_model_name()])
    pickle_file("models", model.get_model_name() + "_" + suffix + "_" + str(cfg["dataset"]) + ".pkl", model)

if __name__ == "__main__":

    # Parse config file of choice
    parser = argparse.ArgumentParser("Autoencoder project")
    parser.add_argument('--config', default="config.json", type=str, help="Select configuration file to load")
    parser.add_argument('--train', default=False, type=bool, help="Choose whether to train or not")

    args = parser.parse_args()
    cfg = read_config_from_json(args.config)
    print(cfg)

    #load the correct dataset
    D1, D2_train, D2_val, D2_train, picture_shape, input_shape, label_shape, outputs_label = load_dataset(cfg)

    '''autoencoder = initializate_model(cfg, "autoencoder")
    if args.train:
        train_model(cfg, autoencoder, D1)
    losses = autoencoder.get_losses()'''
    autoencoder = initializate_model(cfg, "autoencoder")
    classifier = initializate_model(cfg, "classifier", classifier_output=outputs_label, use_softmax= False)
    classifier.import_weights(autoencoder)
    if args.train:
        train_model(cfg, autoencoder, D1, None)
        train_model(cfg, classifier, D2_train, D2_val)

    #plot losses
    losses = classifier.get_losses()
    time = np.linspace(0, len(losses), num=len(losses))
    plt.plot(time, losses)
    plt.show()

    #plot val
    val = classifier.get_val_results()
    time = np.linspace(0, len(val), num=len(val))
    plt.plot(time, val)
    plt.show()

    #show inputs and recustructions
    for x_batch, y_batch in D1:
        x_batch = x_batch.to("cuda:0")
        #Flatten and forward pass
        x_batch = x_batch.view(len(x_batch), 1, -1)
        #print("in ", x_batch.size())
        out = autoencoder.forward(x_batch)
        out = out.to("cuda:0")
        #print("out ", out.size())
        for i in range(len(x_batch)):
            showTensor(x_batch[i], picture_shape)
            showTensor(out[i], picture_shape)


