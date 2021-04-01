import torch 
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm

def calculate_conv2d_output(input_size, stride, padding, dilatation, kernel_size):
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
    #format (height, width)
    height_output = math.floor((input_size[0] + 2 * padding[0] - dilatation[0] * (kernel_size[0] - 1)/ stride[0]) + 1) 
    width_output = math.floor((input_size[1] + 2 * padding[1] - dilatation[1] * (kernel_size[1] - 1)/ stride[1]) + 1) 
    return height_output, width_output

class Autoencoder(nn.Module):

    def __init__(self, input_shape, latent_vector_size):
        super(Net, self ).__init__()

        # encoder
        self.encoder = nn.Sequential(
            '''nn.Conv2d(in_channels=input_shape[0], out_channels=4, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=5, stride=1, padding=1),
            nn.ReLU,
            nn.MaxPool2d(kernel_size= 2, stride=2)'''
            
            nn.Linear(in_features=input_shape[1], out_features=100)),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50)),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=20)),

        )

        self.latent_vector = nn.Sequential(
            nn.Linear(in_features=20, out_features = latent_vector_size),
            nn.ReLU()
        )

        #decoder
        self.decoder = nn.Sequential(
            '''nn.ConvTranspose2d(in_channels= 1, out_channels = 4, kernel_size = 5, stride= 1, padding= 1),
            nn.ReLU,
            nn.ConvTranspose2d(in_channels= 4, out_channels = 1, kernel_size = 5, stride= 1, padding= 1),
            nn.ReLU()'''
            nn.Linear(in_features=latent_vector_size, out_features=50)),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=100)),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=input_shape[1])),

        )


    def forward(self, x):
        #forward through the encoder first
        x = self.encoder(x)
        #reshape and pass through the latent vector
        #x = x.view(x.size(0), -1)
        x = self.latent_vector(x)
        #reshape and pass throught the decoder
        #x = x.view(1, 5, 5)
        x = self.decoder(x)
        return x

class Classifier(nn.Module):

    def __init__(self, input_shape, latent_vector_size, use_softmax, classifier_output):
        super(Net, self ).__init__()

        # encoder
        self.encoder = nn.Sequential(
        
            nn.Linear(in_features=input_shape[1], out_features=100)),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50)),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=20)),

        )

        self.latent_vector = nn.Sequential(
            nn.Linear(in_features=20, out_features = latent_vector_size),
            nn.ReLU()
        )

        #classifier

        self.classifier = nn.Sequential( 

            nn.Linear(in_features=latent_vector_size, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=classifier_output),
        )

        self.use_softmax = use_softmax
        if use_softmax:
            self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        #forward through the encoder 
        x = self.encoder(x)
        #pass through the latent vector
        x = self.latent_vector(x)
        #pass throught the classifier
        x = self.classifier(x)
        if self.use_softmax:
            x = self.softmax(x)
        return x


class Model():
    
    def __init__(self, model_type, input_shape, latent_vector_size, use_softmax, optim_name, loss_name, classifier_output=None):
        
        if model_type == "autoencoder":
            self.model = Autoencoder(input_shape, latent_vector_size)
        elif model_type == "classifier":
            self.model = Classifier(input_shape, latent_vector_size, use_softmax, classifier_output)

        self.model_type = model_type
        self.optim_name = optim_name
        self.loss_name = loss_name
        
        self.optim = self.initiate_optim()
        self.loss_fn = self.initiate_loss()

        self.device = "cpu"

    def initiate_loss(self):
        if self.loss_name == "mse":
            return nn.MSELoss(reduction="mean")
        elif self.loss_name == "kld":
            return nn.KLDivLoss(reduction = 'batchmean')
        elif self.loss_name =="nl":
            return nn.NLLLoss()
        elif self.loss_name =="ce":
            return cross_entropy()


    def initiate_optim(self):
        if optim_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=self.lr)
        elif optim_name == "adam":
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif optim_name == "rms":
            return optim.RMSprop(self.model.parameters(), lr=self.lr)

    def fit(self, train_loader, n_epochs):
        losses = list()
        for i in tqdm(range(n_epochs), "Training " + self.model_type):
            for x_batch, y_batch in train_loader:
                #send minibatch to device from cpu
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                #performe training: Unsupervised for autoencoder and supervised for classifier
                if self.model_type == "autoencoder":
                    loss = self.train_step(x_batch, x_batch)
                if self.model_type == "classifier":
                    loss = self.train_step(x_batch, y_batch)
                losses.append(losses)
        return losses


    def train_step(self, input_data, label):
        self.model.train()
        self.optimizer.zero_grad()
        prediction = self.model(input_data)
        loss = self.loss_fn(prediction, label)
        print("loss ", loss)
        loss.backward()
        self.optimizer.step()
        self.model.train(mode=False)
        return loss

