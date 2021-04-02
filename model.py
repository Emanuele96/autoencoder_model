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
        super(Autoencoder, self ).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_shape[1], out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=150),
            nn.ReLU(),
            nn.Linear(in_features=150, out_features=20),

        )

        self.latent_vector = nn.Sequential(
            nn.Linear(in_features=20, out_features = latent_vector_size),
            nn.ReLU()
        )

        #decoder
        self.decoder = nn.Sequential(

            nn.Linear(in_features=latent_vector_size, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=150),
            nn.ReLU(),
            nn.Linear(in_features=150, out_features=input_shape[1]),

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
        super(Classifier, self ).__init__()

        # encoder
        self.encoder = nn.Sequential(
        
            nn.Linear(in_features=input_shape[1], out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=20),

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

    def freeze_encoder_lv(self):
        self.encoder.weight.requires_grad = False
        self.encoder.bias.requires_grad = False
        
        self.latent_vector.weight.requires_grad = False
        self.latent_vector.bias.requires_grad = False

    def unfreeze_encoder_lv(self):
        self.encoder.weight.requires_grad = True
        self.encoder.bias.requires_grad = True
        
        self.latent_vector.weight.requires_grad = True
        self.latent_vector.bias.requires_grad = True


class Model():
    
    def __init__(self, model_type, input_shape, latent_vector_size, use_softmax, optim_name, loss_name, lr, classifier_output=None):
        
        if model_type == "autoencoder":
            self.model = Autoencoder(input_shape, latent_vector_size)
        elif model_type == "classifier":
            self.model = Classifier(input_shape, latent_vector_size, use_softmax, classifier_output)

        self.model_type = model_type
        self.optim_name = optim_name
        self.loss_name = loss_name
        
        self.lr = lr
        self.optim = self.initiate_optim(self.model.parameters())
        self.loss_fn = self.initiate_loss()
        
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.model.cuda()
        else:
            self.device = "cpu"
        
        #self.device = "cpu"

    def initiate_loss(self):
        if self.loss_name == "mse":
            return nn.MSELoss(reduction="mean")
        elif self.loss_name == "kld":
            return nn.KLDivLoss(reduction = 'batchmean')
        elif self.loss_name =="nl":
            return nn.NLLLoss()
        elif self.loss_name =="ce":
            return cross_entropy()


    def initiate_optim(self, params):
        if self.optim_name == "sgd":
            return optim.SGD(params, lr=self.lr)
        elif self.optim_name == "adam":
            return optim.Adam(params, lr=self.lr)
        elif self.optim_name == "rms":
            return optim.RMSprop(params, lr=self.lr)

    def freeze_encoder_lv(self):
        #Freeze model weights and biases
        self.model.freeze_encoder_lv()
        #Get all parameters that require gradient calculation
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        #Reload the optimizer
        self.optim = self.initiate_optim(params)
        
    def unfreeze_encoder_lv(self):
        #Unfreeze model weights and biases
        self.model.unfreeze_encoder_lv()
        #Get all parameters that require gradient calculation
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        #Reload the optimizer
        self.optim = self.initiate_optim(params)
        #optimizer.add_param_group({'params': net.fc2.parameters()})

    def fit(self, train_loader, n_epochs):
        losses = list()
        for i in tqdm(range(n_epochs), "Training " + self.model_type):
            for x_batch, y_batch in train_loader:
                #send minibatch to device from cpu
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                #flatten x_batch
                x_batch = x_batch.view(len(x_batch), 1, -1)
                #performe training: Unsupervised for autoencoder and supervised for classifier
                if self.model_type == "autoencoder":
                    loss = self.train_step(x_batch, x_batch)
                if self.model_type == "classifier":
                    loss = self.train_step(x_batch, y_batch)
                losses.append(loss)
        return losses


    def train_step(self, input_data, label):
        self.model.train()
        self.optim.zero_grad()
        prediction = self.model(input_data)
        loss = self.loss_fn(prediction, label)
        loss.backward()
        self.optim.step()
        self.model.train(mode=False)
        return loss

    def forward(self, x_batch):
        return self.model(x_batch)