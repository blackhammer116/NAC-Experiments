import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(28*28, 512) # in_dim prev value = 28*28
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, latent_dim)
        self.fc5 = nn.Linear(128, latent_dim)
        # Decoder
        self.fc6 = nn.Linear(latent_dim, 64)
        self.fc7 = nn.Linear(64, 128)
        self.fc8 = nn.Linear(128, 512)
        self.fc9 = nn.Linear(512, 28*28) # out_dim prev value = 28*28

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        mu = self.fc4(h3)
        logvar = self.fc5(h3)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = F.relu(self.fc6(z))
        h5 = F.relu(self.fc7(h4))
        h6 = F.relu(self.fc8(h5))
        return torch.sigmoid(self.fc9(h6))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
