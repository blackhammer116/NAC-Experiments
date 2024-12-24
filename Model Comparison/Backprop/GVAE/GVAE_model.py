import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(1, 512) # in_dim prev value = 28*28
        self.fc2 = nn.Linear(512, 240)
        self.fc3 = nn.Linear(240, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, latent_dim)
        # Decoder
        self.fc6 = nn.Linear(latent_dim, 64)
        self.fc7 = nn.Linear(64, 128)
        self.fc8 = nn.Linear(128, 512)
        self.fc9 = nn.Linear(512, 1) # out_dim prev value = 28*28

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
        h4 = F.relu(self.fc4(h3))
        mu = self.fc5(h4)
        logvar = self.fc5(h4)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = F.relu(self.fc6(z))
        h5 = F.relu(self.fc7(h4))
        h6 = F.relu(self.fc8(h5))
        h7 = F.relu(self.fc9(h6))
        return torch.sigmoid(h7)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 1))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# latent_dim = 20
# model = VAE(latent_dim=latent_dim)
# print("Model Architecture: \n", model)

# print("\nLayer details:")
# for name, layer in model.named_children():
#     print(f"Layer name: {name}, Layer type: {layer}")
#     if isinstance(layer, nn.Linear):
#         print(f"Weight: {layer.weight.shape}")
#         # print(f"Weight-shape: {layer.weight.shape}")
#         # print(f"Bias: {layer.bias}")
#         # print(f"Bias-shape: {layer.bias.shape}")