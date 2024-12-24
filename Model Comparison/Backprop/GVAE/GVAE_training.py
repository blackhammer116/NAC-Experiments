import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from GVAE_model import VAE  
import sys, getopt as gopt, time
from ngclearn.utils.metric_utils import measure_KLD
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

class NumpyDataset(Dataset):
    def __init__(self, dataX, dataY=None):
        self.dataX = np.load(dataX) 
        self.dataY = np.load(dataY) if dataY is not None else None 

    def __len__(self):
        return len(self.dataX)

    def __getitem__(self, idx):
        data = torch.tensor(self.dataX[idx], dtype=torch.float32)
        label = torch.tensor(self.dataY[idx], dtype=torch.long) if self.dataY is not None else None
        return data, label
    
options, remainder = gopt.getopt(sys.argv[1:], '',
                                 ["dataX=", "dataY=", "devX=", "devY=", "testX", "testY","verbosity="]
                                 )

dataX = "../../../data/mnist/trainX.npy"
dataY = "../../../data/mnist/trainY.npy"
devX =  "../../../data/mnist/validX.npy"
devY =  "../../../data/mnist/validY.npy"
testX = "../../../data/mnist/testX.npy"
testY = "../../../data/mnist/testY.npy"
verbosity = 0

device = "cuda" if torch.cuda.is_available() else "cpu"

for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--dataY"):
        dataY = arg.strip()
    elif opt in ("--devX"):
        devX = arg.strip()
    elif opt in ("--devY"):
        devY = arg.strip()
    elif opt in ("--testX"):
        testX = arg.strip()
    elif opt in ("--testY"):
        testY = arg.strip()
    elif opt in ("--verbosity"):
        verbosity = int(arg.strip())

print("Train-set: X: {} | Y: {}".format(dataX, dataY))
print("  Dev-set: X: {} | Y: {}".format(devX, devY))
print("  Test-set: X: {} | Y: {}".format(testX, testY))

latent_dim = 64
weight_decay = 1e-4 #change this data to the desired value
model = VAE(latent_dim=latent_dim)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=weight_decay)

train_dataset = NumpyDataset(dataX, dataY)
train_loader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)

dev_dataset = NumpyDataset(devX, devY)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=200, shuffle=False)

test_dataset = NumpyDataset(testX, testY)
test_loader = DataLoader(dataset=test_dataset, batch_size = 200, shuffle = False)

def train(model, loader, optimizer, epoch, gradinet_rescaling_factor=1.0, raduis=5.0):
    model.train()
    total_bce = 0.0
    total_correct = 0
    total_samples = 0
    threshold = 0.1
    latent_rep = []
    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = data.to(device)
        data = data.view(data.size(0), -1)  # Flatten the input data to shape (batch_size, input_dim)

        optimizer.zero_grad()

        reconstructed, mu, logvar = model(data)
        reconstructed = reconstructed.view(data.size(0), -1)  # Flatten the output to (batch_size, input_dim)
        
        # Calculating accuracy
        diff = torch.abs(reconstructed - data) 
        correct = torch.sum(diff < threshold, dim=1)  
        total_correct += correct.sum().item()  
        total_samples += data.size(0)

        # Loss for reconstruction
        bce_loss = F.binary_cross_entropy(reconstructed, data)
        log_probs = torch.log(reconstructed + 1e-9)  # Add small value for numerical stability
        bce_loss.backward()

        # Gradient rescaling
        if gradinet_rescaling_factor !=1.0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(gradinet_rescaling_factor)
        
        # Project gradients to a gaussian ball of radius 5
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        clip_coef = raduis / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)

        optimizer.step()

        latent_rep.append((model.reparameterize(mu, logvar)))

        total_bce += bce_loss.item()
        torch.save(model.state_dict(), "trained_model.pth")

    # Fitting it into Gaussian Mixture Model with 75 components
    gmm = GaussianMixture(n_components=75)
    gmm.fit(latent_rep)

    avg_bce = total_bce / len(loader)
    accuracy = total_correct / (total_samples * data.size(1)) * 100
    print(f'Epoch [{epoch}], BCE: {avg_bce:.4f}, Accuracy: {accuracy}%')
    return avg_bce, accuracy



def evaluate(model, loader):
    model.eval()
    total_bce = 0.0
    total_correct = 0
    total_samples = 0
    threshold = 0.1  
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader):
            data = data.to(device)
            data = data.view(data.size(0), -1) 
            reconstructed, mu, logvar = model(data)
            reconstructed = reconstructed.view(reconstructed.size(0), -1) 

            data_np = data.cpu().numpy()
            reconstructed_np = reconstructed.cpu().numpy()
            
            # Calculating BCE
            bce_loss = F.binary_cross_entropy(reconstructed, data)
            total_bce += bce_loss.item()

            # Calculating NLL
            log_probs = torch.log(reconstructed + 1e-9)  # Add small value for numerical stability
            nll_loss = F.nll_loss(log_probs, data.argmax(dim=-1))
            total_nll += nll_loss.item()

            # Calculating accuracy
            diff = torch.abs(reconstructed - data) 
            correct = torch.sum(diff < threshold, dim=1)  
            total_correct += correct.sum().item()  
            total_samples += data.size(0) 

    avg_bce = total_bce / len(loader)
    accuracy = total_correct / (total_samples * data.size(1)) * 100
    
    print(f'BCE: {avg_bce:.4f}, Accuracy: {accuracy:.2f}%,')

    return  avg_bce, accuracy

num_epochs = 50
sim_start_time = time.time()  # Start time profiling

print("--------------- Training ---------------")
for epoch in range(1, num_epochs + 1): 
    train_bce, train_accuracy = train(model, train_loader, optimizer, epoch)
    print(f'Epoch [{epoch}/{num_epochs}]')
    print(f'Train BCE: {train_bce:.4f}, Train Accuracy: {train_accuracy:.2f}%')

# Stop time profiling
sim_time = time.time() - sim_start_time
print(f"Training Time = {sim_time:.4f} seconds")

print("--------------- Evaluating ---------------")
eval_bce, eval_accuracy = evaluate(model, dev_loader)
print(f'Eval BCE: {eval_bce:.4f}, Eval Accuracy: {eval_accuracy:.2f}%')

print("--------------- Testing ---------------")
inference_start_time = time.time()
test_bce, test_accuracy = evaluate(model, test_loader)
inference_time = time.time() - inference_start_time
print(f"Inference Time = {inference_time:.4f} seconds")
print(f'Test BCE: {test_bce:.4f}, Test Accuracy: {test_accuracy:.2f}%')
