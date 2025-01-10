import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
# from GVAE_model import VAE
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

sys.argv = sys.argv[0]
options, remainder = gopt.getopt(sys.argv[1:], '',
                                 ["dataX=", "dataY=", "devX=", "devY=", "testX", "testY","verbosity="]
                                 )

# dataX = "../../../data/mnist/trainX.npy"
# dataY = "../../../data/mnist/trainY.npy"
# devX =  "../../../data/mnist/validX.npy"
# devY =  "../../../data/mnist/validY.npy"
# testX = "../../../data/mnist/testX.npy"
# testY = "../../../data/mnist/testY.npy"

device = "cuda" if torch.cuda.is_available() else "cuda"

dataX = "/content/mnist/trainX.npy"
dataY = "/content/mnist/trainY.npy"
devX  = "/content/mnist/validX.npy"
devY  = "/content/mnist/validY.npy"
testX = "/content/mnist/testX.npy"
testY = "/content/mnist/testY.npy"

verbosity = 0

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

print("> Train-set: X: {} | Y: {}".format(dataX, dataY))
print("> Dev-set: X: {} | Y: {}".format(devX, devY))
print("> Test-set: X: {} | Y: {}".format(testX, testY))

latent_dim = 20
weight_decay = 1e-4 # 1e-4 #change this data to the desired value
model = VAE(latent_dim=latent_dim)
model.to(device)
optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=2e-2, weight_decay=weight_decay)

train_dataset = NumpyDataset(dataX, dataY)
train_loader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True)

dev_dataset = NumpyDataset(devX, devY)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=200, shuffle=False)

test_dataset = NumpyDataset(testX, testY)
test_loader = DataLoader(dataset=test_dataset, batch_size=200, shuffle = False)


def train_reconstruction(model, loader, optimizer, epoch, device, gradinet_rescaling_factor=1.0, raduis=5.0):
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

        # training on the reconstruction
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

        latent_rep.append((model.reparameterize(mu, logvar)).detach().cpu().numpy())

        total_bce += bce_loss.item()
        torch.save(model.state_dict(), "trained_model.pth")

    latent_rep = np.concatenate(latent_rep, axis=0)

    gmm, gmm_score = fit_gmm_on_latent(latent_rep)

    mcs_sample = monte_carlo_log_likelihood(gmm)

    avg_bce = total_bce / len(loader)
    accuracy = total_correct / (total_samples * data.size(1)) * 100
    print(f'Epoch [{epoch}], BCE: {avg_bce:.4f}, Accuracy: {accuracy:.4f}%, GMM score: {gmm_score:.4f}, MCS_log_likelihood: {mcs_sample:.4f}')
    # print(f'Epoch [{epoch}], BCE: {avg_bce:.4f}, Accuracy: {accuracy:.4f}%')
    return avg_bce, accuracy, total_bce


def train_pattern_completion(model, loader, optimizer, epoch, device, gradinet_rescaling_factor=1.0, raduis=5.0):
    model.train()
    total_m_mse = 0.0
    total_correct = 0
    total_samples = 0
    threshold = 0.1
    latent_rep = []
    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = data.to(device)
        data = data.view(data.size(0), -1)  # Flatten the input data to shape (batch_size, input_dim)

        optimizer.zero_grad()

        # for pattern completion
        mask = np.random.binomial(1, 0.5, size=data.shape)
        masked_data = data * torch.tensor(mask, device=device) # masking 50% of the data randomly
        reconstructed_masked, mu_masked, logvar_masked = model(masked_data)
        reconstructed_masked = reconstructed_masked.view(data.size(0), -1)

        # Calculating accuracy for pattern completion
        diff = torch.abs(reconstructed_masked - data)
        correct = torch.sum(diff < threshold, dim=1)
        total_correct += correct.sum().item()
        total_samples += data.size(0)

        # Loss for reconstruction
        m_mse_loss = F.mse_loss(reconstructed_masked, masked_data)
        log_probs = torch.log(reconstructed_masked + 1e-9)  # Add small value for numerical stability
        m_mse_loss.backward()

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

        latent_rep.append((model.reparameterize(mu_masked, logvar_masked)).detach().cpu().numpy())

        total_m_mse += m_mse_loss.item()
        torch.save(model.state_dict(), "trained_model.pth")

    latent_rep = np.concatenate(latent_rep, axis=0)

    gmm, gmm_score = fit_gmm_on_latent(latent_rep)

    mcs_sample = monte_carlo_log_likelihood(gmm)

    avg_m_mse = total_m_mse / len(loader)
    accuracy = total_correct / (total_samples * data.size(1)) * 100
    err_prc = 1 - (accuracy / 100)
    print(f'Epoch [{epoch}], M_MSE: {avg_m_mse:.4f}, Accuracy: {accuracy:.4f}%, GMM score: {gmm_score:.4f}, MCS_log_likelihood: {mcs_sample:.4f}')
    # print(f'Epoch [{epoch}], BCE: {avg_bce:.4f}, Accuracy: {accuracy:.4f}%')
    return avg_m_mse, accuracy, total_m_mse, err_prc


def evaluate_reconstruction(model, loader):
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


            # Calculating accuracy
            diff = torch.abs(reconstructed - data)
            correct = torch.sum(diff < threshold, dim=1)
            total_correct += correct.sum().item()
            total_samples += data.size(0)

    avg_bce = total_bce / len(loader)
    accuracy = total_correct / (total_samples * data.size(1)) * 100

    print(f'BSE: {avg_bce:.4f}, Accuracy: {accuracy:.2f}%,')

    return  avg_bce, accuracy, total_bce


def evaluate_pattern_completion(model, loader):
    model.eval()
    total_m_mse = 0.0
    total_correct = 0
    total_samples = 0
    threshold = 0.1
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader):
            data = data.to(device)
            data = data.view(data.size(0), -1)

            mask = np.random.binomial(1, 0.5, size=data.shape)
            masked_data = data * torch.tensor(mask, device=device)

            reconstructed_masked, mu_masked, logvar_masked = model(masked_data)
            reconstructed_masked = reconstructed_masked.view(reconstructed_masked.size(0), -1)

            masked_data_np = masked_data.cpu().numpy()
            reconstructed_np = reconstructed_masked.cpu().numpy()

            # Calculating BCE
            m_mse_loss = F.mse_loss(reconstructed_masked, masked_data)
            total_m_mse += m_mse_loss.item()


            # Calculating accuracy
            diff = torch.abs(reconstructed_masked - masked_data)
            correct = torch.sum(diff < threshold, dim=1)
            total_correct += correct.sum().item()
            total_samples += masked_data.size(0)

    avg_m_mse = total_m_mse / len(loader)
    accuracy = total_correct / (total_samples * masked_data.size(1)) * 100

    print(f'M_MSE: {avg_m_mse:.4f}, Accuracy: {accuracy:.2f}%,')

    return  avg_m_mse, accuracy, total_m_mse


def fit_gmm_on_latent(latent_rep):
  """
  Fits the Gaussian mixture model witht the latent space
  Args:
    latent_rep: the latent representation
  Returns:
    gmm: the gmm model
    gmm_score: the score of the gmm model
  """
  print("Fitting the GMM with latent_rep...")
  gmm = GaussianMixture(n_components=75)
  gmm.fit(latent_rep)
  gmm_score = gmm.score(latent_rep) # average log-liklelihood
  return gmm, gmm_score


def monte_carlo_log_likelihood(gmm, num_samples=5000):
  """
  Estimates a property of the GMM using Monte Carlo sampling.

  Args:
      gmm: The fitted Gaussian Mixture Model.
      num_samples: The number of Monte Carlo samples to draw.
  Returns:
      The estimated property value (expected value).
  """
  sample = gmm.sample(num_samples)[0]
  property_value = gmm.score_samples(sample)
  return np.sum(property_value) / num_samples


num_epochs = 50
seed = 64
torch.random.manual_seed(seed=seed)
np.random.seed(seed)
sim_start_time = time.time()  # Start time profiling


print("--------------- Training Reconstruction ---------------")
for epoch in range(1, num_epochs + 1):
    train_bce, train_accuracy, total_bce = train_reconstruction(model, train_loader, optimizer, epoch, device=device)
    print(f'Epoch [{epoch}/{num_epochs}]')
    print(f'Train BCE: {train_bce:.4f}, Train Accuracy: {train_accuracy:.2f}%, Total BCE: {total_bce:.4f}')

# Stop time profiling
sim_time = time.time() - sim_start_time
print(f"Training Time = {sim_time:.4f} seconds")

print("--------------- Evaluating - Reconstruction ---------------")
eval_bce, eval_accuracy, total_bce = evaluate_reconstruction(model, dev_loader)
print(f'Eval BCE: {eval_bce:.4f}, Eval Accuracy: {eval_accuracy:.2f}%, Total Eval BCE: {total_bce:.4f}')

print("--------------- Testing - Reconstruction ---------------")
inference_start_time = time.time()
test_bce, test_accuracy, total_bce = evaluate_reconstruction(model, test_loader)
inference_time = time.time() - inference_start_time
print(f"Inference Time = {inference_time:.4f} seconds")
print(f'Test BCE: {test_bce:.4f}, Test Accuracy: {test_accuracy:.2f}%, Total Test BCE: {total_bce:.4f}')

print("\n-------------------------------------------------------------")
print("-------------------------------------------------------------\n")

print("--------------- Training - Pattern Completion ---------------")
sim_start_time = time.time()

for epoch in range(1, num_epochs + 1):
    train_m_mse, train_accuracy, total_m_mse, err_prc = train_pattern_completion(model, train_loader, optimizer, epoch, device=device)
    print(f'Epoch [{epoch}/{num_epochs}]')
    print(f'Train M_MSE: {train_m_mse:.4f}, Train Accuracy: {train_accuracy:.2f}%, Total M_MSE: {total_m_mse:.4f}, Err %: {err_prc}')

# Stop time profiling
sim_time = time.time() - sim_start_time
print(f"Training Time = {sim_time:.4f} seconds")

print("--------------- Evaluating - Pattern Completion ---------------")
eval_m_mse, eval_accuracy, total_m_mse = evaluate_pattern_completion(model, dev_loader)
print(f'Eval M_MSE: {eval_m_mse:.4f}, Eval Accuracy: {eval_accuracy:.2f}%, Total Eval M_MSE: {eval_m_mse:.4f}')

print("--------------- Testing - Pattern Completion ---------------")
inference_start_time = time.time()
test_m_mse, test_accuracy, total_test_m_mse = evaluate_pattern_completion(model, test_loader)
inference_time = time.time() - inference_start_time
print(f"Inference Time = {inference_time:.4f} seconds")
print(f'Test M_MSE: {test_m_mse:.4f}, Test Accuracy: {test_accuracy:.2f}%, Total Test M_MSE: {total_test_m_mse:.4f}')
