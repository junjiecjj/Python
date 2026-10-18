# -*- coding: utf-8 -*-
"""
Implementation of a PAPR-Aware OFDM Scheme using Deep Neural Networks

This script implements a PAPR-aware adaptive modulation scheme for OFDM, 
which optimizes energy efficiency by dynamically adjusting modulation and 
power allocation across subcarriers.

Created on Sun Oct 20 13:32:18 2024
@author: nickm
"""

import torch 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

# Neural Network Definition (DNN for Modulation and Power Allocation)
class net(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Initialize the neural network architecture.
        Args:
            input_size: Number of input features (K_d subcarriers)
            output_size: Total output size (modulation power allocations)
        """
        super(net, self).__init__()
        self.l1 = nn.Linear(input_size, 100)  # Hidden layer 1
        self.tanh = nn.Tanh()  # Activation function
        self.l4 = nn.Linear(100, 100)  # Hidden layer 2
        self.l2 = nn.Linear(100, output_size)  # Output layer
        self.l3 = nn.Softmax(dim=1)  # Softmax for probability output

    def forward(self, x): 
        """
        Forward pass of the neural network.
        Args:
            x: Input tensor
        Returns:
            Output tensor after modulation and power allocation
        """
        output = self.l1(x) 
        output = self.tanh(output) 
        output = self.l4(output)
        output = self.tanh(output) 
        output = self.l2(output)

        # Apply softmax to the first K_d values (power allocation)
        y1 = self.l3(output[:, 0:K_d])
        
        # Gumbel softmax applied to the modulation part
        for i in range(K_d):        
            y2 = nn.functional.gumbel_softmax(output[:, K_d+i*J:(i+1)*J+K_d], hard=True, dim=-1)
            y1 = torch.cat((y1, y2), -1)  # Concatenate modulation outputs
        return y1

# Custom Loss Function for Energy Efficiency Optimization (Including PAPR)
def loss_function_vol2(y, order, K_d, h_k, N_0, epsilon, Pmax):
    """
    Custom loss function to calculate energy efficiency of the OFDM transmission.
    Args:
        y: Predicted modulation and power allocation
        order: Modulation orders (e.g., 4-QAM, 16-QAM, etc.)
        K_d: Number of subcarriers
        h_k: Channel state information (CSI)
        N_0: Noise power
        epsilon: Efficiency parameter of the Power Amplifier (PA)
        Pmax: Maximum power constraint
    """
    term1 = torch.FloatTensor([0])

    # Calculate throughput based on the modulation order and symbol error rate (SER)
    for i in range(K_d):
        gamma = Pmax * y[0, i] * h_k[0, i] / N_0
        ser = 1 - (1 - 2 * (torch.sqrt(torch.sum(y[:, K_d+i*J:(i+1)*J+K_d] * order)) - 1) / 
                   torch.sqrt(torch.sum(y[:, K_d+i*J:(i+1)*J+K_d] * order))) * \
                  0.5 * torch.erfc(1 / 1.4142 * torch.sqrt(3 * gamma / (torch.sum(y[:, K_d+i*J:(i+1)*J+K_d] * order) - 1))) ** 2
        term1 = term1 + torch.log(torch.log2(torch.sum(y[:, K_d+i*J:(i+1)*J+K_d] * order)) * (1 - ser))
   
    # Monte Carlo procedure to compute average PAPR
    av_papr = 0
    ap_av = 0
    l = 3  # Zero padding factor for accuracy in DFT
    counter = 1 * 10**2
    for j in range(counter):
        symbol = torch.zeros((1, K_d), dtype=torch.complex32)
        for i in range(K_d):
            D = 2**(0.5 * torch.log2(torch.sum(y[:, K_d+i*J:(i+1)*J+K_d] * order))).detach().numpy()
            Delta = torch.sqrt(2 / 3 * (torch.sum(y[:, K_d+i*J:(i+1)*J+K_d] * order) - 1)).detach().numpy()
            psi = torch.randint(1, 2*int(D)+1, (1,))
            z = torch.randint(1, 2*int(D)+1, (1,))
            imag = (2*z - 2*int(D) - 1) / Delta
            real = (2*psi - 2*int(D) - 1) / Delta
            symbol[0, i] = complex(real, imag)

        symbol = symbol * torch.sqrt(y[:, 0:K_d])
        symbol = torch.cat((symbol[:, 0:int(K_d/2)], torch.zeros(1, l*K_d), symbol[:, int(K_d/2):K_d]), dim=-1)
        ofdma_symbol = torch.fft.ifft(symbol)  
        vector = torch.abs(ofdma_symbol) ** 2
        papr = torch.max(vector) / torch.mean(vector)
        ap_av += 0.5 / papr
        av_papr += papr

    av_papr /= counter
    ap_aver = ap_av / counter
    consumption = torch.sum(y[0, 0:K_d]) ** epsilon * Pmax ** (1 - epsilon) / ap_aver
    
    # Average energy efficiency
    x = -term1 / (1 * consumption)
    return x 

# Prepare DataLoader for the dataset
def prepare_dataloader(size_tot, K_d, batch_size, device):
    """
    Prepares the DataLoader for the randomly generated dataset.
    Args:
        size_tot: Total size of the dataset
        K_d: Number of subcarriers
        batch_size: Mini-batch size
        device: CPU or GPU
    """
    htot = np.random.exponential(scale=1, size=(size_tot, K_d))
    h = torch.from_numpy(htot).to(torch.float32).to(device)

    # Create TensorDataset and DataLoader
    dataset = TensorDataset(h)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

######################### MAIN CODE BEGINS #########################

# OFDM system parameters
K_d = 64  # Number of subcarriers
order = torch.tensor([[4, 64, 128, 256, 1024]])  # Modulation orders
J = order.size(dim=1)

# Noise power and other constants
Pmax = 10
B = 5 * 10**6
N_0 = 10**(-135/10) / (K_d * B * Pmax)
epsilon = 0.5

# Hyperparameters
batch_size = 64
epochs = 151
learning_rate = 0.005
size_tot = 5000  # Total number of samples

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare DataLoader
train_loader = prepare_dataloader(size_tot, K_d, batch_size, device)

# Initialize Model and Optimizer
model = net(K_d, (1 + J) * K_d).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Variable to track loss over epochs
var = torch.zeros(1, epochs)

# Training Loop
for epoch in range(epochs):
    for batch_idx, (h_batch,) in enumerate(train_loader):
        h_batch = h_batch.to(device)

        # Forward pass
        y_pred = model(h_batch)
        
        # Split output into power and modulation
        pk = y_pred[:, :K_d]
        arr = y_pred[:, K_d:].reshape(J, K_d)

        # Initialize or accumulate output tensors
        if epoch == 0 and batch_idx == 0:
            ptot = pk
            at = arr
        else:
            ptot = torch.cat((ptot, pk), 0)
            at = torch.cat((at, arr), 0)
        
        # Compute loss
        cost = loss_function_vol2(y_pred, order, K_d, h_batch, N_0, epsilon, Pmax)
        var[:, epoch] += cost.item()

        # Backpropagation and optimizer step
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Print progress every 5 epochs
        if epoch % 5 == 0 and batch_idx == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {cost.item()}")

# Convert final output tensors to NumPy
ptot = ptot.detach().numpy()
at = at.detach().numpy()
var = var.detach().numpy()
