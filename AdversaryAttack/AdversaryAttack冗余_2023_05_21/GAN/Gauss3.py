#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 22:31:01 2023

@author: jack

https://www.pytorchtutorial.com/50-lines-of-codes-for-gan/

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator


# ### Uncomment only one of these
(name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)
# (name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x * 2)

print("Using data [%s]" % (name))

# ##### DATA: Target data and generator input data

def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian

def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian

# ##### MODELS: Generator model and discriminator model

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.sigmoid(self.map2(x))
        return self.map3(x)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return F.sigmoid(self.map3(x))

def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

def decorate_with_diffs(data, exponent):
    mean = torch.mean(data.data, 1)
    print(f"{data.data} mean = {mean }, data.size() = {data.size()}, exponent = {exponent}")
    # mean = tensor([4.0757]), data.shape = torch.Size([1, 100]), exponent = 2.0
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    return torch.cat([data, diffs], 1)

# Data params
data_mean = 4
data_stddev = 1.25

# Model params
g_input_size = 1     # Random noise dimension coming into generator, per output vector
g_hidden_size = 50   # Generator complexity
g_output_size = 1    # size of generated output vector

d_input_size = 100   # Minibatch size - cardinality of distributions
d_hidden_size = 50   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
minibatch_size = d_input_size

d_learning_rate = 2e-4  # 2e-4
g_learning_rate = 2e-4
optim_betas = (0.9, 0.999)
num_epochs = 30000
print_interval = 200
d_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1

d_sampler = get_distribution_sampler(data_mean, data_stddev)
gi_sampler = get_generator_input_sampler()

G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)                    # 1,   50,  1
D = Discriminator(input_size=d_input_func(d_input_size), hidden_size=d_hidden_size, output_size=d_output_size)  # 100, 50,  1

criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)

for epoch in range(num_epochs):
    for d_index in range(d_steps):
        # 1. Train D on real+fake
        D.zero_grad()

        #  1A: Train D on real
        d_real_data = Variable(d_sampler(d_input_size))
        d_real_decision = D(preprocess(d_real_data))
        D_x = d_real_decision.mean().item()

        d_real_error = criterion(d_real_decision, Variable(torch.ones(1, 1)))  # ones = true
        d_real_error.backward() # compute/store gradients, but don't change params
        # print(f"d_real_data.shape = {d_real_data.shape}, d_real_decision.shape = {d_real_decision.shape}, d_real_error = {d_real_error}")
        # d_real_data.shape = torch.Size([1, 100]), d_real_decision.shape = torch.Size([1, 1]), d_real_error = 0.5455108284950256

        #  1B: Train D on fake
        d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
        d_fake_decision = D(preprocess(d_fake_data.transpose(1,0)))
        D_G_z1 = d_fake_decision.mean().item()
        
        d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(1, 1)))  # zeros = fake
        d_fake_error.backward()
        d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()
        # print(f"d_gen_input.shape = {d_gen_input.shape}, d_fake_data.shape = {d_fake_data.shape}, d_fake_decision.shape = {d_fake_decision.shape}, d_fake_error = {d_fake_error}")
        # d_gen_input.shape = torch.Size([100, 1]), d_fake_data.shape = torch.Size([100, 1]), d_fake_decision.shape = torch.Size([1, 1]), d_fake_error = 0.45902201533317566

    for g_index in range(g_steps):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        G.zero_grad()

        gen_input = Variable(gi_sampler(minibatch_size, g_input_size))
        g_fake_data = G(gen_input)
        dg_fake_decision = D(preprocess(g_fake_data.transpose(1,0)))
        D_G_z2 = dg_fake_decision.mean().item()
        
        g_error = criterion(dg_fake_decision, Variable(torch.ones(1,1)))  # we want to fool, so pretend it's all genuine
        g_error.backward()
        g_optimizer.step()  # Only optimizes G's parameters
        # print(f"gen_input.shape = {gen_input.shape}, g_fake_data.shape = {g_fake_data.shape}, dg_fake_decision.shape = {dg_fake_decision.shape}, g_error = {g_error}")
        # gen_input.shape = torch.Size([100, 1]), g_fake_data.shape = torch.Size([100, 1]), dg_fake_decision.shape = torch.Size([1, 1]), g_error = 1.0169317722320557
    
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    np.set_printoptions(formatter={'double': '{: 0.3f}'.format})
    np.set_printoptions(precision=3, suppress=True)
    if epoch % print_interval == 0:
        print(f"Epoch: {epoch}: D: {extract(d_real_error)[0]:.3f}/{extract(d_fake_error)[0]:.3f} \t G: {extract(g_error)[0]:.4f} (Real: {np.array(stats(extract(d_real_data)))}, Fake: {np.array(stats(extract(d_fake_data)))}), D(x): {D_x:.4f}, D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f} ")


N = 100000
noise = gi_sampler(N, g_input_size)
fakeGauss = G(noise).detach().storage().tolist()
print(f"len(fakeGauss) = {len(fakeGauss)}")

fig, ax1 = plt.subplots()
ax1.hist(fakeGauss, bins=100, )
ax1.set_ylabel("Count", fontsize='12')

plt.savefig("/home/jack/snap/test.eps")

plt.show()


plt.close()






































































































































































































































































