# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 16:10:03 2024

@author: esnow
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.sparse import random as sparse_random
from skimage.transform import resize
import math

# Function to load retinal wave data
def load_retinal_wave_data(wave_directory):
    wave_files = [os.path.join(wave_directory, f) for f in os.listdir(wave_directory) if f.endswith('.npy')]
    waves = [np.load(wave_file) for wave_file in wave_files]
    return waves

# Function to resize data
def resize_data(data, new_size):
    T = data.shape[2]
    resized_data = np.zeros((new_size[0], new_size[1], T))
    for t in range(T):
        resized_frame = resize(data[:, :, t], new_size, anti_aliasing=True, mode='reflect')
        threshold = 0.01
        resized_data[:, :, t] = (resized_frame > threshold).astype(int)
    return resized_data

# Function to normalize data
def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val - min_val == 0:
        return data
    return (data - min_val) / (max_val - min_val)

# Function to initialize variables
def initialize_variables(N, g, p, alpha):
    scale = 1.0 / np.sqrt(p * N)
    sparse_matrix = sparse_random(N, N, density=p, format='csr', data_rvs=np.random.randn)
    J_d = sparse_matrix.toarray() * g * scale
    u_in = np.random.uniform(-1, 1, size=(N, N))
    u = np.random.uniform(-1, 1, size=(N, N))
    P = np.eye(N) / alpha
    x_0 = 0.5 * np.random.randn(N)
    return J_d, u_in, u, P, x_0

# Function to downsample data
def downsample_data(data, original_dt, new_dt):
    factor = int(new_dt / original_dt)
    return data[:, ::factor]

# Load Retinal Wave Data
wave_directory = 'data/generated_waves/Trial0_Directional/numpy_waves/'
waves = load_retinal_wave_data(wave_directory)
retinal_data = waves[1]  # Using the second wave file (index starts from 0)

# Determine Frame Dimensions
frame_height, frame_width, T = retinal_data.shape
print(f"Frame dimensions: {frame_height}x{frame_width}, Number of frames: {T}")

# Load the excitatory data
exc_data = np.load('D:/MA/activity_code/activity_code/data/rates_exc_Htriggered_Hampdirect_trial0.npy')

# Downsample the excitatory data to match the retinal data's temporal resolution
original_dt = 1  # Original temporal resolution of exc_data is 1 ms
new_dt = 10   # New temporal resolution is 10 ms
exc_data_downsampled = downsample_data(exc_data, original_dt, new_dt)

# Resizing the excitatory data frames from 40x40 to 10x10
new_size = (40, 40)
T_resized = retinal_data.shape[2]
resized_exc_data = np.zeros((new_size[0] * new_size[1], T_resized))

for t in range(T_resized):
    frame = exc_data_downsampled[:, t].reshape(40, 40)  # Assuming original size of exc_data frames is 40x40
    resized_frame = resize(frame, new_size, anti_aliasing=True, mode='reflect')
    resized_exc_data[:, t] = resized_frame.reshape(new_size[0] * new_size[1])

print(f"Resized excitatory data dimensions: {resized_exc_data.shape}")

# Resizing the retinal data frames from 64x64 to 10x10
resized_retinal_data = resize_data(retinal_data, new_size)
print(f"Resized retinal data dimensions: {resized_retinal_data.shape}")

# Normalize the data
normalized_exc_data = normalize_data(resized_exc_data)
normalized_retinal_data = normalize_data(resized_retinal_data)

# Vectorize retinal data
vectorized_retinal_data = normalized_retinal_data.reshape(new_size[0] * new_size[1], T_resized)
N, T_resized = vectorized_retinal_data.shape

# Set the output directory
output_dir = os.path.expanduser('D:\MA\FORCE_1_A_in_actionplots')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.0
dt = 1
tau = 1.25
t_vec = np.linspace(0, T, int(T/dt))
# Initialize variables
W = np.zeros((N, N)) # for each element of 1600 we have a destinated w vector
J = g * np.random.randn(N, N) / math.sqrt(N)
w_f = 2.0 * (np.random.rand(N) - 0.5)  # fixed weight after init
x_0 = 0.5 * np.random.randn(N)
z_0 = 0.5 * np.random.randn()
z = np.zeros((N,))  # Initialize z as a vector for 2D output
# Target function: sum of 4 sinusoids

# Generate x positions

# a = exc_data[:N, :int(T/dt)]  # Adjust size to match the RNN training duration
# Training Loop only time loop

# neuron_num = 10
x = np.zeros((N, T_resized))
x[:, 0] = x_0
r = np.tanh(x_0)
ps_vec = []
P = np.eye(N) / alpha  # Initialize learning update matrix P
z = z_0
for t in range(1, len(t_vec)):
    # Update neuron state retinal_data_flat_cycled
    # if t <= retinal_data.shape[2]:
    
    x[:, t] = (1 - dt / tau) * x[:, t-1] + (dt / tau) * (np.dot(J, r) + z * w_f)
    # Update firing rate
    r = np.tanh(x[:, t])  # Ensure r is (N,)
    
    # Update network output
    z = np.dot(W.T , r) 
    
    # Compute error
    e_minus = z - normalized_exc_data[:,t] # 
    
                                              
    # RLS update
    k = np.dot(P, r)
    rPr = np.dot(r.T, k)
    c = 1.0 / (1.0 + rPr)
    W -= c * np.outer(k, e_minus) #key training process #sensitive to errorc * np.outer(k, e)
    P = P - c * np.outer(k, k)
    
    # Post-error calculation
    e_plus = np.dot(W.T, r) - normalized_exc_data[:,t]

# Testing Phase
simtime_len = len(t_vec)
final_zpt = np.zeros((N,simtime_len))
x_test = x[:, -1]  # Start from the last state of the training phase
r_test = np.tanh(x_test)
z = z_0
for ti in range(simtime_len):
    x_test = (1 - dt / tau) * x_test + (dt / tau) * (np.dot(J, r_test) + z * w_f)
    r_test = np.tanh(x_test)
    z = np.dot(W.T, r_test)  # Use the final trained weights
    final_zpt[:, ti] = z
# Calculate Mean Absolute Error (MAE)
  # Assuming ft2 is the same as the target function used for training
error_avg = np.mean(np.abs(final_zpt - normalized_exc_data))
print(f'Testing MAE: {error_avg:.3f}')

for iter in range(simtime_len):
    # Plot the results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    im_1 = plt.imshow(final_zpt[:, iter].reshape(40, 40))
    plt.colorbar(im_1)
    plt.title(f'Output Frame {iter} FORCE Fig 1A')
    plt.subplot(1, 2, 2)
    im_2 = plt.imshow(normalized_exc_data[:, iter].reshape(40, 40))
    plt.colorbar(im_2)
    plt.title(f'Target Frame {iter}')
    plt.tight_layout()