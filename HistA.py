# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 21:57:46 2024

@author: esnow
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.sparse import random as sparse_random
from skimage.transform import resize
import math
import numpy.random as npr
import seaborn as sns

def calculate_mse(frameA, frameB):
    # Calculate the Mean Squared Error between two frames
    err = np.sum((frameA.astype("float") - frameB.astype("float")) ** 2)
    err /= float(frameA.shape[0])
    return err

def calculate_errors(frames1, frames2):
    if len(frames1) != len(frames2):
        raise ValueError("The two sets of frames must contain the same number of elements.")
    
    errors = []
    for frame1, frame2 in zip(frames1, frames2):
        if frame1.shape != frame2.shape:
            raise ValueError("All corresponding frames must have the same dimensions.")
        error = calculate_mse(frame1, frame2)
        errors.append(error)
    
    return errors

def plot_histogram(errors):
    # Set up the Seaborn style
    sns.set(style="whitegrid")
    
    # Plot the histogram with Seaborn
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, bins=20, color='blue', edgecolor='black', alpha=0.7, stat='probability')
    
    # Labeling the plot
    plt.xlabel('Mean Square Error', fontsize=14)
    plt.ylabel('Occurrence Probability', fontsize=14)
    plt.title('Histogram of Error Distributions', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Show the plot
    plt.tight_layout()
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
new_dt = 10    # New temporal resolution is 10 ms
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
a = normalized_exc_data
# Vectorize retinal data
vectorized_retinal_data = normalized_retinal_data.reshape(new_size[0] * new_size[1], T_resized)
N, T = vectorized_retinal_data.shape
g = 1.5  # g greater than 1 leads to chaotic networks.
alpha = 1.0
dt = 1
tau = 1
t_vec = np.linspace(0, T, int(T/dt))
ampInWN=0.001
P0=1.0
tauWN= 0.1
ampWN = math.sqrt(tauWN / dt)
iWN = ampWN * npr.randn(N, len(t_vec))
inputWN = np.ones((N, len(t_vec)))
for tt in range(1, len(t_vec)):
    inputWN[:, tt] = iWN[:, tt] + (inputWN[:, tt - 1] - iWN[:, tt]) * np.exp(- (dt / tauWN))
inputWN = ampInWN * inputWN
# Initialize variables
W = np.zeros((N, N))
W_v = np.zeros((N, N))
# Assuming p, N, and g are defined
p = 1 # example value for p
u = np.random.uniform(-1, 1, size=(N))
# Calculate scale
scale = 1.0 / np.sqrt(p * N)

# Generate a sparse random matrix with normally distributed values
# The random function generates values in [0, 1), so we need to use a normal distribution
sparse_matrix = sparse_random(N, N, density=p, format='csr', data_rvs=np.random.randn)

# Convert to dense matrix and scale
J = sparse_matrix.toarray() * g * scale
P = np.eye(N) / alpha  # Initialize learning update matrix P
x = np.zeros((N, len(t_vec)))
x_0 = 0.5 * np.random.randn(N)
z_0 = 0.5 * np.random.randn()

x[:, 0] = x_0
r = np.tanh(x_0)
r_1 = r
r_2 = r
r_3 = r
r_4 = r
r_v = r
z = z_0
ps_vec = []
x_1 = x
x_2 = x
x_3 = x
x_4 = x
x_v = x
J_1 = J
J_2 = J
J_3 = J
J_4 = J
J_v = J
P_1 = P
P_2 = P
P_3 = P
P_4 = P
P_v = P
# Initialize feed-forward connection matrix
C1 = np.random.randn(N, N) / np.sqrt(N)
C2 = np.random.randn(N, N) / np.sqrt(N)
C3 = np.random.randn(N, N) / np.sqrt(N)
C_v = np.random.randn(N, N) / np.sqrt(N)
# The data is generated from a normal distribution with a small variance
sparsity = 0.1
rvs = lambda x: np.random.randn(x) / np.sqrt(N)   
C_sparse = sparse_random(N, N, density=sparsity, data_rvs=rvs)
C_sparse = C_sparse.toarray()
# Training Loop only time loop
for t in range(1, len(t_vec)):
    # Update neuron state
    x_1[:, t] = (1 - dt / tau) * x_1[:, t-1] + (dt / tau) * (np.dot(J_1, r_1)) + u*vectorized_retinal_data[:, t-1]
    # Update firing rate
    r_1 = np.tanh(x_1[:, t])  # Ensure r is (N,)
    y_input_1 = np.dot(C1, r_1)
    
    x_2[:, t] = (1 - dt / tau) * x_2[:, t-1] + (dt / tau) * (np.dot(J_2, r_2)) + u * y_input_1
    # Update firing rate
    r_2 = np.tanh(x_2[:, t])  # Ensure r is (N,)
    
    y_input_2 = np.dot(C2, r_2)
    x_3[:, t] = (1 - dt / tau) * x_3[:, t-1] + (dt / tau) * (np.dot(J_3, r_3)) + u * y_input_2
    # Update firing rate
    r_3 = np.tanh(x_3[:, t])  # Ensure r is (N,)
    
    y_input_3 = np.dot(C3, r_3)
    x_4[:, t] = (1 - dt / tau) * x_4[:, t-1] + (dt / tau) * (np.dot(J_4, r_4)) + u * y_input_3
    # Update firing rate
    r_4 = np.tanh(x_4[:, t])  # Ensure r is (N,)
    # Update network output
    z = np.dot(W.T, r_4)  # Ensure z is a scalar
    
    # Compute error
    e_minus = z - a[:,t-1]
    
    # #### comment out to check the simple mode
    k1 = np.dot(P_1, r_1)
    rPr_1 = np.dot(r_1.T, k1)
    c1 = 1.0 / (1.0 + rPr_1)
    # C1 -=  c1 * np.outer(k1, e_minus)
    J_1 = J_1 - c1 * np.outer(e_minus, k1)
    P_1 = P_1 - c1 * np.outer(k1, k1)
    
    k2 = np.dot(P_2, r_2)
    rPr = np.dot(r_2.T, k2)
    c2 = 1.0 / (1.0 + rPr)
    # C2 -=  c2 * np.outer(k2, e_minus)
    J_2 = J_2 - np.outer(e_minus, k2)*c2
    P_2 = P_2 - c2 * np.outer(k2, k2)
    
    
    k3 = np.dot(P_3, r_3)
    rPr = np.dot(r_3.T, k3)
    c3 = 1.0 / (1.0 + rPr)
    # C3 -=  c3 * np.outer(k3, e_minus)
    J_3 = J_3 - np.outer(e_minus, k3)*c3
    P_3 = P_3 - c3 * np.outer(k3, k3)
    # ####
    k4 = np.dot(P_4, r_4)
    rPr = np.dot(r_4.T, k4)
    c = 1.0 / (1.0 + rPr)
    W -= c * np.outer(k4, e_minus) #key training process #sensitive to errorc * np.outer(k, e)
    J_4 = J_4 - np.outer(e_minus, k4)*c
    P_4 = P_4 - c * np.outer(k4, k4)
    #########
    y_input_v = y_input_2
    x_v[:, t] = (1 - dt / tau) * x_v[:, t-1] + (dt / tau) * (np.dot(J_v, r_v)) + u * y_input_v
    # Update firing rate
    r_v = np.tanh(x_v[:, t])  # Ensure r is (N,)
    # Update network output
    z_v = np.dot(W_v.T, r_v)  # Ensure z is a scalar
    
    # Compute error
    e_v = z_v - a[:,t-1]
    
    # #### comment out to check the simple mode
    kv = np.dot(P_v, r_v)
    rPr_v = np.dot(r_v.T, kv)
    cv = 1.0 / (1.0 + rPr_v)
    W_v -= cv * np.outer(kv, e_v) #key training process #sensitive to errorc * np.outer(k, e)
    # C1 -=  c1 * np.outer(k1, e_minus)
    J_v = J_v - cv * np.outer(e_v, kv)
    P_v = P_v - cv * np.outer(kv, kv)
# Testing Phase
simtime_len = len(t_vec)
zpt = np.zeros((N,simtime_len))
zpt_v = np.zeros((N, simtime_len))
x_test_1 = x_1[:, 0]
x_test_3 = x_3[:, 0]
x_test_2 = x_2[:, 0]
x_test_4 = x_4[:, 0]
x_test_v = x_v[:, 0]
r_test_v = np.tanh(x_test_v)
r_test_1 = np.tanh(x_test_1)
r_test_3 = np.tanh(x_test_3)
r_test_2 = np.tanh(x_test_2)
r_test_4 = np.tanh(x_test_4)

for ti in range(simtime_len):
    x_test_1 = (1.0 - dt) * x_test_1 + np.dot(J_1, r_test_1 * dt)  + u*vectorized_retinal_data[:, ti-1]
    r_test_1 = np.tanh(x_test_1)
    # y_test_input_1 = np.dot(C_sparse, r_test_1)
    y_test_input_1 = np.dot(C1 * 1, r_test_1)
    x_test_2 = (1.0 - dt) * x_test_2 + np.dot(J_2, r_test_2 * dt) +  u*y_test_input_1
    r_test_2 = np.tanh(x_test_2)
    y_test_input_2 = np.dot(C2, r_test_2)
    x_test_3 = (1.0 - dt) * x_test_3+ np.dot(J_3, r_test_3 * dt) +  u*y_test_input_2 + u * inputWN[:, ti-1]
    r_test_3 = np.tanh(x_test_3)
    y_test_input_3 = np.dot(C3, r_test_3)
    x_test_4 = (1.0 - dt) * x_test_4+ np.dot(J_4, r_test_4 * dt) +  u*y_test_input_3
    r_test_4 = np.tanh(x_test_4)
    zpt[:, ti] = np.dot(W.T, r_test_4)  # Use the final trained weights
    y_test_input_v = y_test_input_2
    x_test_v = (1.0 - dt) * x_test_v+ np.dot(J_v, r_test_v * dt) +  u*y_test_input_v
    r_test_v = np.tanh(x_test_v)
    zpt_v[:, ti] = np.dot(W_v.T, r_test_v)  # Use the final trained weights


error_avg_d = np.mean(np.abs(zpt - a))
print(f'Testing MAE: {error_avg_d:.3f}')

error_avg_v = np.mean(np.abs(zpt_v - a))
print(f'Testing MAE: {error_avg_v:.3f}')
errors = calculate_errors(zpt, a)
plot_histogram(errors)
# Reset matplotlib parameters to avoid Seaborn style affecting other plots
plt.rcdefaults()
errors = calculate_errors(zpt_v, a)
plot_histogram(errors)
# Reset matplotlib parameters to avoid Seaborn style affecting other plots
plt.rcdefaults()
for iter in range(simtime_len):
    # Create a figure and 2x2 layout of subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # Plot on each subplot
    im_1 = axes[0, 0].imshow(zpt[:, iter].reshape(new_size))
    axes[0, 0].set_title(f'Output Frame {iter} Branch dHVA')
    fig.colorbar(im_1, ax=axes[0, 0])
    im_2 =axes[0, 1].imshow(a[:, iter].reshape(new_size ))
    axes[0, 1].set_title(f'Target Frame {iter}')
    fig.colorbar(im_2, ax=axes[0, 1])
    im_3 =axes[1, 0].imshow(zpt_v[:, iter].reshape(new_size))
    axes[1, 0].set_title(f'Output Frame {iter} Branch vHVA')
    fig.colorbar(im_3, ax=axes[1, 0])
    im_4 =axes[1, 1].imshow(a[:, iter].reshape(new_size ))
    axes[1, 1].set_title(f'Target Frame {iter}')
    fig.colorbar(im_4, ax=axes[1, 1])
    # Add spacing between subplots to avoid overlap
    fig.tight_layout()
    