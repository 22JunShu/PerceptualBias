# Import necessary libraries
import torch
import numpy as np
import math
import os
import matplotlib.pyplot as plt

# --- 1. Configuration based on the filename ---
# Filename: run_noel_analysis.py_G1_B1_P2_F2_R0.01_GR60_params.pt
# From this, we infer:
GROUP_ID = 1      # ASD
BLOCK_ID_VAL = 1  # wFB1
P_VAL = 2         # L2 loss (posterior mean)
FOLD_HERE = 2
REG_WEIGHT = 0.1
GRID_SIZE = 60    # GR60
DEVICE = 'cpu'    # Always load to CPU for analysis portability

# --- Fixed experimental/model parameters (must match fitting script) ---
MIN_GRID = 0.0
MAX_GRID = 180.0  # Orientation range for Noel et al.
SENSORY_SPACE_VOLUME_NORMALIZED = 2 * math.pi
STIMULUS_SPACE_RANGE = MAX_GRID - MIN_GRID 
STIMULUS_STEP = MAX_GRID / GRID_SIZE

# --- 2. Construct File Paths ---
# You might need to adjust this base path if your notebook is not in the same relative location as the fitting script's execution
base_output_dir = "." # Assuming the params_G... folders are in the current directory or specify full path

script_name_base_list = ["run_noel_uniform_encoding.py", "run_noel_uniform_prior.py","run_noel_analysis.py", "run_noel_natural_prior.py"]
script_name_base = script_name_base_list[3] #"run_noel_uniform_encoding.py" # Base name of your fitting script

param_dir = os.path.join(base_output_dir, f"params_G{GROUP_ID}_B{BLOCK_ID_VAL}_P{P_VAL}/")
# The predictions file should have the same base name and be in the same directory
run_details_suffix = f"_G{GROUP_ID}_B{BLOCK_ID_VAL}_P{P_VAL}_F{FOLD_HERE}_R{REG_WEIGHT}_GR{GRID_SIZE}"
losses_dir = f"losses_G{GROUP_ID}_B{BLOCK_ID_VAL}_P{P_VAL}/"
figure_dir = f"figures_G{GROUP_ID}_B{BLOCK_ID_VAL}_P{P_VAL}/"
os.makedirs(figure_dir, exist_ok=True)

# Construct the specific part of the filename
run_details_suffix = f"_G{GROUP_ID}_B{BLOCK_ID_VAL}_P{P_VAL}_F{FOLD_HERE}_R{REG_WEIGHT}_GR{GRID_SIZE}"

params_filename = f"{script_name_base}{run_details_suffix}_params.pt"
losses_filename = f"{script_name_base}{run_details_suffix}_losses.npz"
predictions_filename = f"{script_name_base}{run_details_suffix}_predictions.npz" # Assumes it exists

PARAMS_FILE_PATH = os.path.join(param_dir, params_filename)
PREDICTIONS_FILE_PATH = os.path.join(param_dir, predictions_filename)

print(f"Attempting to load parameters from: {PARAMS_FILE_PATH}")
print(f"Attempting to load predictions from: {PREDICTIONS_FILE_PATH}")

# --- 3. Load Parameters and Predictions ---
loaded_parameters = None
grid_loaded_np = None
bias_total_predicted_np = None

if os.path.exists(PARAMS_FILE_PATH):
    loaded_parameters = torch.load(PARAMS_FILE_PATH, map_location=DEVICE)
    print(f"Successfully loaded parameters from: {PARAMS_FILE_PATH}")
else:
    print(f"ERROR: Parameters file not found at {PARAMS_FILE_PATH}")

if os.path.exists(PREDICTIONS_FILE_PATH):
    predictions_data = np.load(PREDICTIONS_FILE_PATH)
    if 'grid' in predictions_data:
        grid_loaded_np = predictions_data['grid']
    else:
        print(f"WARNING: 'grid' not found in {PREDICTIONS_FILE_PATH}. Reconstructing...")
        grid_loaded_np = np.linspace(MIN_GRID, MAX_GRID - STIMULUS_STEP, GRID_SIZE)
        
    if 'bias_total' in predictions_data:
        bias_total_predicted_np = predictions_data['bias_total']
    else:
        print(f"WARNING: 'bias_total' not found in {PREDICTIONS_FILE_PATH}.")
        
    print(f"Successfully loaded predictions from: {PREDICTIONS_FILE_PATH}")
else:
    print(f"ERROR: Predictions file not found at {PREDICTIONS_FILE_PATH}. Will reconstruct grid if needed.")
    # Reconstruct grid if predictions file doesn't exist, as it's essential
    grid_loaded_np = np.linspace(MIN_GRID, MAX_GRID - STIMULUS_STEP, GRID_SIZE)
# --- Load Losses ---
LOSSES_FILE_PATH = os.path.join(losses_dir, losses_filename) # Defined earlier based on your config

train_losses_loaded = None  # Initialized to None
test_losses_loaded = None   # Initialized to None

if os.path.exists(LOSSES_FILE_PATH):
    try:
        losses_data = np.load(LOSSES_FILE_PATH)
        if 'train_losses' in losses_data:
            train_losses_loaded = losses_data['train_losses']
        else:
            print(f"WARNING: 'train_losses' key not found in {LOSSES_FILE_PATH}")
        
        if 'test_losses' in losses_data:
            test_losses_loaded = losses_data['test_losses']
        else:
            print(f"WARNING: 'test_losses' key not found in {LOSSES_FILE_PATH}")
            
        if train_losses_loaded is not None or test_losses_loaded is not None:
             print(f"Successfully loaded some losses from: {LOSSES_FILE_PATH}")
        else:
            print(f"ERROR: 'train_losses' and 'test_losses' keys not found in {LOSSES_FILE_PATH}")

    except Exception as e:
        print(f"ERROR: Could not load or process losses file {LOSSES_FILE_PATH}. Exception: {e}")
else:
    print(f"ERROR: Losses file not found at {LOSSES_FILE_PATH}")

# --- 4. Reconstruct Model Components from Loaded Parameters ---
if loaded_parameters:
    # Reconstruct prior distribution
    prior_raw = loaded_parameters["prior_raw"].to(DEVICE)
    fitted_prior_torch = torch.softmax(prior_raw, dim=0)
    fitted_prior_np = fitted_prior_torch.detach().numpy()

    # Reconstruct encoding resources (volume_element)
    volume_raw = loaded_parameters["volume_raw"].to(DEVICE)
    # This is dF_sensory for each grid step in stimulus space (scaled by total sensory volume)
    fitted_volume_element_torch = SENSORY_SPACE_VOLUME_NORMALIZED * torch.softmax(volume_raw, dim=0)
    fitted_volume_element_np = fitted_volume_element_torch.detach().numpy()
    
    # Reconstruct sensory noise (sigma^2)
    sigma_logit = loaded_parameters["sigma_logit"].to(DEVICE)
    fitted_sigma2_sensory_torch = 4 * torch.sigmoid(sigma_logit[0]) # sigma_logit was saved as a single element tensor
    fitted_sigma2_sensory_np = fitted_sigma2_sensory_torch.detach().item()

    # Calculate Fisher Information J(theta)
    # F_prime_theta_approx = d(Sensory Space) / d(Stimulus Space)
    # fitted_volume_element_torch is the step in sensory space for one grid index step.
    # The corresponding step in stimulus space is STIMULUS_STEP = MAX_GRID / GRID_SIZE.
    F_prime_theta_approx_torch = fitted_volume_element_torch / STIMULUS_STEP
    fitted_J_theta_torch = (F_prime_theta_approx_torch**2) / fitted_sigma2_sensory_torch
    fitted_J_theta_np = fitted_J_theta_torch.detach().numpy()
    
    print("\n--- Reconstructed Model Components ---")
    print(f"Sensory Noise (sigma^2): {fitted_sigma2_sensory_np:.4f}")
    print(f"Prior shape: {fitted_prior_np.shape}")
    print(f"J(theta) shape: {fitted_J_theta_np.shape}")
    if grid_loaded_np is not None:
        print(f"Grid shape: {grid_loaded_np.shape}")

# --- 5. Numerical Derivative Helper Function (for circular data) ---
def circular_gradient(y, x, max_val):
    """
    Computes gradient for circular data y with respect to x.
    Pads y and x to handle wrap-around for circularity.
    """
    # Pad y: y_n, y_1, y_2, ..., y_n, y_1
    # Pad x: x_n - max_val, x_1, x_2, ..., x_n, x_1 + max_val
    
    # Ensure y and x are numpy arrays
    y_np = np.asarray(y)
    x_np = np.asarray(x)
    
    # Check if data is already effectively circular from padding in fitting (unlikely for this analysis)
    # For now, assume standard circular padding needed
    
    # Pad y: y_last, y_0, ..., y_n-1, y_0
    padded_y = np.concatenate(([y_np[-1]], y_np, [y_np[0]]))
    
    # Pad x: x_last - period, x_0, ..., x_n-1, x_0 + period
    period = x_np[-1] - x_np[0] + (x_np[1] - x_np[0]) # Full range
    # Should be MAX_GRID if x_np spans 0 to MAX_GRID-step
    
    dx = x_np[1] - x_np[0] # Assuming uniform grid
    padded_x = np.concatenate(([x_np[-1] - period], x_np, [x_np[0] + period]))
    
    # Compute gradient on padded data
    grad_padded = np.gradient(padded_y, padded_x, edge_order=2)
    
    # Return the central part corresponding to original data
    return grad_padded[1:-1]

# --- 6. Calculate Bias Components (Prior Attraction & Likelihood Repulsion) ---
if loaded_parameters and grid_loaded_np is not None:
    # Calculate log prior and its derivative
    log_fitted_prior_np = np.log(fitted_prior_np + 1e-12) # Add small epsilon for stability

    # Derivative of log prior: (log p(theta))'
    log_prior_prime_np = circular_gradient(log_fitted_prior_np, grid_loaded_np, MAX_GRID)

    # Calculate 1/J(theta) and its derivative: (1/J(theta))'
    one_over_J_theta_np = 1.0 / (fitted_J_theta_np + 1e-12) # Add epsilon for stability
    one_over_J_prime_np = circular_gradient(one_over_J_theta_np, grid_loaded_np, MAX_GRID)

    # Calculate Bias Components (Hahn & Wei 2024, Eq. 2 & 3)
    # Ensure J_theta is not zero where it's used in the denominator
    prior_attraction_term_np = (1.0 / (fitted_J_theta_np + 1e-12)) * log_prior_prime_np

    if P_VAL == 0: # MAP estimator
        C_p = 1.0 / 4.0
    else: # Lp loss (P_VAL > 0)
        C_p = (P_VAL + 2.0) / 4.0
    
    likelihood_repulsion_term_np = C_p * one_over_J_prime_np

    # Total predicted bias from decomposition
    bias_decomposed_total_np = prior_attraction_term_np + likelihood_repulsion_term_np
    
    print("\n--- Calculated Bias Components ---")
    # print(f"Prior Attraction (sample): {prior_attraction_term_np[:5]}")
    # print(f"Likelihood Repulsion (sample): {likelihood_repulsion_term_np[:5]}")

# --- 7. Visualization ---
if loaded_parameters and grid_loaded_np is not None:
    #plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Analysis for G{GROUP_ID}_B{BLOCK_ID_VAL}_P{P_VAL}_F{FOLD_HERE}_R{REG_WEIGHT}_GR{GRID_SIZE}", fontsize=16)

    # Plot 1: Fitted Prior P(θ)
    axs[0, 0].plot(grid_loaded_np, fitted_prior_np, label='Fitted Prior P(θ)', color='blue')
    axs[0, 0].set_title('Fitted Prior Distribution')
    axs[0, 0].set_xlabel('Orientation (deg)')
    axs[0, 0].set_ylabel('Probability Density')
    axs[0, 0].legend()
    axs[0, 0].set_xlim(MIN_GRID, MAX_GRID)

    # Plot 2: Fitted Fisher Information J(θ)
    axs[0, 1].plot(grid_loaded_np, fitted_J_theta_np, label='Fitted Fisher Info J(θ)', color='green')
    axs[0, 1].set_title('Fitted Fisher Information')
    axs[0, 1].set_xlabel('Orientation (deg)')
    axs[0, 1].set_ylabel('Fisher Information')
    axs[0, 1].legend()
    axs[0, 1].set_xlim(MIN_GRID, MAX_GRID)

    # Plot 3: Bias Components
    if bias_total_predicted_np is not None:
        axs[1, 0].plot(grid_loaded_np, bias_total_predicted_np, label='Total Bias (from _predictions.npz)', linestyle='--', color='purple', alpha=0.7)
    
    axs[1, 0].plot(grid_loaded_np, bias_decomposed_total_np, label='Total Bias (Recomputed from Decomp.)', color='black')
    axs[1, 0].plot(grid_loaded_np, prior_attraction_term_np, label=f'Prior Attraction (P={P_VAL})', linestyle='-.', color='dodgerblue')
    axs[1, 0].plot(grid_loaded_np, likelihood_repulsion_term_np, label=f'Likelihood Repulsion (P={P_VAL})', linestyle=':', color='red')
    axs[1, 0].set_title('Bias Components')
    axs[1, 0].set_xlabel('Orientation (deg)')
    axs[1, 0].set_ylabel('Bias (deg)')
    axs[1, 0].legend(fontsize='small')
    axs[1, 0].axhline(0, color='gray', linewidth=0.5)
    axs[1, 0].set_xlim(MIN_GRID, MAX_GRID)
    
    # Plot 4: Training and Test Losses (if loaded)
    if train_losses_loaded is not None and test_losses_loaded is not None:
        iterations = np.arange(len(train_losses_loaded)) * 50 # Assuming saved every 50 iterations
        axs[1, 1].plot(iterations, train_losses_loaded, label='Training NLL')
        axs[1, 1].plot(iterations, test_losses_loaded, label='Test (CV) NLL')
        axs[1, 1].set_title('Loss Curves')
        axs[1, 1].set_xlabel('Iteration')
        axs[1, 1].set_ylabel('Negative Log-Likelihood')
        axs[1, 1].legend()
        if len(test_losses_loaded) > 0:
             min_cv_nll_val = np.nanmin(test_losses_loaded) # Use nanmin if NaNs are possible
             min_cv_iter = iterations[np.nanargmin(test_losses_loaded)]
             axs[1,1].scatter([min_cv_iter], [min_cv_nll_val], color='red', s=50, zorder=5, label=f'Best CV NLL: {min_cv_nll_val:.3f}')
             axs[1,1].legend()

    else:
        axs[1, 1].text(0.5, 0.5, 'Losses file not found or empty.', ha='center', va='center')
        axs[1, 1].set_title('Loss Curves')


    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    #os.makedirs("figures_G{GROUP_ID}_B{BLOCK_ID_VAL}_P{P_VAL}", exist_ok=True)
    #plt.savefig(f"figures_G{GROUP_ID}_B{BLOCK_ID_VAL}_P{P_VAL}.jpg")
    plt.savefig(f"{figure_dir}{script_name_base}{run_details_suffix}.jpg")
    plt.show()

else:
    print("Parameters file could not be loaded. Cannot proceed with full analysis.")