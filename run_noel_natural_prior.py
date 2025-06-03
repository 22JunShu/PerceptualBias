import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import torch
from scipy.io import loadmat # For loading .mat files
# Assuming util.py contains all necessary helper functions from the original script
# You'll need to ensure this util.py file is available and contains these functions:
# MakeFloatTensor, MakeLongTensor, MakeZeros, ToDevice, 
# computeCircularMeanWeighted, computeCircularSDWeighted, 
# makeGridIndicesCircular, savePlot

# For demonstration, let's define dummy versions of util functions if util.py is not set up
# In a real scenario, you'd have a proper util.py file
# def MakeFloatTensor(data): return torch.tensor(data, dtype=torch.float33)
# def MakeLongTensor(data): return torch.tensor(data, dtype=torch.long)
# def MakeZeros(shape): return torch.zeros(shape)
# def ToDevice(tensor): global DEVICE; return tensor.to(DEVICE)

# def makeGridIndicesCircular(GRID_SIZE, MIN_VAL, MAX_VAL, max_angle_circular=180.0):
#     grid_out = ToDevice(MakeFloatTensor([x / GRID_SIZE * (MAX_VAL - MIN_VAL) for x in range(GRID_SIZE)]) + MIN_VAL)
#     # grid_indices_here_out assumes a potentially larger grid for some estimator internal workings,
#     # but for 0-180, it's usually just 0 to GRID_SIZE-1 scaled.
#     # The original template script had a more complex grid_indices_here for some estimators.
#     # For circular 0-180, a simple mapping is often sufficient for the model logic part.
#     # The specific estimators might internally use a more complex grid construction.
#     # Let's ensure grid_indices_here has a shape compatible with estimator expectations (e.g. (GRID_SIZE, GRID_SIZE))
#     # For now, this is simplified. The estimator's set_parameters should handle its specific needs.
#     grid_indices_here_out = ToDevice(MakeFloatTensor([x for x in range(GRID_SIZE)]))
#     if grid_indices_here_out.max() < GRID_SIZE and GRID_SIZE > 0 : # from template, ensure it's not too small
#          grid_indices_here_out_temp = torch.arange(0, GRID_SIZE*2, device=DEVICE).float() # Example for a larger effective grid for some computations
#          # This specific part might need to align with how mapCircularEstimator expects grid_indices_here
#          # The template 'RunGardelle_NaturalPrior_Zero.py' uses:
#          # grid, grid_indices_here = makeGridIndicesCircular(GRID, MIN_GRID, MAX_GRID)
#          # assert grid_indices_here.max() >= GRID
#          # For simplicity, let's make it a 1D tensor of indices for now.
#          # The estimators should be robust or take simple grid/indices.
#     # The crucial part is that `grid` represents the actual stimulus values.
#     return grid_out, grid_indices_here_out

from util import *
# Estimators (ensure these .py files are in your Python path)
# Make sure the imported estimators are the base versions or the specific numbered ones you intend to use.
from mapCircularEstimator import MAPCircularEstimator # For P = 0
from cosineEstimator import CosineEstimator # For P > 0


def load_noel_data(data_path_root, group_id, block_id_val):
    """
    Loads Noel et al. (2021) data for a specific group and block.
    """
    print(f"Loading data from root: {data_path_root} for Group ID: {group_id}, Block ID: {block_id_val}")

    stimuli_list = []
    responses_list = []
    subject_ids_list_collector = [] # To collect numerical subject IDs
    block_ids_list_collector = []

    if group_id == 1: # ASD
        group_name_folder = "ASD"
        subject_prefix = "A"
        num_subjects = 17
    elif group_id == 0: # TD (Neurotypical)
        group_name_folder = "TD"
        subject_prefix = "H"
        num_subjects = 25
    else:
        raise ValueError("group_id must be 0 (TD) or 1 (ASD)")

    if block_id_val == 0:
        block_name_folder = "woFB"
        block_name_short_in_file = "woFB"
    elif block_id_val == 1:
        block_name_folder = "wFB1"
        block_name_short_in_file = "wFB1"
    elif block_id_val == 2:
        block_name_folder = "wFB2"
        block_name_short_in_file = "wFB2"
    else:
        raise ValueError("block_id_val must be 0 (woFB), 1 (wFB1), or 2 (wFB2)")

    # !!! IMPORTANT: Adjust this key to the actual variable name in your .mat files !!!
    mat_data_key = 'all_data' # Common placeholder, check your .mat files

    files_found_count = 0
    for i_subj in range(1, num_subjects + 1):
        subject_id_str = f"{subject_prefix}{i_subj}"
        filename = f"{subject_id_str}{block_name_short_in_file}.mat"
        filepath = os.path.join(data_path_root, block_name_folder, group_name_folder, filename)

        if os.path.exists(filepath):
            files_found_count +=1
            try:
                mat_data = loadmat(filepath)
                if mat_data_key not in mat_data:
                    print(f"ERROR: Key '{mat_data_key}' not found in {filepath}. Available keys: {list(mat_data.keys())}")
                    continue

                data_array = mat_data[mat_data_key]
                if data_array.shape[0] != 2:
                    print(f"ERROR: Data in {filepath} does not have 2 rows (shape: {data_array.shape}). Skipping.")
                    continue
                
                targets = data_array[0, :].astype(np.float32)%180 # First row: target
                responses = data_array[1, :].astype(np.float32)%180 # Second row: response
                num_trials = targets.shape[0]

                stimuli_list.append(torch.from_numpy(targets))
                responses_list.append(torch.from_numpy(responses))
                subject_ids_list_collector.append(torch.full((num_trials,), i_subj, dtype=torch.long)) # Use numerical subject ID
                block_ids_list_collector.append(torch.full((num_trials,), block_id_val, dtype=torch.long))

            except Exception as e:
                print(f"Error loading or processing file {filepath}: {e}")
        else:
            print(f"Warning: File not found {filepath}")

    if not stimuli_list:
        raise FileNotFoundError(f"No data files successfully loaded for Group: {group_name_folder}, Block: {block_name_folder}. Files found: {files_found_count}")

    all_stimuli = ToDevice(torch.cat(stimuli_list))
    all_responses = ToDevice(torch.cat(responses_list))
    all_subject_ids = ToDevice(torch.cat(subject_ids_list_collector))
    all_block_ids = ToDevice(torch.cat(block_ids_list_collector))
    
    print(f"Successfully loaded {all_stimuli.shape[0]} trials for {files_found_count} subjects.")
    return all_stimuli, all_responses, all_subject_ids, all_block_ids


# --- Global DEVICE definition ---
DEVICE = 'cpu' # Default, will be updated by command line arg

def main():
    global OPTIMIZER_VERBOSE, P, FOLD_HERE, REG_WEIGHT, GRID, SHOW_PLOT, DEVICE
    global FILE, MIN_GRID, MAX_GRID, grid, grid_indices_here, xValues, stimulus_, responses_
    global N_FOLDS, Fold, subject_ids, block_ids, observations_x, observations_y
    global init_parameters, learning_rate, optim
    global SENSORY_SPACE_VOLUME_NORMALIZED
    global PARAMS_FILE_PATH, LOSSES_FILE_PATH

    # 1. OBTAIN ARGUMENTS
    if len(sys.argv) < 9:
        print("Usage: python script_name.py <DATA_ROOT_PATH> <GROUP_ID> <BLOCK_ID> <P_VAL> <FOLD_HERE> <REG_WEIGHT> <GRID_SIZE> <DEVICE> [SHOW_PLOT]")
        print("  DATA_ROOT_PATH: e.g., F:\\Courses2025\\认知建模基础\\ASD_Encoding_2020")
        print("  GROUP_ID: 0 for TD, 1 for ASD")
        print("  BLOCK_ID: 0 for woFB, 1 for wFB1, 2 for wFB2")
        print("  P_VAL: Loss exponent (0 for MAP, >0 for Lp/Cosine)")
        print("  DEVICE: 'cpu' or 'cuda'")
        sys.exit(1)

    DATA_ROOT_PATH = sys.argv[1]
    GROUP_ID = int(sys.argv[2])
    BLOCK_ID_VAL = int(sys.argv[3]) # Renamed to avoid conflict
    P = int(sys.argv[4])
    FOLD_HERE = int(sys.argv[5])
    REG_WEIGHT = float(sys.argv[6])
    GRID = int(sys.argv[7])
    DEVICE = sys.argv[8]
    SHOW_PLOT = (len(sys.argv) < 10) or (sys.argv[9] == "SHOW_PLOT")
    
    OPTIMIZER_VERBOSE = False # Set to True for detailed optimization output

    script_name = os.path.basename(__file__)
    log_dir = "logs/CROSSVALID/"
    param_dir = f"params_G{GROUP_ID}_B{BLOCK_ID_VAL}_P{P}/" # Store params in subfolders
    loss_dir = f"losses_G{GROUP_ID}_B{BLOCK_ID_VAL}_P{P}/"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(param_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)

    FILE = f"{log_dir}{script_name}_G{GROUP_ID}_B{BLOCK_ID_VAL}_P{P}_F{FOLD_HERE}_R{REG_WEIGHT}_GR{GRID}.txt"
    PARAMS_FILE_PATH = f"{param_dir}{script_name}_G{GROUP_ID}_B{BLOCK_ID_VAL}_P{P}_F{FOLD_HERE}_R{REG_WEIGHT}_GR{GRID}_params.pt"
    LOSSES_FILE_PATH = f"{loss_dir}{script_name}_G{GROUP_ID}_B{BLOCK_ID_VAL}_P{P}_F{FOLD_HERE}_R{REG_WEIGHT}_GR{GRID}_losses.npz"


    observations_x, observations_y, subject_ids, block_ids = load_noel_data(DATA_ROOT_PATH, GROUP_ID, BLOCK_ID_VAL)

    N_FOLDS = 10
    if FOLD_HERE >= N_FOLDS:
        raise ValueError(f"FOLD_HERE ({FOLD_HERE}) must be less than N_FOLDS ({N_FOLDS})")

    Fold = torch.zeros_like(subject_ids, dtype=torch.long, device=DEVICE)
    unique_subjects_tensor = torch.unique(subject_ids)
    random_generator = random.Random(10) 
    for subj_id_tensor in unique_subjects_tensor:
        subj_id = subj_id_tensor.item()
        subj_trials_indices = (subject_ids == subj_id).nonzero(as_tuple=True)[0]
        
        # Ensure subj_trials_indices is a Python list for shuffle
        subj_trials_indices_list = subj_trials_indices.cpu().tolist()
        random_generator.shuffle(subj_trials_indices_list)
        
        fold_size = len(subj_trials_indices_list) // N_FOLDS
        for k in range(N_FOLDS):
            start_idx = k * fold_size
            end_idx = (k + 1) * fold_size if k < N_FOLDS - 1 else len(subj_trials_indices_list)
            # Convert indices back to tensor to assign to Fold tensor
            Fold[torch.tensor(subj_trials_indices_list[start_idx:end_idx], device=DEVICE)] = k


    MIN_GRID = 0
    MAX_GRID = 180.0

    grid_torch, grid_indices_here_torch = makeGridIndicesCircular(GRID, MIN_GRID, MAX_GRID)
    grid = grid_torch # Use the global variable name from the template
    grid_indices_here = grid_indices_here_torch


    xValues_list = []
    for x_obs in observations_x:
       xValues_list.append(int(torch.argmin((grid - x_obs).abs()))) # grid is already on DEVICE
    xValues = ToDevice(MakeLongTensor(xValues_list))

    stimulus_ = xValues
    responses_ = observations_y

    init_parameters = {}
    init_parameters["sigma_logit"] = MakeFloatTensor([-1.0]).view(1).to(DEVICE)
    init_parameters["log_motor_var"] = MakeFloatTensor([-4.0]).view(1).to(DEVICE)
    init_parameters["mixture_logit"] = MakeFloatTensor([-5.0]).view(1).to(DEVICE)
    init_parameters["prior_raw"] = MakeZeros(GRID).to(DEVICE)
    init_parameters["volume_raw"] = MakeZeros(GRID).to(DEVICE)

    for param_name, param_tensor in init_parameters.items():
        param_tensor.requires_grad = True

    learning_rate = 0.01
    optim = torch.optim.Adam([p for p in init_parameters.values() if p.requires_grad], lr=learning_rate)
    
    SENSORY_SPACE_VOLUME_NORMALIZED = 2 * math.pi


    SCALE_ESTIMATOR = 50.0
    KERNEL_WIDTH_ESTIMATOR = 0.05

    if P == 0:
        MAPCircularEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE,
                                           KERNEL_WIDTH=KERNEL_WIDTH_ESTIMATOR, SCALE=SCALE_ESTIMATOR,
                                           MIN_GRID=MIN_GRID, MAX_GRID=MAX_GRID,
                                           UPDATE_DECAY_FACTOR = 10)
        print("Using MAPCircularEstimator (P=0)")
    else:
        CosineEstimator.set_parameters(GRID=GRID, OPTIMIZER_VERBOSE=OPTIMIZER_VERBOSE, P=P,
                                       SQUARED_SENSORY_SIMILARITY=SQUARED_SENSORY_SIMILARITY,
                                       SQUARED_SENSORY_DIFFERENCE=SQUARED_SENSORY_DIFFERENCE,
                                       SCALE=SCALE_ESTIMATOR) # CosineEstimator from example also had SCALE
        print(f"Using CosineEstimator (P={P})")

    run_model_fitting_loop() # Call the main loop

def SQUARED_STIMULUS_DIFFERENCE(x):
    global MAX_GRID
    return torch.sin(math.pi * x / MAX_GRID)
def SQUARED_STIMULUS_SIMILARITY(x):
    global MAX_GRID
    return torch.cos(math.pi * x / MAX_GRID)

def SQUARED_SENSORY_SIMILARITY(x_sensory_diff):
    return torch.cos(x_sensory_diff)
def SQUARED_SENSORY_DIFFERENCE(x_sensory_diff):
    return torch.sin(x_sensory_diff)


def computeBias(stimulus_mapped_to_grid, 
                responses_observed,
                current_block_id_val, # from global BLOCK_ID_VAL
                current_fold_indices,
                all_fold_assignments,
                model_parameters,
                loss_exponent_p, # from global P
                compute_predictions=True,
                loss_reduce='mean'):
    global GRID, MIN_GRID, MAX_GRID, grid, grid_indices_here, DEVICE, SENSORY_SPACE_VOLUME_NORMALIZED

    motor_variance = torch.exp(model_parameters["log_motor_var"])
    sigma2 = 4 * torch.sigmoid(model_parameters["sigma_logit"][0]) # sigma_logit is now (1,)

    volume_element = SENSORY_SPACE_VOLUME_NORMALIZED * torch.softmax(model_parameters["volume_raw"], dim=0)
    F = torch.cat([MakeZeros(1).to(DEVICE), torch.cumsum(volume_element, dim=0)], dim=0)
    prior = 2-(torch.sin(grid*math.pi/180).abs())
    prior = prior/prior.sum()
    model_parameters['prior_raw'] = torch.log(prior)

    MASK_fold = (all_fold_assignments.unsqueeze(0) == current_fold_indices.unsqueeze(1)).any(dim=0)
    stimulus = stimulus_mapped_to_grid[MASK_fold]
    responses = responses_observed[MASK_fold]

    if stimulus.view(-1).size()[0] == 0:
        empty_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True) if loss_reduce == 'sum' else torch.tensor(float('nan'), device=DEVICE)
        return empty_loss, None, None, (None, None)

    sensory_diff = F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1)
    log_sensory_likelihoods = (SQUARED_SENSORY_SIMILARITY(sensory_diff) / sigma2)
    log_sensory_likelihoods = log_sensory_likelihoods + volume_element.unsqueeze(1).log() # Add log F'(theta)
    sensory_likelihoods = torch.softmax(log_sensory_likelihoods, dim=0)
    likelihoods = sensory_likelihoods

    # Posterior p(theta | m) ~ p(m | theta) p(theta)
    # Estimator scripts expect posterior where columns are p(theta_i | m_j)
    posterior_for_estimator = (likelihoods.t() * prior.unsqueeze(0)).t() 
    posterior_for_estimator = posterior_for_estimator / (posterior_for_estimator.sum(dim=0, keepdim=True) + 1e-9)


    if loss_exponent_p == 0:
        bayesianEstimate = MAPCircularEstimator.apply(grid_indices_here, posterior_for_estimator)
    else:
        bayesianEstimate = CosineEstimator.apply(grid_indices_here, posterior_for_estimator)

    bayesianEstimate_degrees = bayesianEstimate * (MAX_GRID / GRID)
    
    # p(response | m)
    motor_error_term = SQUARED_STIMULUS_SIMILARITY(bayesianEstimate_degrees.unsqueeze(0) - responses.unsqueeze(1))
    log_normalizing_constant_motor = torch.logsumexp(SQUARED_STIMULUS_SIMILARITY(grid) / motor_variance, dim=0) + math.log(MAX_GRID / GRID)
    log_motor_response_likelihoods_given_m = (motor_error_term / motor_variance) - log_normalizing_constant_motor
    
    # p(m | stimulus_obs)
    prob_m_given_stimulus_obs = likelihoods[:, stimulus].t()
    
    log_likelihood_per_trial_per_m = log_motor_response_likelihoods_given_m + torch.log(prob_m_given_stimulus_obs + 1e-9) # Add 1e-9 for stability
    log_likelihood_per_trial = torch.logsumexp(log_likelihood_per_trial_per_m, dim=1)
    
    uniform_part_logit = model_parameters["mixture_logit"]
    uniform_part_prob = torch.sigmoid(uniform_part_logit)
    log_uniform_likelihood = torch.tensor(-math.log(MAX_GRID), device=DEVICE)

    # # debug: print the shape of each variable
    # print("log_likelihood_per_trial.shape:", log_likelihood_per_trial.shape)
    # print("uniform_part_prob.shape:", uniform_part_prob.shape)
    # print("log_uniform_likelihood", log_uniform_likelihood)
    # log_final_likelihood_per_trial = torch.logsumexp(
    #     torch.stack([
    #         torch.log(1 - uniform_part_prob + 1e-9) + log_likelihood_per_trial,
    #         torch.log(uniform_part_prob + 1e-9) + log_uniform_likelihood
    #     ]), dim=0
    # )

    # --- START OF CORRECTED SECTION ---
    # Ensure the uniform component is broadcastable to the shape of log_likelihood_per_trial
    model_component_log_likelihood = torch.log(1 - uniform_part_prob + 1e-9) + log_likelihood_per_trial
    
    # This part creates the scalar value for the uniform component's log likelihood
    uniform_component_log_likelihood_scalar = torch.log(uniform_part_prob + 1e-9) + log_uniform_likelihood
    
    # Expand the scalar uniform component to match the shape of the model component
    uniform_component_log_likelihood_expanded = uniform_component_log_likelihood_scalar.expand_as(model_component_log_likelihood)

    log_final_likelihood_per_trial = torch.logsumexp(
        torch.stack([
            model_component_log_likelihood,
            uniform_component_log_likelihood_expanded # Use the expanded tensor here
        ]), dim=0
    )
    loss = -log_final_likelihood_per_trial.mean() if loss_reduce == 'mean' else -log_final_likelihood_per_trial.sum()

    pred_bias_total, pred_variability, pred_prior_attraction, pred_likelihood_repulsion = None, None, None, None
    if compute_predictions:
        expected_estimate_given_theta_true = torch.zeros(GRID, device=DEVICE)
        for i_theta_true in range(GRID):
            prob_m_dist = likelihoods[:, i_theta_true]
            expected_estimate_given_theta_true[i_theta_true] = torch.sum(bayesianEstimate_degrees * prob_m_dist)
        
        pred_bias_total = expected_estimate_given_theta_true - grid
        pred_bias_total = (pred_bias_total + MAX_GRID / 2) % MAX_GRID - MAX_GRID / 2
        
        # Placeholder for variability calculation
        # pred_variability = ... 

        # Placeholder for bias decomposition - requires numerical derivatives of log(prior) and 1/J(theta)
        # J_theta = (volume_element / (SENSORY_SPACE_VOLUME_NORMALIZED / GRID))**2 / sigma2
        # log_prior_prime = ...
        # one_over_J_prime = ...
        # pred_prior_attraction = (1/J_theta) * log_prior_prime
        # C_p = (P+2)/4 if P > 0 else 1/4
        # pred_likelihood_repulsion = C_p * one_over_J_prime
        pass


    if torch.isnan(loss):
        print("NAN loss detected in computeBias!")
        # print all inputs and intermediate variables if this happens
        
    return loss, pred_bias_total, pred_variability, (pred_prior_attraction, pred_likelihood_repulsion)


def run_model_fitting_loop():
    global optim, learning_rate, init_parameters, stimulus_, responses_, BLOCK_ID_VAL, Fold, P, GRID, FILE, PARAMS_FILE_PATH, LOSSES_FILE_PATH, N_FOLDS, REG_WEIGHT

    losses_by_epoch = []
    cross_valid_losses_by_epoch = []
    best_cv_nll = float('inf')
    
    MAX_GRAD_NORM = 1.0 # Example value for gradient clipping

    for iteration in range(5000): # Max iterations
        current_parameters = init_parameters
        optim.zero_grad()
        
        train_fold_indices = ToDevice(MakeLongTensor([i for i in range(N_FOLDS) if i != FOLD_HERE]))
        
        loss_train, _, _, _ = computeBias(
            stimulus_, responses_, BLOCK_ID_VAL, train_fold_indices, Fold,
            current_parameters, P, compute_predictions=False, loss_reduce='sum'
        )
        
        num_train_trials = sum([(Fold == f_idx).sum().item() for f_idx in train_fold_indices.cpu().tolist()])

        #reg_prior = ((current_parameters["prior_raw"][1:] - current_parameters["prior_raw"][:-1]).pow(2).sum() + \
        #             (current_parameters["prior_raw"][0] - current_parameters["prior_raw"][-1]).pow(2)) / GRID
        reg_prior = 0
        reg_volume = ((current_parameters["volume_raw"][1:] - current_parameters["volume_raw"][:-1]).pow(2).sum() + \
                      (current_parameters["volume_raw"][0] - current_parameters["volume_raw"][-1]).pow(2)) / GRID
        
        # Normalize loss by number of training trials
        normalized_loss_train = loss_train / num_train_trials if num_train_trials > 0 else torch.tensor(0.0, device=DEVICE)
        total_loss_train_with_reg = normalized_loss_train + REG_WEIGHT * (reg_prior + reg_volume)

        if torch.isnan(total_loss_train_with_reg):
            print(f"Iteration {iteration}: NaN training loss with regularization. Stopping.")
            break
            
        total_loss_train_with_reg.backward()
        torch.nn.utils.clip_grad_norm_([p_val for p_val in current_parameters.values() if p_val.grad is not None and p_val.requires_grad], MAX_GRAD_NORM)
        optim.step()

        if iteration % 50 == 0:
            test_fold_idx = ToDevice(MakeLongTensor([FOLD_HERE]))
            with torch.no_grad():
                loss_test, bias_pred, var_pred, (attr_pred, rep_pred) = computeBias(
                    stimulus_, responses_, BLOCK_ID_VAL, test_fold_idx, Fold,
                    current_parameters, P, compute_predictions=True, loss_reduce='sum'
                )
            
            num_test_trials = (Fold == FOLD_HERE).sum().item()
            cv_nll = loss_test.item() / num_test_trials if num_test_trials > 0 else float('nan')
            
            train_nll_for_print = normalized_loss_train.item()
            losses_by_epoch.append(train_nll_for_print)
            cross_valid_losses_by_epoch.append(cv_nll)

            print(f"Iter {iteration}: Train NLL: {train_nll_for_print:.4f}, Test NLL: {cv_nll:.4f}, LR: {learning_rate:.6f}")
            # print(f"  SigmaLogit: {current_parameters['sigma_logit'].item():.3f}, MotorLogVar: {current_parameters['log_motor_var'].item():.3f}, MixLogit: {current_parameters['mixture_logit'].item():.3f}")

            if cv_nll < best_cv_nll and not np.isnan(cv_nll):
                best_cv_nll = cv_nll
                print(f"Saving model parameters at iteration {iteration} with Test NLL: {cv_nll:.4f}")
                # torch.save(current_parameters, PARAMS_FILE_PATH)
                if bias_pred is not None:
                     np.savez(PARAMS_FILE_PATH.replace("_params.pt", "_predictions.npz"),
                              grid=grid.cpu().numpy(),
                              bias_total=bias_pred.cpu().numpy(),
                             )
                     
                # --- CORRECTED SAVING MECHANISM ---
                try:
                    with open(PARAMS_FILE_PATH, 'wb') as f: # Open in binary write mode
                        torch.save(current_parameters, f)
                    print(f"Successfully saved parameters to {PARAMS_FILE_PATH}")
                except Exception as e:
                    print(f"ERROR: Could not save parameters to {PARAMS_FILE_PATH}. Exception: {e}")
                # --- END OF CORRECTION ---

        if iteration % 200 == 0 and iteration > 0: # Fuller log less frequently
             with open(FILE, "a") as outFile:
                outFile.write(f"Iter: {iteration}, TrainNLL: {train_nll_for_print}, TestNLL: {cv_nll}\n")
                for z_name, y_param in current_parameters.items():
                    outFile.write(f"{z_name}: {y_param.detach().cpu().numpy().tolist()}\n")
        
        # Basic convergence / early stopping based on CV loss improvement
        if iteration > 500 and len(cross_valid_losses_by_epoch) > 10:
            if cv_nll > np.mean(cross_valid_losses_by_epoch[-11:-1]) - 1e-5: # If no improvement
                print(f"Stopping early at iteration {iteration} due to lack of CV improvement.")
                break


    print("Training complete.")
    np.savez(LOSSES_FILE_PATH,
             train_losses=np.array(losses_by_epoch),
             test_losses=np.array(cross_valid_losses_by_epoch))

if __name__ == '__main__':
    # Define globals that will be initialized in main()
    OPTIMIZER_VERBOSE, P, FOLD_HERE, REG_WEIGHT, GRID, SHOW_PLOT, DEVICE = None, None, None, None, None, None, None
    FILE, MIN_GRID, MAX_GRID, grid, grid_indices_here, xValues, stimulus_, responses_ = None, None, None, None, None, None, None, None
    N_FOLDS, Fold, subject_ids, block_ids, observations_x, observations_y = None, None, None, None, None, None
    init_parameters, learning_rate, optim = None, None, None
    SENSORY_SPACE_VOLUME_NORMALIZED = None
    PARAMS_FILE_PATH, LOSSES_FILE_PATH = None, None
    BLOCK_ID_VAL = None # Define it globally to be accessible by computeBias

    main() # Call the main function that sets up these globals and starts training