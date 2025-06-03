# Set up

## Create environment

```
conda create -n perceptual_biases python=3.7
conda activate perceptual_biases
```

1. Enter your root folder:

```
F:
cd F:/Courses2025/认知建模基础/
```

2. Clone the git repo for Wei's Model:

```
conda install git
git clone https://gitlab.com/m-hahn/unifying-theory-biases.git 

```

This copies the git repo in the current folder

3. Install required packages

```
cd ./unifying-theory-biases/code
pip install -r requirements.txt
pip install torch==1.11.0
```

> 如果报错，尝试关掉VPN

## Clone the git repo for Stocker's data

```
cd ../.. # enter the root folder
git clone https://github.com/cpc-lab-stocker/ASD_Encoding_2020.git
```

The data is now stored in `./ASD_Encoding_2020/`

1. Download the Noel folder to `./unifying-theory-biases/code/`

# Run Files

## A quick example

```
cd ./unifying-theory-biases/code/Noel
python run_noel_analysis.py "F:/Courses2025/认知建模基础/ASD_Encoding_2020/" 0 1 2 2 0.1 60 cpu # replace with your root path
```

You should then get two folders in the `Noel` folder after running the file: `losses_G1_B1_P2` and `params_G1_B1_P2`

Then plot the results by

```
python plot_results.py
```

## Parameters

```
python run_noel_analysis.py <DATA_ROOT_PATH> <GROUP_ID> <BLOCK_ID> <P_VAL> <FOLD_HERE> <REG_WEIGHT> <GRID_SIZE> <DEVICE>
```

Command-Line Parameters Explained:

- **<DATA_ROOT_PATH> (String)**: Path to the root of the Noel et al. dataset (e.g., "Data/ASD_Encoding_2020" if your Data folder is in the same directory as the script).
- **GROUP_ID** (Integer):
  - 0: For Neurotypical/Typically Developing (TD) group.
  - 1: For Autism Spectrum Disorder (ASD) group.
- **BLOCK_ID** (Integer):
  - 0: woFB block (without feedback).
  - 1: wFB1 block (first block with feedback).
  - 2: wFB2 block (second block with feedback).
- **P_VAL** (Integer):
  - Exponent p of the L&lt;sup>p&lt;/sup> loss function.
  - 0: Uses MAP estimation (mapCircularEstimator.py).
  - 2: Posterior mean (L&lt;sup>2&lt;/sup> loss).
  - 8: Recommended by Hahn & Wei (2024) for orientation data.
    Uses cosineEstimator.py for P > 0.
- **FOLD_HERE** (Integer):
  - Cross-validation test fold index (e.g., 0 to 9 for 10-fold CV).
- **REG_WEIGHT** (Float):
  - Regularization weight for smoothness of prior and - - encoding. Requires tuning (e.g., try 0.001, 0.01, 0.1).
- **GRID_SIZE** (Integer):
  - Number of points for discretizing the 0-360 degree orientation space.
- **DEVICE** (String):
  - 'cpu' or 'cuda'.

## Other settings
```
python run_noel_uniform_prior.py "F:/Courses2025/认知建模基础/ASD_Encoding_2020/" 0 1 8 2 0.1 30 cpu
python run_noel_uniform_encoding.py "F:/Courses2025/认知建模基础/ASD_Encoding_2020/" 0 1 8 2 0.1 60 cpu
python run_noel_natural_prior.py "F:/Courses2025/认知建模基础/ASD_Encoding_2020/" 1 2 8 2 1. 90 cpu
```