# GEMENAI
A python package and collection of scripts for computing protein surface meshes, chemical, electrostatic, geometric and MSA features, and  building/training graph neural network models of protein-nucleic acid binding.
![overview](docs/overview.png)

## Installation
To run our code, it is recommended to create a new anaconda enviroment and install this package and required dependencies to the fresh environment.
```
conda create -n gemenai python=3.9
conda activate gemenai
```
The main dependencies for training and running models are pytorch and torch-geometric. You may try installing the latest versions, but the following should work:

```
# Choose one of the following that is most appropriate for your system

# CUDA 10.2
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
# CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
# CPU Only
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
```

```
conda install pyg -c pyg
```
Finally, to install the code needed to run our scripts, clone this repository to a local directory and install:

```
gh repo clone jaredsagendorf/gemenai
cd gemenai
pip install -e .
```

## Soft Requirements
Required for using our mesh and structure feature generation pipeline
- Biopython (1.76)+
- trimesh 
- NanoShaper available on system path
- MSMS available on system path
- EDTSurf available on system path (optional)
- APBS available on system path (optional)
- PDB2PQR available on system path (optional)
