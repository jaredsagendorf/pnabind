# PNAbind
A python package and collection of scripts for computing protein surface meshes, chemical, electrostatic, geometric and MSA features, and  building/training graph neural network models of protein-nucleic acid binding.
![overview](docs/overview.png)

## Installation
To run our code, it is recommended to create a new anaconda enviroment and install this package and required dependencies to the fresh environment.
```
conda create -n pnabind python=3.9
conda activate pndabind
```
The main dependencies for training and running models are pytorch and torch-geometric. You may try installing the latest versions, but the following should work:

```
# Choose one of the following that is most appropriate for your system

# CUDA 10.2
conda install pytorch==1.12.1 cudatoolkit=10.2 -c pytorch
# CUDA 11.3
conda install pytorch==1.12.1 cudatoolkit=11.3 -c pytorch
# CUDA 11.6
conda install pytorch==1.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
# CPU Only (not recommended)
conda install pytorch==1.12.1 cpuonly -c pytorch
```

```
conda install pyg -c pyg
conda install pytorch-scatter -c pyg
```
Finally, to install the code needed to run our scripts, clone this repository to a local directory and install:

```
gh repo clone jaredsagendorf/pnabind
cd pnabind
pip install -e .
```

### Soft Requirements
Required for using our mesh and structure feature generation pipeline
- [Biopython](https://github.com/biopython/biopython) (1.76)+
- [trimesh](https://github.com/mikedh/trimesh) 
- NanoShaper or MSMS or EDTSurf available on system path (provided in `share`)
#### For electrostatic feature calculations
- [APBS](https://apbs.readthedocs.io/en/latest/getting/index.html) or [TABI-PB](https://github.com/Treecodes/TABI-PB) available on system path
- [PDB2PQR](https://pdb2pqr.readthedocs.io/en/latest/getting.html) available on system path
#### For MSA feature calculations
- [PSI-BLAST](https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html) available on system path
- [HHBlits](https://github.com/soedinglab/hh-suite) available on system path
