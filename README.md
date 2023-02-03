# GeoBind
A python package and scripts for building and training graph neural network models of protein-nucleic acid binding.
![overview](docs/overview.png)

## Installation
- clone this repository to a local directory
- cd into that directory and run 'pip install -e .'
- install the required dependencies

## Requirements
Required for running and training models
- python (3.6+)
- pytorch (1.6)+
- pytorch-geometric (1.6.1)+
- sklearn
- numpy
- scipy

Required for using our mesh and mesh feature generation pipeline
- Biopython (1.76)+
- trimesh 
- NanoShaper available on system path
- MSMS available on system path
- EDTSurf available on system path (optional)
- APBS available on system path (optional)
- PDB2PQR available on system path (optional)
