## Introduction
This tutorial shows how to use the script `run_classifier_ensemble.py` to generate predictions on a test set with known ground-truth labels. The directory `mesh_data` contains 10 pre-processed datafiles for the purpose of demonstration. Replace this directory with your own datafiles as desired. 

**Note:** If you are using this script to perform inference on unlabeled data, make sure you run it with the flag `--prediction`. This will output `-1` under the `Y_gt` column. 

We provide pre-trained models for three classification tasks (dna-binding vs non-binding, rna-binding vs non-binding and dna-binding vs rna-binding). Data for these models are found in `pnabind/models/classification`. We will set one of these directories in an environment variable:

```
MODEL_DIR=../../../models/classification/trained_models/dna_vs_rna
```

The file `config.json` stores all the information needed to load the pre-trained models. If you have trained your own models, these configuration options should be the same as the model you trained. We simply pass this configuraiton file along with a list of model state dictionaries (`fold1-5.tar`) and pickled scaler objects to center/scale the input mesh data (`scaler1-5.pkl`). 

```
python ../../../scripts/run_classifier_ensemble.py datafiles.txt datafiles -c config.json --checkpoint_files $MODEL_DIR/*.tar --scalers $MODEL_DIR/*.pkl
```

The script will generate output files with predictions for each input mesh data file. For example, using the provided datafiles, the file `predictions.csv` will be contain:

```
sequence_id,Y_gt,Y_pr,P
C5CC51,1,1,0.621
O54962,0,0,0.447
P0A7M2,1,1,0.635
P29538,0,0,0.430
P32104,0,1,0.531
P54988,0,0,0.427
Q5UX22,1,1,0.523
Q81RU6,0,0,0.428
Q9FKQ6,0,0,0.416
Q9UTQ0,1,0,0.473
```

The column `Y_pr` shows the predicted class label. On this dataset, the ensemble model achieves 80% accuracy. 
