# Membrane Segmentation with DMT Loss

### Overview
This project performs membrane segmentation (FCN) regularized by topological loss with critical structure extracted from Discrete Morse Theory (DMT). It aims to reproduce the work of Hu et al. (2021) and applies the method on plant & MIBI-ToF images. 

### Running
Please first create directories for input & output<br>
```mkdir data & mkdir results```<br>

Running the DMT pipeline for training:<br>
```./run.sh [loss] # options: (1). bce; (2). dmt```

### Command-line Options
```shell
usage: train.py [-h] -i ROOT_PATH --option OPTION [-o OUT_PATH] [-d DIST] [-b BATCH_SIZE] [-l LOSS] [-n N_EPOCHS] [-m MODEL_PATH] [-r LR] [-p PATIENCE_COUNTER] [--early-stop] [--region-option]

Unet training options

required arguments:
  -i ROOT_PATH         Root directory of input image datasets for training/testing
  --option OPTION      Training option: (1). binary, (2). multi

optional arguments:
  -o OUT_PATH          Directory to output file
  -d DIST              Distance function for weighted loss
                        Options: (1).dist; (2).saw; (3).class; (4).boundary
  -b BATCH_SIZE        Batch size
  -l LOSS              Loss function
                         Options: (1).bce; (2).dmt
  -n N_EPOCHS          Total number of epoches for training
  -m MODEL_PATH        Saved model file
  -r LR                Learning rate
  -p PATIENCE_COUNTER  Patience counter for early-stopping or lr-tuning
  -t THLD, --thld THLD Persistent Homology threshold for pruning DMT critical structure
  --early-stop         Whether to perform early-stopping; If False, lr is halved when reaching each patience
```

### Directories
```
.
 EDA.ipynb:         Exploratory analysis + visualization
 net.py:            FCN architecture
 dataset.py:        io
 train.py:          Training; command-line interface
 utils:             Utility & self-defined loss functions
├── dmt:            Modified (automated) code from https://github.com/HuXiaoling/DMT_loss
├── docs:           Html "screenshot" of the analysis notebooks
├── data:           Dataset 
├── results:        Training log & output files
```

### References
Hu, X., Wang, Y., Fuxin, L., Samaras, D., & Chen, C. (2020, September). Topology-Aware Segmentation Using Discrete Morse Theory. In International Conference on Learning Representations.
