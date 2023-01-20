# Code for "Bayesian inference in ring attractor networks"

This repo contains the code for simulations and figures in 

Anna Kutschireiter, Melanie A Basnak, Jan Drugowitsch. 2022. Bayesian inference in ring attractor networks. bioRxiv 2021.12.17.473253; doi: https://doi.org/10.1101/2021.12.17.473253


## Code structure and environment setup
* `root` - python scripts and jupyter notebooks containing the main code to reproduce figures
* `/data_processed` - `analysed' artificial data
* `/data_raw` - raw data, empty. 

### Python environment

Our code uses Python 3.9.7. 
Packages we used are listed in the file `environment.yml`.
These can either be installed by hand, or alternatively installed in a virtual environment with name `fltenv` by running
```
conda env create -f environment.yml
```
To activate this virtual environment, run
```
conda activate fltenv
```
before running the code below.


### filtering.py
The file `filtering.py` includes the core implementation of filtering algorithms we used (including network filters).

Artificial data can be generated with the function `generateData()`. 
This will result in a single trajectory of a diffusion on the circule ("ground-truth HD"), as well as increment observations and/or HD observations.

These observations can be fed into the filter implementations, i.e., 
* `vM_Projection_Run()` for the circular Kalman filter,
* `vM_Projection_quad_Run()` for the quadratic approximation of the circular Kalman filter,
* `no_uncertainty_filter_Run()` for an approximation to the circular Kalman filter with constant $\kappa_t$, 
* `network_filter_Run()` for a "network-like" filter (algorithmically the same as the circKF with quadratic approximation, but with parameters in network form)
* `RNN_filter_Run()` for a filter based on a single-population network.
* `PF_run()` for the particle filter, and `GaussADF_run()` for the Gaussian assumed-density filter.

This file also contains several helper functions, including the function `circplot()`, which is useful for plotting a time series of angular data.

### network_filter.py

This file contains a network filter that implements a Drosophila-like multiple network filter, with connectivities as in Figure 4C.

## Reproduce figures
Figures in the manuscript can be reproduced by running the Jupyter notebooks of the same name, i.e., `Figure_2h.ipynb` for Figure 2H etc. 

## Simulations for raw data

`Figure_2i.ipynb`, `Figure_3b,c,e,d_FigureS3.ipynb`, `Figure3f_FigureS4.ipynb`, `Figure4g,h.ipynb` and `FigureS2.ipynb` depend on further data to reproduce the corresponding plots. Unaltered, these notebookes reproduce the plots in the manuscript based on readily packed (artificial) data we provide in the folder `/data_preprocessed` as `.npz' binaries. 

To run the simulations that underlie the raw data for these binaries (i.e., the performance scans), the corresponding python scripts have to be run, as specified in the following for each figure. Running such a simulation will result in an `.npz` archive in the folder `/data-raw`. These simulations have to be repeated for all paramters (or combinations thereof, parameter ranges can be retrieved from the notebooks). To further assess, process and plot this data, set 
```
preprocess = True
```
in in the figure notebooks, and run the corresponding cell for preprocessing in these notebooks.

### Figure 2I & Figure 3B
```
python performance_scan_figure2i.py kappaz_idx kappa_star
```
will simulate 10000 trajectories of the hidden process with angular velocity observations and HD observations with information rate as as indexed by the first parameter (here, it would pick the kappa_idx-th $\kappa_z$ as specified in the script). The second parameter denotes the $\kappa^{\*} $ value that is used in the fixed-undertainty filter (for Figure 3B, we ran this for several values of $\kappa^{\*} $, and then picked the value that achievied best performance when averaged over all observation rates).
For our manuscript we ran this script with information rates $\kappa_z$ and fixed uncertainty $\kappa^{\*} $  ranging from $\kappa_z = 0.01$ to $\kappa_z = 100$, respectively.

### Figure 3D
```
python performance_scan_figure3d.py kappa_star beta
```
will simulate 10000 trajectories of the hidden process with angular velocity observations and HD observations with information rate ranging from $\kappa_z = 0.01$ to $\kappa_z = 100$, and perform filtering with a network filter with dynamic parameters $\kappa^{\*} $ and $\beta$, corresponding to the first and second input parameter, respectively.

### Figure 3F & Figure S4
```
python performance_scan_figure3f.py kappa_star beta sigma_N
```
will simulate 10000 trajectories of the hidden process with angular velocity observations and HD observations with information rate ranging from $\kappa_z = 0.01$ to $\kappa_z = 100$, and perform filtering with a network filter with dynamic parameters $\kappa^{\*} $ and $\beta$, and neural noise $\sigma_N$, corresponding to the first, second and third input parameter, respectively.

### Figure 4G, H
```
python performance_scan_figure4g.py kappaz_idx
```
will simulate 10000 trajectories of the hidden process with angular velocity observations and HD observations with information rate $\kappa_z$ as as indexed by the first parameter.

### Figure S2
```
python performance_scan_figureS2.py N
```
will simulate 10000 trajectories of the hidden process with angular velocity observations and HD observations. Here, the input parameter N denotes the number of neurons in a corresponding network simulation.


