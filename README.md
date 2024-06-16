# MolOpt: Autonomous Molecular Geometry Optimization Using Multiagent Reinforcement Learning

<p align="center" width="100%">
    <img width="45%" src="https://www.researchgate.net/publication/373371947/figure/fig1/AS:11431281183664381@1692966932840/The-figure-shows-the-workflow-of-MolOpt-model-The-molecules-structure-is-in-Cartesian.ppm">
    <img width="45%" src="https://pubs.acs.org/cms/10.1021/acs.jpcb.3c04771/asset/images/medium/jp3c04771_0006.gif">
</p>

[**MolOpt**](https://pubs.acs.org/doi/10.1021/acs.jpcb.3c04771) that uses multiagent reinforcement learning (MARL) for autonomous molecular geometry optimization (MGO). Typically MGO algorithms are hand-designed, but MolOpt uses MARL to learn a learned optimizer (policy) that can perform MGO without the need for other hand-designed optimizers. We cast MGO as a MARL problem, where each agent corresponds to a single atom in the molecule. MolOpt performs MGO by minimizing the forces on each atom of the molecule. Our experiments demonstrate the generalizing ability of MolOpt for the MGO of propane, pentane, heptane, hexane, and octane when trained on ethane, butane, and isobutane. In terms of performance, MolOpt outperforms the MDMin optimizer and demonstrates performance similar to that of the FIRE optimizer. However, it does not surpass the BFGS optimizer. The results demonstrate that MolOpt has the potential to introduce innovative advancements in MGO by providing a novel approach using reinforcement learning (RL), which may open up new research directions for MGO. Overall, this work serves as a proof-of-concept for the potential of MARL in MGO.

This repository allows to train and evaluate Molopt for MGO directly in Cartesian coordinates.

**MolOpt: Autonomous Molecular Geometry Optimization Using Multiagent Reinforcement Learning**<br>
Rohit Modee, Sarvesh Mehta, Siddhartha Laghuvarapu, U. Deva Priyakumar<br>
*Journal of Physical Chemistry B*, 2023.<br>
https://pubs.acs.org/doi/10.1021/acs.jpcb.3c04771

## Setup

Dependencies:
* python=3.9
* rllib=2.0
* pytorch
* torchaudio==0.9.0
* cudatoolkit=10.2
* torchvision
* pandas
* scikit-learn
* ase
* gpaw

## The .yml files for conda environment setup

Install required packages and library:
```
conda env create -f rlopt_with_history.yml
```
OR
```
conda env create -f rlopt_without_conda_history.yml
```

## Usage

You can use this code to train and evaluate MolOpt model for MGO of alkanes.
*dataset* directory contains file named "c2c4_smiles_filtered_frames.xyz" for training the model and "c3c5c6c7c8_smiles_alkanes.xyz" for evaluation of the model.
*other_optimizers* directory contains optimization data from other optimizers such as FIRE and MDMin. This is used to compare MolOpt with FIRE and MDMin to generate plots.
*rl_traj* and *bfgs_traj* directories contain optimization data from MolOpt and BFGS respectively.

### Training
To train the MolOpt model, run the following
```shell
python3 run.py
```

### Evaluation

To perform MGO, run the following command:
```shell
python3 eval.py
```
plots are generated during evaluation. 

You can visualize the structures in XYZ file using, for example, [ASE-GUI](https://wiki.fysik.dtu.dk/ase/ase/gui/gui.html#index-0).

## Citation

If you use this code, please cite our papers:
```txt
@article{doi:10.1021/acs.jpcb.3c04771,
author = {Modee, Rohit and Mehta, Sarvesh and Laghuvarapu, Siddhartha and Priyakumar, U. Deva},
title = {MolOpt: Autonomous Molecular Geometry Optimization Using Multiagent Reinforcement Learning},
journal = {The Journal of Physical Chemistry B},
volume = {127},
number = {48},
pages = {10295-10303},
year = {2023},
doi = {10.1021/acs.jpcb.3c04771},
    note ={PMID: 38013420},
URL = {https://doi.org/10.1021/acs.jpcb.3c04771},
eprint = {https://doi.org/10.1021/acs.jpcb.3c04771}
}
```