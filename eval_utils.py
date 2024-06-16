import gym
from gym import spaces
from scipy.spatial import distance
from ase import Atoms
from gpaw import GPAW, PW, FD
from ase.optimize import QuasiNewton, BFGS, LBFGS
from ase.io.trajectory import Trajectory
from ase.io import read, write
from ase.build import minimize_rotation_and_translation
from sklearn.metrics import mean_squared_error
import torch
import torchani


def do_optim(system, traj_file):
    system.set_cell((20.0, 20.0, 20.0))
    system.center(vacuum=3.0)
    # calc = GPAW(mode='lcao', basis='szp(dzp)', maxiter=300)
    # calc = GPAW(xc="PBE", mode=FD(nn=4))
    # calc = GPAW(xc="LDA", mode=PW(500), maxiter=300, txt=None)
    calc = torchani.models.ANI1ccx(periodic_table_index=True).ase()

    system.set_calculator(calc)
    # print(f"{system.get_potential_energy()} eV")
    # print(system.get_forces())
    # print(system.get_all_distances())

    dyn = BFGS(system, trajectory=traj_file)
    smvar = dyn.run(fmax=0.0005) # hartree
    # smvar = dyn.run(fmax=0.05) # eV
    print(f"{system.get_potential_energy()} eV")
    print(torchani.units.hartree2ev(system.get_forces()))
    # print(system.get_all_distances())
    # print(smvar)

def read_traj(in_traj):
    traj = Trajectory(in_traj)
    return traj

def calc_rmsd(stru1, stru2):
    minimize_rotation_and_translation(stru1, stru2)
    return mean_squared_error(stru1.positions, stru2.positions, squared=False)
