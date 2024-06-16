import os, pdb, matplotlib, tempfile, sys
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from sklearn.metrics import mean_squared_error
import inspect
from pathlib import Path

import gym, ray, natsort
from gym import spaces
from scipy.spatial import distance
from ase import Atoms
from gpaw import GPAW, PW, FD
from ase.optimize import QuasiNewton, BFGS
from ase.io.trajectory import Trajectory
from ase.io import read, write
from ase.build import minimize_rotation_and_translation

import torch
import torchani

from ray import tune
from typing import Dict
from ray.tune.logger import pretty_print
from ray.tune.logger import Logger, UnifiedLogger
from ray.rllib.utils.annotations import override
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.schedules import ConstantSchedule, LinearSchedule, ExponentialSchedule, PiecewiseSchedule

from eval_utils import do_optim, read_traj, calc_rmsd

model_restore = "model/PPO_MA_env_2022-06-26_17-36-29ogu5moul/"

sys.path.insert(1, model_restore)
import MultiAgentEnv as ma_env
from policy import PolicyNetwork

# print(inspect.getfile(ma_env))
model_A = PolicyNetwork
ModelCatalog.register_custom_model("modelA", model_A)

# define action space and observation space
# action space is step the policy takes in angstrom
# observation space are the coordinates of the single atom
act_space = spaces.Box(low=-0.05,high=0.05, shape=(3,))
obs_space = spaces.Box(low=-1000,high=1000, shape=(128+5+3,))

def gen_policy(atom):
    model = "model{}".format(atom)
    config = {"model": {"custom_model": model,},}
    return (None, obs_space, act_space, config)


policies = {"policy_A": gen_policy("A")}
policy_ids = list(policies.keys())

def policy_mapping_fn(agent_id, **kwargs):
    pol_id = "policy_A"
    return pol_id

def env_creator(env_config):
    return ma_env.MA_env(env_config)  # return an env instance

register_env("MA_env", env_creator)

config = ppo.DEFAULT_CONFIG.copy()

config["multiagent"] = {
        "policy_mapping_fn": policy_mapping_fn,
        "policies": policies,
        "policies_to_train": ["policy_A"],#, "policy_N", "policy_O", "policy_H"],
    }

config["in_evaluation"] = True
config["explore"] = False
config["log_level"] = "WARN"
config["framework"] = "torch"
# config["num_gpus"] =  int(os.environ.get("RLLIB_NUM_GPUS", "0"))
config["num_gpus"] =  1
config["num_workers"] = 1
config["env_config"] =  {"atoms":["C", "H"]}
config["rollout_fragment_length"] = 200
config["vf_share_layers"] = True


def custom_log_creator(custom_path, custom_str):
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)
    def logger_creator(config):
        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)
    return logger_creator

def get_atomization_energy(Ener, S):
    self_energies = {'H':-0.500607632585, 'C':-37.8302333826,'N':-54.5680045287,'O':-75.0362229210}
    self_interaction_energies = 0
    for specie in S:
        self_interaction_energies += self_energies[specie]
    Ener -= torchani.units.hartree2ev(self_interaction_energies)
    Ener = torchani.units.ev2kcalmol(Ener)
    # Ener = Ener*627.51
    return Ener


ray.init()

class setup_optimization():
    def __init__(self, root_dir, testset_name, checkpoint_num, all_time):
        working_dir = os.getcwd()
        self.all_time = all_time
        self.root_dir = root_dir
        self.testset_name = testset_name
        self.model_dir = model_dir
        self.bfgs_traj_dir = f"bfgs_traj/{self.testset_name}/"
        self.mdmin_traj_dir = f"./other_optimizers/mdmin/{self.testset_name}/"
        self.fire_traj_dir = f"./other_optimizers/fire/{self.testset_name}/"
        self.rl_traj_dir = f"{working_dir}/rl_traj/{self.testset_name}/{self.model_dir}"
        self.mol_name_dict = {}
        self.plots_dir = f"/{working_dir}/plots/{model_dir}/{testset_name}"

        self.eval_structures_file = self.root_dir + "dataset/c3c5c6c7c8_smiles_alkanes.xyz"

        bfgs_traj_dir_list = natsort.natsorted(os.listdir(self.bfgs_traj_dir))
        for filename in bfgs_traj_dir_list:
            idx = int(filename.split("_")[0])
            self.mol_name_dict[idx] = filename

        Path(self.rl_traj_dir).mkdir(parents=True, exist_ok=True)
        Path(self.plots_dir).mkdir(parents=True, exist_ok=True)

        self.checkpoint_num = checkpoint_num
        self.iteration_num = checkpoint_num
        checkpoint_dir = f"checkpoint_{self.iteration_num:06d}/checkpoint-{self.iteration_num}"
        self.agent = ppo.PPOTrainer(config=config, env="MA_env", logger_creator=custom_log_creator(os.path.expanduser("/home/rohit/ssds/nnp/proj7_RL4Opt/team_rew_param_share/results"), 'custom_dir'))
        self.agent.restore(model_restore + checkpoint_dir)
        self.env = ma_env.MA_env({})

    def get_energy_from_traj(self, optimizer_dir, rl_energy, mol_idx, rl_frame):
        bfgs_traj = read_traj(f"{optimizer_dir}/{self.mol_name_dict[mol_idx]}")

        bfgs_energy = []
        bfgs_forces = []
        for frame in bfgs_traj:
            bfgs_energy.append(get_atomization_energy(frame.get_potential_energy(), frame.get_chemical_symbols()))
            bfgs_forces.append(frame.get_forces())

        bfgs_energy = np.array(bfgs_energy)
        bfgs_rmsd = round(calc_rmsd(bfgs_traj[0], bfgs_traj[-1]), 5)
        final_rmsd = round(calc_rmsd(bfgs_traj[-1], rl_frame), 5)
        return bfgs_traj, bfgs_energy, bfgs_rmsd, final_rmsd


    def load_and_optim(self, mol_idx):
        obs = self.env.reset(mol_idx)

        action = {}
        done = {}
        done["__all__"] = False
        step_sch = True
        start = time.time()

        for i in range(300):
            for agent_id, agent_obs in obs.items():
                policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
                action[agent_id] = self.agent.compute_single_action(agent_obs, policy_id=policy_id)
            obs, rew, done, info = self.env.step(action)
            self.delta_energy = abs(self.env.energies[-2] - self.env.energies[-1])
            if self.env.energies[-1] > self.env.energies[0] and step_sch:
                print("energy up Break triggered")
                break
            if self.delta_energy <= 0.000001:
                print(f"energy diff Break triggered {self.delta_energy}")
                break
        end = time.time()

        rl_energy = torchani.units.ev2kcalmol(np.array(self.env.energies))
        rl_traj = self.env.trajectory
        write(f"{self.rl_traj_dir}/{self.mol_name_dict[mol_idx]}_rl_optim{self.iteration_num}.xyz", self.env.trajectory)

        most_stable_mol_rl_idx = rl_energy.argmin()
        rl_frame = rl_traj[most_stable_mol_rl_idx]
        atoms = read(self.eval_structures_file, index="{}".format(mol_idx)) #includes only CH4, H2O, NH3

        all_traj_dir = [self.mdmin_traj_dir, self.fire_traj_dir, self.bfgs_traj_dir]
        all_traj = []
        all_energy = []
        all_rmsd = []
        all_final_rmsd = []

        for iid in range(3):
            at, ae, ar, af = self.get_energy_from_traj(all_traj_dir[iid], rl_energy, mol_idx, rl_frame)
            all_traj.append(at)
            all_energy.append(ae)
            all_rmsd.append(ar)
            all_final_rmsd.append(af)

        try:
            rl_rmsd = round(calc_rmsd(all_traj[0][0], rl_frame), 5)
        except ValueError:
            pdb.set_trace()

        print(all_rmsd[2], all_final_rmsd[2], rl_rmsd)

        alignment = {'size': 'small', 'horizontalalignment': 'center', 'verticalalignment': 'baseline'}
        plt.title("Energy vs number of steps.")
        plt.xlabel("Steps")
        plt.ylabel("Energy (kcal/mol)")

        plt.plot(all_energy[0], 'o-', label=f"MDMin ({all_energy[0].min().round(decimals=2)} t={len(all_energy[0])})")
        plt.plot(all_energy[1], 'o-', label=f"FIRE ({all_energy[1].min().round(decimals=2)} t={len(all_energy[1])})")
        plt.plot(all_energy[2], 'o-', label=f"BFGS ({all_energy[2].min().round(decimals=2)} t={len(all_energy[2])})")
        plt.plot(rl_energy, 'o-', label=f"RL ({rl_energy[most_stable_mol_rl_idx].round(decimals=2)} t={most_stable_mol_rl_idx})")
        
        plt.axhline(y=all_energy[2][-1], color='r', linestyle='--')
        plt.legend()
        plt.savefig(f"{self.plots_dir}/{self.mol_name_dict[mol_idx]}_{self.iteration_num}.png", bbox_inches='tight', dpi=300)
        print("RMSD Initial structure vs BFSG final structure = {}\nRMSD Initial structure vs RL final structure {}\nRMSD BFGS final structure vs RL final structure {}".format(all_rmsd[2], rl_rmsd, all_final_rmsd[2]))
        print(f"Energy BFGS = {all_energy[2][-1]}")
        self.all_time = self.all_time + (end-start)
        print(f"Time taken = {self.all_time}")
        plt.clf()
        delta_energy = []
        delta_energy_kcal = []
        for iid in range(3):
            delta_energy_kcal.append(all_energy[iid].min() - rl_energy[most_stable_mol_rl_idx])
            delta_energy.append(delta_energy_kcal[iid] * 0.043364)

        data_return = []
        for iid in range(3):
            tmp = [self.checkpoint_num, mol_idx, rl_energy[most_stable_mol_rl_idx], all_energy[iid].min(), all_rmsd[iid], all_final_rmsd[iid], rl_rmsd, delta_energy[iid], delta_energy_kcal[iid], len(atoms), all_energy[iid][0]]
            data_return.append(tmp)
        return data_return

root_dir = "./"
testset_name = "c3c5c6c7c8"

model_dir = model_restore.split("/")[2]
checkpoint_list = [1401]
# checkpoint_list = [91, 101]
all_time = 0

for cpk_idx in checkpoint_list:
    optimizer = setup_optimization(root_dir, testset_name, cpk_idx, all_time)
    alkanes_eval_c3c5c6c7c8 = range(0, 30, 5)
    opt_name = ["MDMin", "FIRE", "BFGS"]

    all_file_handle = []
    for iid in range(3):
        header = f"Epoch,mol_id,RL_energy,{opt_name[iid]}_energy,{opt_name[iid]}_initial_vs_optim_RMSD,{opt_name[iid]}_optim_vs_RL_optim_rmsd,{opt_name[iid]}_initial_vs_RL_optim_RMSD,Delta_Energy_eV,Delta_Energy_kcal,Mol_size,Initial_Energy\n"
        file_handle = open(f"plots/{model_dir}/exp_nonperturbed_checkpoint_{cpk_idx}_{testset_name}_{opt_name[iid]}.csv", "w")
        file_handle.write(header)
        all_file_handle.append(file_handle)

    for molecule_idx in alkanes_eval_c3c5c6c7c8:
        energy_rmsd = optimizer.load_and_optim(mol_idx=molecule_idx)
        for iid in range(3):
            all_file_handle[iid].write(", ".join(map(str,energy_rmsd[iid])) + "\n")
        # all_mol_energy.append(energy_rmsd)
    # fmt = '%d,%d,%f,%f,%f,%f,%f,%f,%f,%d'
    # np.savetxt(f"plots/{model_dir}/exp_checkpoint_{cpk_idx}_{testset_name}_81.csv", all_mol_energy, delimiter=",", fmt = fmt, header=header, comments='')
    # tmp_data = np.array(all_mol_energy)
    # print(f"mean {tmp_data[:,-2].mean():.5f} {tmp_data[:,-2].mean()*23.0605419:.5f}")
    # print(f"std  {tmp_data[:,-2].std():.5f} {tmp_data[:,-2].std()*23.0605419:.5f}")
    for handles in all_file_handle:
        handles.close()