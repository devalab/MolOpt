import gym, ray, pdb
from gym import spaces
import numpy as np
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances

from ase import Atoms
from ase.io import Trajectory, read
# from ase.calculators.emt import EMT
# from ase.calculators.gaussian import Gaussian
from gpaw import GPAW, PW, FD
import torch
import torchani
import warnings
import string
from datetime import datetime

#used for creating a "unique" id for a run (almost impossible to generate the same twice)
def id_generator(size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

gen = id_generator()

warnings.simplefilter(action='ignore')

methane_low_ener = read("dataset/c2c4_smiles_filtered_frames.xyz", index=":") #Training
# methane_low_ener = read("dataset/c3c5c6c7c8_smiles_alkanes.xyz", index=":") #Evaluation

## AEV parameters
Rcr = 5.1000e+00
Rca = 3.5000e+00

device=torch.device("cpu")

EtaR = torch.tensor([1.9700000e+01], device=device)
ShfR = torch.tensor([8.0000000e-01,1.0687500e+00,1.3375000e+00,1.6062500e+00,1.8750000e+00,2.1437500e+00,2.4125000e+00,2.6812500e+00,2.9500000e+00,3.2187500e+00,3.4875000e+00,3.7562500e+00,4.0250000e+00,4.2937500e+00,4.5625000e+00,4.8312500e+00], device=device)
Zeta = torch.tensor([1.4100000e+01], device=device)
ShfZ = torch.tensor([3.9269908e-01,1.1780972e+00,1.9634954e+00,2.7488936e+00], device=device)
EtaA = torch.tensor([1.2500000e+01], device=device)
ShfA = torch.tensor([8.0000000e-01,1.1375000e+00,1.4750000e+00,1.8125000e+00,2.1500000e+00,2.4875000e+00,2.8250000e+00,3.1625000e+00], device=device)

num_species = 2

aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
species_to_tensor = torchani.utils.ChemicalSymbolsToInts('HC')
# species_to_tensor = torchani.utils.ChemicalSymbolsToInts('HCNO')


def cartesian_to_spherical(pos: np.ndarray) -> np.ndarray:
    theta_phi = np.empty(shape=pos.shape[:-1] + (3, ))

    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    r = np.linalg.norm(pos, axis=-1)
    theta_phi[..., 0] = r
    theta_phi[..., 1] = np.arccos(z / r)  # theta
    theta_phi[..., 2] = np.arctan2(y, x)  # phi
    return theta_phi

def spherical_to_cartesian(theta_phi: np.ndarray) -> np.ndarray:
    r, theta, phi = theta_phi[..., 0], theta_phi[..., 1], theta_phi[..., 2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)

def team_reward(resultant_forces):
    t_rew = -np.log(resultant_forces).mean()
    return t_rew

def cont_atomic_plus_team_reward(spherical_forces, idx, atom_symbol):
    if spherical_forces[:,0][idx] >= 10:
        atomic_rew = -1.0
        custom_rew = atomic_rew + team_reward(spherical_forces[:,0])
    elif spherical_forces[:,0][idx] < 10 and spherical_forces[:,0][idx] >= 1.0:
        atomic_rew = 0.0
        custom_rew = atomic_rew + team_reward(spherical_forces[:,0])
    elif spherical_forces[:,0][idx] < 1.0 and spherical_forces[:,0][idx] >= 0.1:
        atomic_rew = 1.0
        custom_rew = atomic_rew + team_reward(spherical_forces[:,0])
    elif spherical_forces[:,0][idx] < 0.1 and spherical_forces[:,0][idx] >= 0.01:
        atomic_rew = 20.0
        custom_rew = atomic_rew + team_reward(spherical_forces[:,0])
    elif spherical_forces[:,0][idx] < 0.01 and spherical_forces[:,0][idx] >= 0.001:
        atomic_rew = 500.0
        custom_rew = atomic_rew + team_reward(spherical_forces[:,0])
    elif spherical_forces[:,0][idx] < 0.001:
        atomic_rew = 2000.0
        custom_rew = atomic_rew + team_reward(spherical_forces[:,0])
    return custom_rew

def get_atomization_energy(Ener, S):
    self_energies = {'H':-0.500607632585, 'C':-37.8302333826,'N':-54.5680045287,'O':-75.0362229210}
    self_interaction_energies = 0
    for specie in S:
        self_interaction_energies += self_energies[specie]
    Ener -= torchani.units.hartree2ev(self_interaction_energies)
    # Ener = Ener*627.51
    return Ener

class MA_env(MultiAgentEnv):
    def __init__(self, env_config):
        # pick actual env based on worker and env indexes
        self.action_space = spaces.Box(low=-0.05,high=0.05, shape=(3,))
        self.observation_space = spaces.Box(low=-1000,high=1000, shape=(128+5+3,))
        self.calc = torchani.models.ANI1ccx(periodic_table_index=True).to("cpu")

    def reset(self):
        self.all_atoms = []
        self.trajectory = []
        self.energies = []
        self.all_forces = []
        self.agents = []
        # self.atom_count = {}
        self.atom_agent_map = []
        # self.all_aevs = []

        print("\nReset called")
        self.dones = set()

        # system = np.random.choice(methane)
        system = np.random.choice(methane_low_ener)

        atoms = system.get_chemical_symbols()
        self.all_atoms.append(atoms)

        for atom_id, key in enumerate(atoms):
            self.atom_agent_map.append("A"+"_"+str(atom_id))
            self.agents.append(Atom_Agent())

        system.center(vacuum=3.0)
        system.set_calculator(self.calc.ase())

        e = system.get_potential_energy()
        unit_forces = system.get_forces()
        e = get_atomization_energy(e, atoms)
        spherical_forces = cartesian_to_spherical(unit_forces)
        # e = e.item()
        self.energies.append(e)
        self.all_forces.append(unit_forces)
        self.trajectory.append(system)
        res_max_force = np.max(spherical_forces[:,0])
        # print(f"energies = {e:.5f} fmax = {res_max_force:.5f} {''.join(atoms)} forces = {[round(spf , 5) for spf in spherical_forces[:,0]]}")#" bonds = {np.array([dist_mat[i] for i in bonds])}")
        print(f"energies = {e:.5f} fmax = {res_max_force:.5f} {''.join(atoms)} disp = {[round(spf , 5) for spf in spherical_forces[:,0]]}")#" bonds = {np.array([dist_mat[i] for i in bonds])}")

        """feature vector.
        input can be a tuple of two tensors: species, coordinates.
        species must have shape ``(C, A)``, coordinates must have shape
        ``(C, A, 3)`` where ``C`` is the number of molecules in a chunk,
        and ``A`` is the number of atoms."""
        # tmp_energy = torch.tensor([self.energies[0]-self.energies[-1]]*len(spherical_forces)).reshape(-1, 1)
        species = species_to_tensor(atoms).unsqueeze(dim=0)
        coor = torch.FloatTensor(system.get_positions()).unsqueeze(dim=0)
        species_one_hot = torch.nn.functional.one_hot(species, num_classes=2)
        result = aev_computer((species.to(device), coor.to(device)))
        aev = torch.cat((result.aevs, species_one_hot, torch.tensor(unit_forces).unsqueeze(0), torch.zeros_like(coor)), 2)
        return_dict = {self.atom_agent_map[i]: agent.reset(aev[0,i], system.get_positions()[i]) for i, agent in enumerate(self.agents)}
        return return_dict

    def step(self, action):
        obs, rew, done, info = {}, {}, {}, {}
        self.dones = set()
        for val in self.atom_agent_map:
            obs[val] = None
            rew[val] = None
            done[val] = None
            info[val] = None
        
        for idx, val in action.items():            
            res = self.agents[self.atom_agent_map.index(idx)].step(val)
            obs[idx], rew[idx], done[idx], info[idx] = res
            # print(f"{idx}:- {obs[idx]} {val}")

        final_coordinates = np.array([obs[key] for key in self.atom_agent_map])
        len_final_coor = len(final_coordinates)
        if len(self.all_atoms[-1]) != len_final_coor:
            print("Error species and coordinates do NOT match")
        atoms = Atoms(self.all_atoms[-1], positions=final_coordinates)
        self.trajectory.append(atoms)
        atoms.center(vacuum=3.0)
        terminate = False
        try:
            atoms.set_calculator(self.calc.ase())
            f = atoms.get_forces()
            e = atoms.get_potential_energy()
            e = get_atomization_energy(e, atoms.get_chemical_symbols())
            
            max_force = np.max(np.abs(f))
            self.energies.append(e)
            self.all_forces.append(f)
            spherical_forces = cartesian_to_spherical(f)
            res_max_force = np.max(spherical_forces[:,0])

            if max_force < 0.001:
                print("Converged")
                terminate = True
                for idx, key in enumerate(self.atom_agent_map):
                    rew[key] = 5000.0

            for idx, key in enumerate(self.atom_agent_map):
                    rew[key] = cont_atomic_plus_team_reward(spherical_forces, idx, atoms.get_chemical_symbols()[idx])
            print(f"energies = {e:.5f} fmax = {res_max_force:.5f} {''.join(self.all_atoms[-1])}")#" bonds = {np.array([dist_mat[i] for i in bonds])}")
        except:
            print("Gaussian Converge error")
            terminate = True
            for idx, key in enumerate(self.atom_agent_map):
                rew[key] = -1.0
                # self.dones.add(idx)
            self.energies.append("None")

        ######## calculate aev (feature vector) from final_coordinates after step has been taken
        ######## create observation dict of all aevs
        species = species_to_tensor(atoms.get_chemical_symbols()).unsqueeze(dim=0)
        coor = torch.FloatTensor(final_coordinates).unsqueeze(dim=0)
        species_one_hot = torch.nn.functional.one_hot(species, num_classes=2)
        result = aev_computer((species.to(device), coor.to(device)))
        if self.energies[-1] == "None":
            get_shape = atoms.positions.shape
            dummy_forces = np.ones(get_shape)*700
            delta_forces = -dummy_forces
            aev = torch.cat((result.aevs, species_one_hot, torch.from_numpy(dummy_forces).unsqueeze(0)), 2)
        else:
            delta_forces = self.all_forces[-2]-self.all_forces[-1]
            f = np.c_[f, delta_forces]
            aev = torch.cat((result.aevs, species_one_hot, torch.tensor(f).unsqueeze(0)), 2)
        obs = {self.atom_agent_map[i]: np.array(aev[0,i]) for i, agent in enumerate(self.agents)}

        ## convert obs to feature vector here
        ## calculate atom wise reward here
        # done["__all__"] = len(self.dones) == len(self.agents)
        done["__all__"] = terminate
        return obs, rew, done, info


class Atom_Agent(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-0.05,high=0.05, shape=(3,))
        self.observation_space = spaces.Box(low=-1000,high=1000, shape=(128+5+3,))

    def reset(self,feature,coordinates):
        self.feature = np.array(feature)
        self.coordinates = coordinates

        ## return features instead of coordintaes
        return self.feature
    
    def step(self, action):
        self.coordinates = self.coordinates + action
        return self.coordinates, None, False, {}