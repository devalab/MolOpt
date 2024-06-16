import gym, ray, pdb, sys
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

if "eval" in sys.argv[0]:
    root_dir = "./"
    # methane = [read(root_dir + "data/ani_subset/readers/eval_structures.xyz", index=":")] #includes only CH4, H2O, NH3
    # methane = [read(root_dir + "data/ani_subset/readers/c3c5c6c7c8_smiles_alkanes.xyz", index=":")]
    methane = [read(root_dir + "dataset/c3c5c6c7c8_smiles_alkanes.xyz", index=":")]
    # methane = [read(root_dir + "data/ani_subset/readers/c3c5c6c7c8_smiles_perturbed_alkanes.xyz", index=":")] #includes only CH4, H2O, NH3
    # methane = [read(root_dir + "data/ani_subset/readers/ani_c5plus_lt30_alkanes.xyz", index=":")] #includes only CH4, H2O, NH3
    # methane = [read(root_dir + "data/ani_subset/readers/ani_hdf5_4heavy_lt5_alkanes_eval.xyz", index=":")] #includes only CH4, H2O, NH3
    # methane = [read(root_dir + "data/ani_subset/readers/ani_hdf5_4heavy_lt30_alkanes_eval.xyz", index=":")] #includes only CH4, H2O, NH3
else:
    print("Not In evaluation")
    # methane = read("./data/ani.xyz", index=":")
    # methane = read("./data/ethane_gt30_ani.xyz", index="0")
    # methane = read("./data/ani.xyz", index=":") #includes only CH4, H2O, NH3

## AEV parameters
# Rcr = 5.2000e+00
# Rca = 3.5000e+00
Rcr = 5.1000e+00
Rca = 3.5000e+00

device=torch.device("cpu")

# EtaR = torch.tensor([1.6000000e+01], device=device)
# ShfR = torch.tensor([9.0000000e-01, 1.0193548e+00, 1.1387097e+00, 1.2580645e+00, 1.3774194e+00, 1.4967742e+00, 1.6161290e+00, 1.7354839e+00, 1.8548387e+00, 1.9741935e+00, 2.0935484e+00, 2.2129032e+00, 2.3322581e+00, 2.4516129e+00, 2.5709677e+00, 2.6903226e+00, 2.8096774e+00, 2.9290323e+00, 3.0483871e+00, 3.1677419e+00, 3.2870968e+00, 3.4064516e+00, 3.5258065e+00, 3.6451613e+00, 3.7645161e+00, 3.883871e+00, 4.0032258e+00, 4.1225806e+00, 4.2419355e+00, 4.3612903e+00, 4.4806452e+00, 4.6e+00], device=device)
# Zeta = torch.tensor([3.2000000e+01], device=device)
# ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
# EtaA = torch.tensor([8.0000000e+00], device=device)
# ShfA = torch.tensor([9.0000000e-01, 1.1785714e+00, 1.4571429e+00, 1.7357143e+00, 2.0142857e+00, 2.2928571e+00, 2.5714286e+00, 2.8500000e+00], device=device)

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
    # t_rew = np.clip(1/resultant_forces, a_min=1, a_max=10).mean()
    # t_rew = np.clip(-np.exp(resultant_forces), a_min=-10, a_max=1).mean()
    # t_rew = -np.log(resultant_forces.max())
    t_rew = -np.log(resultant_forces).mean()
    # t_rew = -np.log(resultant_forces.sum()) * 10
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
    # # if atom_symbol == "C":
    # #     custom_rew = custom_rew * 1.5
    # custom_rew = -np.log(spherical_forces[:,0][idx]) + team_reward(spherical_forces[:,0])
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
        # self.action_space = spaces.Tuple((spaces.Box(high=0.05, low=-0.05, shape=(3,)), spaces.MultiDiscrete([5,5,5])))
        # self.observation_space = spaces.Box(low=-1000,high=1000, shape=(768+7,))
        # self.observation_space = spaces.Box(low=-100,high=100, shape=(128+5,))
        self.observation_space = spaces.Box(low=-1000,high=1000, shape=(128+5+3,))
        # self.calc = GPAW(xc="PBE", mode=FD(nn=3), maxiter=300, txt=None)
        # self.calc = GPAW(xc="PBE", mode=PW(500), eigensolver='rmm-diis', maxiter=300, txt=None) #much better
        # self.calc = GPAW(xc="PBE", mode=PW(500), maxiter=300, txt=None) #much better
        # self.calc = GPAW(xc="LDA", mode=PW(500), maxiter=300, txt=None) # much better in convergence and speed
        # much better in convergence and speed test on 3 molecules
        # self.calc = GPAW(xc="LDA", mode=FD(nn=3), eigensolver='rmm-diis', maxiter=300, txt=None)
        self.calc = torchani.models.ANI1ccx(periodic_table_index=True).to("cpu")

    def reset(self, mol_num=None):
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

        # methane = np.random.choice([methane_low_ener, methane_high_ener], p=[0.70, 0.30])
        # system = np.random.choice(methane)
        # system = random.choice(methane)
        if mol_num == None:
            system = random.choice(methane)
        else:
            system = methane[0][mol_num]

        atoms = system.get_chemical_symbols()
        self.all_atoms.append(atoms)

        for atom_id, key in enumerate(atoms):
            self.atom_agent_map.append("A"+"_"+str(atom_id))
            self.agents.append(Atom_Agent())

        system.center(vacuum=3.0)
        # time_now = datetime.now().strftime('%Y%m%d%H:%M:%S')
        # calc = Gaussian(label='calc/gaussian_{gen}', xc='wB97x', basis='6-31g(d)', scf='maxcycle=100')
        # calc = GPAW(mode='lcao', basis='dzp', txt='gpaw.txt')
        system.set_calculator(self.calc.ase())

        # tmp_sp = torch.tensor(system.get_atomic_numbers()).unsqueeze(dim=0)
        # tmp_coor = torch.tensor(system.get_positions(), requires_grad=True).unsqueeze(dim=0)
        # e = self.calc((tmp_sp, tmp_coor.float())).energies
        # derivative = torch.autograd.grad(e.sum(), tmp_coor)[0]
        # unit_forces = -derivative.squeeze()
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
        ## set new molecule from dataset here
        # coordinates = cartesian_to_spherical(methane)
        # return_dict = {self.atom_agent_map[i]: agent.reset(coordinates[i]) for i, agent in enumerate(self.agents)}

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
        # self.all_aevs.append(result.aevs)
        # aev = torch.cat((result.aevs, species_one_hot, torch.tensor(unit_forces).unsqueeze(0), tmp_energy.unsqueeze(0)), 2)
        aev = torch.cat((result.aevs, species_one_hot, torch.tensor(unit_forces).unsqueeze(0), torch.zeros_like(coor)), 2)
        # aev = torch.cat((result.aevs, species_one_hot, torch.tensor(unit_forces).unsqueeze(0)), 2)
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

            # tmp_sp = torch.tensor(atoms.get_atomic_numbers()).unsqueeze(dim=0)
            # tmp_coor = torch.tensor(atoms.get_positions(), requires_grad=True).unsqueeze(dim=0)

            # e = torchani.units.hartree2ev(self.calc((tmp_sp, tmp_coor.float())).energies)
            # derivative = torch.autograd.grad(e.sum(), tmp_coor)[0]
            # f = -derivative.squeeze()
            # e = e.item()
            
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
                    # rew[key] = (self.energies[-1]-self.energies[-2]) * -1
                # 1 Hartree/Bohr = 51.42208619083232 eV/A
                # 2.571103
                # if spherical_forces[idx][0] < 0.1:
                #     self.dones.add(idx)
            # print(f"energies = {e:.5f} fmax = {res_max_force:.5f} {''.join(self.all_atoms[-1])} forces = {[round(spf, 5) for spf in spherical_forces[:,0]]}")#" bonds = {np.array([dist_mat[i] for i in bonds])}")
            print(f"energies = {e:.5f} fmax = {res_max_force:.5f} {''.join(self.all_atoms[-1])}")#" bonds = {np.array([dist_mat[i] for i in bonds])}")
            # for k, v in info.items():
            #     print(f"disp = {k} : {v} {rew[k]}")
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
        # self.all_aevs.append(result.aevs)
        # delta_aev = self.all_aevs[-1] - self.all_aevs[-2]
        
        if self.energies[-1] == "None":
            get_shape = atoms.positions.shape
            dummy_forces = np.ones(get_shape)*700
            delta_forces = -dummy_forces
            # dummy_energy = torch.tensor([1] * get_shape[0]).reshape(-1,1)
            # aev = torch.cat((result.aevs, species_one_hot, torch.from_numpy(dummy_forces).unsqueeze(0), dummy_energy.unsqueeze(0)), 2)
            aev = torch.cat((result.aevs, species_one_hot, torch.from_numpy(dummy_forces).unsqueeze(0)), 2)
            # aev = torch.cat((delta_aev, species_one_hot, torch.from_numpy(dummy_forces).unsqueeze(0)), 2)
        else:
            # tmp_energy = torch.tensor([self.energies[0]-self.energies[-1]]*len(spherical_forces)).reshape(-1, 1)
            # aev = torch.cat((result.aevs, species_one_hot, torch.tensor(f).unsqueeze(0), tmp_energy.unsqueeze(0)), 2)
            delta_forces = self.all_forces[-2]-self.all_forces[-1]
            f = np.c_[f, delta_forces]
            aev = torch.cat((result.aevs, species_one_hot, torch.tensor(f).unsqueeze(0)), 2)
            # aev = torch.cat((delta_aev, species_one_hot, torch.tensor(f).unsqueeze(0)), 2)
        obs = {self.atom_agent_map[i]: np.array(aev[0,i]) for i, agent in enumerate(self.agents)}

        ## convert obs to feature vector here

        ## calculate atom wise reward here
        # done["__all__"] = len(self.dones) == len(self.agents)
        done["__all__"] = terminate
        return obs, rew, done, info


class Atom_Agent(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-0.05,high=0.05, shape=(3,))
        # self.action_space = spaces.Tuple((spaces.Box(high=0.05, low=-0.05, shape=(3,)), spaces.MultiDiscrete([5,5,5])))
        # self.observation_space = spaces.Box(low=-1000,high=1000, shape=(768+7,))
        self.observation_space = spaces.Box(low=-1000,high=1000, shape=(128+5+3,))
        # self.observation_space = spaces.Box(low=-100,high=100, shape=(128+5,))

    def reset(self,feature,coordinates):
        self.feature = np.array(feature)
        self.coordinates = coordinates

        ## return features instead of coordintaes
        return self.feature
    
    def step(self, action):
        # tmp_act = action[0] * np.power(0.1, action[1])
        # self.coordinates = self.coordinates + tmp_act
        self.coordinates = self.coordinates + action
        # return self.coordinates, None, False, {tuple(np.round(action, decimals=8))}
        # return self.coordinates, None, False, {tuple(np.round(tmp_act, decimals=8))}
        return self.coordinates, None, False, {}
