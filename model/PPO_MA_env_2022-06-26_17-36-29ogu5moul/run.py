import gym, ray
from gym import spaces
import numpy as np
from scipy.spatial import distance
import pdb
import MultiAgentEnv as ma_env

from policy import PolicyNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray import tune
import ray.rllib.agents.ppo as ppo
import os
from ray.tune.logger import pretty_print
from ray.tune.logger import Logger

from typing import Dict
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.schedules import ConstantSchedule, LinearSchedule, ExponentialSchedule, PiecewiseSchedule

from datetime import datetime

LOG_FILE = "/home/rohit.modee/RlOpt/logs/discrete_reward_strict{}.txt".format(datetime.now().strftime("%d_%m_%H_%M"))


class MyCallbacks(DefaultCallbacks):

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        # print(episode.last_info_for("C_1"))
        # # Make sure this episode is really done.
        # print("episode {} (env-idx={}) end.".format(
        #     episode.episode_id, env_index))
        # print("----------------------------------")
        # pdb.set_trace()
        trajectories =  base_env.get_unwrapped()[0].trajectory
        energies =  base_env.get_unwrapped()[0].energies

        # print(trajectories)
        # print(energies)

        with open(LOG_FILE, "a") as outFile:
            for idx,trajectory in enumerate(trajectories):
                outFile.write("episode_id: {} \t env-idx: {} \t pos:{} \t energy:{}\n".format(episode.episode_id, env_index, str(list(trajectory.flatten())),energies[idx]))


# create NN model for each atom type
model_A = PolicyNetwork
ModelCatalog.register_custom_model("modelA", model_A)

# define action space and observation space
# action space is step the policy takes in angstrom
# observation space are the coordinates of the single atom
act_space = spaces.Box(low=-0.05,high=0.05, shape=(3,))
# act_space = spaces.Tuple((spaces.Box(high=0.05, low=-0.05, shape=(3,)), spaces.MultiDiscrete([5,5,5])))
# obs_space = spaces.Box(low=-1000,high=1000, shape=(768+7,))
# act_space = spaces.Box(low=-0.001,high=0.001, shape=(3,))
# obs_space = spaces.Box(low=-1000,high=1000, shape=(256+5,))
obs_space = spaces.Box(low=-1000,high=1000, shape=(128+5+3,))

def gen_policy(atom):
    model = "model{}".format(atom)
    config = {"model": {"custom_model": model,},}
    return (None, obs_space, act_space, config)

# policies = {"policy_C": gen_policy("C"),"policy_N": gen_policy("N"),"policy_O": gen_policy("O"),"policy_H": gen_policy("H")}
policies = {"policy_A": gen_policy("A")}
policy_ids = list(policies.keys())

def policy_mapping_fn(agent_id, episode, **kwargs):
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
        "count_steps_by": "env_steps"
    }

# config["exploration_config"] = {
#         # The Exploration class to use.
#         "type": "EpsilonGreedy",
#         # Config for the Exploration class' constructor:
#         "initial_epsilon": 1.0,
#         "final_epsilon": 0.02,
#         "epsilon_timesteps": 10000,  # Timesteps over which to anneal epsilon.
#     #     "type": "OrnsteinUhlenbeckNoise",
#     #     "random_timesteps": 0,  # timesteps at beginning, over which to act uniformly randomly
#     }

config["entropy_coeff"] = 0.0001
config["kl_coeff"] = 1.0
config["kl_target"] = 0.01
config["gamma"] = 0.90
config["lambda"] = 1.00  #1
config["clip_param"] = 0.3
config["vf_clip_param"] = 10 #10 or 40
config["log_level"] = "INFO"
config["framework"] = "torch"
config["num_gpus"] =  1
# config["num_gpus_per_worker"] = 0.
# config["env_config"] =  {"atoms":["C", "N", "O", "H"]}
config["env_config"] =  {"atoms":["C", "H"]}
config["rollout_fragment_length"] = 200  # train_batch_size / rollout_fragment_length = num_fragments
config["sgd_minibatch_size"] = 512
config["train_batch_size"] = 2048
config["num_workers"] = 36
# config["num_envs_per_worker"] = 1
# config["remote_worker_envs"] = False
config["ignore_worker_failures"] = False
config["horizon"] = 20
config["soft_horizon"] = False
config["batch_mode"] = "truncate_episodes"
config["vf_share_layers"] = True
config["lr"] = 5e-05
# config["lr_schedule"] = {
#         "type": "ExponentialSchedule",
#         "schedule_timesteps": 2048*30,
#         "initial_p": 5e-05,
#         "decay_rate": 0.5,
#         }
# config["vf_loss_coeff"] = 1.00
# config["_fake_gpus"]=True
# config["callbacks"] = MyCallbacks
# config["record_env"] = True
# config["num_gpus"] =  int(os.environ.get("RLLIB_NUM_GPUS", "0"))

print(pretty_print(config))

ray.init()
agent = ppo.PPOTrainer(config, env="MA_env")

# model_restore = "/home/rohit.modee/ray_results/PPO_MA_env_2022-05-25_12-49-2717s7j3br/"
# agent.restore(model_restore + "checkpoint_000031/checkpoint-31")

n_iter = 2800
for n in range(n_iter):
    result = agent.train()
    print(pretty_print(result))

    if n % 5 == 0:
        checkpoint = agent.save()
        print("checkpoint saved at", checkpoint)

checkpoint = agent.save()
print("checkpoint saved at", checkpoint)
