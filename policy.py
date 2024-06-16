import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import SlimFC
import pdb


class PolicyNetwork(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.mlp_f = nn.Sequential(
                        nn.Linear(5+3, 128),
                        nn.Linear(128, 128),
                        nn.Tanh(),
                        nn.Linear(128, 128),
                        nn.Tanh(),
                        nn.Linear(128, 128),
                        )

        self.mlp_aev = nn.Sequential(
                            nn.Linear(128+5+3, 128),
                            nn.Linear(128, 128),
                            nn.LeakyReLU(),
                            nn.Linear(128, 128),
                            nn.LeakyReLU(),
                            )
        
        self.mlp_interaction = nn.Sequential(
                                    nn.Linear(128, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, self.num_outputs),
                                    )

        self.values = nn.Sequential(
                            nn.Linear(128+5+3, 128),
                            nn.Linear(128, 128),
                            nn.Tanh(),
                            nn.Linear(128, 1),
                            )

        self._last_value = None
    
    def forward(self, input_dict, state, seq_lens):
        features = input_dict["obs"]
        force_feature = self.mlp_f(features[:,-8:])

        out = self.mlp_aev(features) + force_feature
        final_out = self.mlp_interaction(out)
        self._last_value = self.values(features)

        return final_out , []

    def value_function(self):
        return torch.squeeze(self._last_value, -1)