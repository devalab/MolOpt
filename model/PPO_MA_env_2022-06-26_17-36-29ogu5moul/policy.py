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
                        # nn.LeakyReLU(),
                        nn.Tanh(),
                        nn.Linear(128, 128),
                        # nn.LeakyReLU(),
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
        # self.layer_1 = nn.Linear(256+5, 256)
        # self.layer_2 = nn.Linear(256, 128)
        # self.layer_3 = nn.Linear(128, 128)
        # self.layer_4 = nn.Linear(128, self.num_outputs)

        # self.layer_1 = SlimFC(
        #     in_size=256+5, out_size=256, activation_fn="linear")
        # self.layer_2 = SlimFC(
        #     in_size=256, out_size=128, activation_fn="relu")
        # self.layer_3 = SlimFC(
        #     in_size=128, out_size=128, activation_fn="relu")
        # self.layer_4 = SlimFC(
        #     in_size=128, out_size=self.num_outputs, activation_fn="linear")
        
        # self.values = SlimFC(in_size=256+5, out_size=1, activation_fn="linear")
        self.values = nn.Sequential(
                            nn.Linear(128+5+3, 128),
                            nn.Linear(128, 128),
                            # nn.LeakyReLU(),
                            nn.Tanh(),
                            nn.Linear(128, 1),
                            # nn.LeakyReLU(),
                            # nn.Tanh(),
                            # nn.Linear(128, 128),
                            )

        self._last_value = None
    
    def forward(self, input_dict, state, seq_lens):
        features = input_dict["obs"]
        # pdb.set_trace()
        force_feature = self.mlp_f(features[:,-8:])

        out = self.mlp_aev(features) + force_feature
        # out = self.layer_2(out)
        # out = self.layer_3(out) + step_feature
        # final_out = self.layer_4(out)
        final_out = self.mlp_interaction(out)
        self._last_value = self.values(features)

        return final_out , []

    def value_function(self):
        return torch.squeeze(self._last_value, -1)