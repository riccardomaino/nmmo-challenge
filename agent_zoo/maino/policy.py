import torch
import torch.nn as nn

import pufferlib
import pufferlib.emulation
import pufferlib.models

from nmmo.entity.entity import EntityState

class Recurrent(pufferlib.models.RecurrentWrapper):
    def __init__(self, env, policy, input_size=256, hidden_size=256, num_layers=0):
        super().__init__(env, policy, input_size, hidden_size, num_layers)
        
class CustomNMMOPolicy(pufferlib.models.Policy):
    def __init__(self, env, input_size=128, hidden_size=256):
        super().__init__(env)
        
        self.tile_encoder = nn.Sequential(
            nn.Linear(7 * 7 * 3, hidden_size), # Example: 7x7 vision with 3 features
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.entity_encoder = nn.Linear(31, hidden_size) # NMMO entities have ~31 features
        
        self.backbone = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        

        self.value_head = nn.Linear(hidden_size, 1)
        self.logits_head = nn.Linear(hidden_size, env.single_action_space.nvec.sum())

    def encode_observations(self, observations):
        # observations is a flattened tensor if using PufferLib
        # We split and process tiles and entities separately
        
        # Simplified example:
        tiles = observations[:, :147]  # Slice based on your env config
        entities = observations[:, 147:178]
        
        tile_emb = self.tile_encoder(tiles)
        entity_emb = self.entity_encoder(entities)
        
        # Combine features
        combined = torch.cat([tile_emb, entity_emb], dim=-1)
        return self.backbone(combined), None

    def decode_actions(self, hidden, lookup):
        # Returns the action logits and the state value
        logits = self.logits_head(hidden)
        value = self.value_head(hidden)
        return logits, value