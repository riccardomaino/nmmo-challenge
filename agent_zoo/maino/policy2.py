import torch
import torch.nn.functional as F
import pufferlib.models
import pufferlib.emulation
from nmmo.entity.entity import EntityState

# Helper to initialize weights for better training stability
def orthogonal_init(layer, gain=1.0):
    torch.nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, 0)

class Recurrent(pufferlib.models.RecurrentWrapper):
    """Wraps the policy with an LSTM for memory."""
    def __init__(self, env, policy, input_size=256, hidden_size=256, num_layers=1):
        super().__init__(env, policy, input_size, hidden_size, num_layers)

class Policy(pufferlib.models.Policy):
    def __init__(self, env, input_size=256, hidden_size=256, task_size=2048):
        super().__init__(env)
        self.unflatten_context = env.unflatten_context

        #  Encoders: process raw data into embedding vectors
        self.tile_encoder = TileEncoder(input_size)
        self.player_encoder = PlayerEncoder(input_size, hidden_size)
        self.item_encoder = ItemEncoder(input_size, hidden_size)
        self.market_encoder = MarketEncoder(input_size, hidden_size)
        self.task_encoder = TaskEncoder(input_size, hidden_size, task_size)

        #  Fuse all embeddings into one vector, as inputs use: Tile, MyAgent, Inventory, Market, Task
        self.proj_fc = torch.nn.Linear(5 * input_size, hidden_size)
        orthogonal_init(self.proj_fc)

        # Decoders: turn the fused vector into Actions and Value
        self.action_decoder = ActionDecoder(input_size, hidden_size)
        self.value_head = torch.nn.Linear(hidden_size, 1)
        orthogonal_init(self.value_head)

    def encode_observations(self, flat_observations):
        """Turn flat buffer into a single 'hidden' vector representing the state."""
        env_outputs = pufferlib.emulation.unpack_batched_obs(
            flat_observations, self.unflatten_context
        )
        
        # Encode each part of the observation
        tile = self.tile_encoder(env_outputs["Tile"])
        player_embeddings, my_agent = self.player_encoder(
            env_outputs["Entity"], env_outputs["AgentId"][:, 0]
        )
        item_embeddings = self.item_encoder(env_outputs["Inventory"])
        market_embeddings = self.item_encoder(env_outputs["Market"])
        
        # Simple pooling for sets of items/market (mean)
        inventory = item_embeddings.mean(dim=1) 
        market = self.market_encoder(market_embeddings)
        task = self.task_encoder(env_outputs["Task"])

        # Fuse
        obs = torch.cat([tile, my_agent, inventory, market, task], dim=-1)
        obs = F.relu(self.proj_fc(obs))

        # Return the fused obs AND the embeddings needed for action masking (lookup)
        return obs, (
            player_embeddings,
            item_embeddings,
            market_embeddings,
            env_outputs["ActionTargets"],
        )

    def decode_actions(self, hidden, lookup):
        """Output action logits and value estimate."""
        actions = self.action_decoder(hidden, lookup)
        value = self.value_head(hidden)
        return actions, value


class ResnetBlock(torch.nn.Module):
    """Standard CNN block for image processing."""
    def __init__(self, in_planes, img_size=(15, 15)):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, in_planes, 3, 1, 1),
            torch.nn.LayerNorm((in_planes, *img_size)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_planes, in_planes, 3, 1, 1),
            torch.nn.LayerNorm((in_planes, *img_size)),
        )
    def forward(self, x):
        return self.model(x) + x  # Skip connection

class TileEncoder(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.type_embedding = torch.nn.Embedding(16, 32)
        # ResNet to process the 15x15 grid
        self.net = torch.nn.Sequential(
            ResnetBlock(34), # 32 (embed) + 2 (coord)
            torch.nn.Flatten(),
            torch.nn.Linear(34 * 15 * 15, input_size),
            torch.nn.ReLU()
        )
        
    def forward(self, tile):
        # Normalize coordinates
        tile_pos = tile[:, :, :2] / 128 - 0.5 
        tile_type = self.type_embedding(tile[:, :, 2].long().clip(0, 15))
        x = torch.cat([tile_pos, tile_type], dim=-1)
        # Reshape for CNN: (Batch, Channels, H, W)
        x = x.transpose(1, 2).view(x.shape[0], -1, 15, 15)
        return self.net(x)

class PlayerEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(512, 32)
        self.net = torch.nn.Linear(32 + 31, hidden_size) # 31 other attrs
        self.my_agent_net = torch.nn.Linear(hidden_size, input_size)

    def forward(self, agents, my_id):
        # Embed ID, concatenate other attributes
        embeds = self.embedding(agents[:, :, 0].long().clip(0, 511))
        x = torch.cat([embeds, agents[:, :, 1:]], dim=-1).float()
        x = F.relu(self.net(x))
        
        # Extract "Self" from the list of agents
        my_row_idx = (agents[:, :, 0] == my_id.unsqueeze(1)).int().argmax(dim=1)
        my_agent = x[torch.arange(x.shape[0]), my_row_idx]
        return x, F.relu(self.my_agent_net(my_agent))

class ItemEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.embed = torch.nn.Linear(14, hidden_size) # Simplification: linear proj of item attrs
    def forward(self, items):
        return F.relu(self.embed(items.float()))

class MarketEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = torch.nn.Linear(hidden_size, input_size)
    def forward(self, market):
        # Average pooling of all market items
        return F.relu(self.net(market).mean(dim=1))

class TaskEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, task_size):
        super().__init__()
        self.net = torch.nn.Linear(task_size, input_size)
    def forward(self, task):
        return F.relu(self.net(task.float()))

class ActionDecoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # Heads for every action
        self.heads = torch.nn.ModuleDict({
            "move": torch.nn.Linear(hidden_size, 5),
            "attack_style": torch.nn.Linear(hidden_size, 3),
            "attack_target": torch.nn.Linear(hidden_size, hidden_size),
        })

    def forward(self, hidden, lookup):
        player_embeds, _, _, targets = lookup
        actions = []
        
        #  Move
        actions.append(self.heads["move"](hidden).masked_fill(targets["Move"]["Direction"]==0, -1e9))
        
        # Attack Style
        actions.append(self.heads["attack_style"](hidden).masked_fill(targets["Attack"]["Style"]==0, -1e9))
        
        # Attack Target (Pointer Network): compare the 'hidden' state with every 'player_embedding' to see who to attack
        target_query = self.heads["attack_target"](hidden).unsqueeze(2) # (B, H, 1)
        target_logits = torch.matmul(player_embeds, target_query).squeeze(2) # (B, Num_Agents)
        actions.append(target_logits.masked_fill(targets["Attack"]["Target"]==0, -1e9))
        
        return actions