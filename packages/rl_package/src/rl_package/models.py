import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImpalaCNN(nn.Module):
    def __init__(self, in_channels=12, feature_dim=256):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 16, 8, stride=4), nn.LeakyReLU(),
            nn.Conv2d(16, 32, 4, stride=2), nn.LeakyReLU(), 
            nn.Flatten(),
            nn.Linear(32 * 81, feature_dim)
        )
        
    def forward(self, obs):
        x = obs.float() / 255.0 - 0.5
        h = self.main(x)
        return F.layer_norm(h, (h.size(-1),))
    
def weight_init(m):
    """Orthogonal initialization for stable gradients in RL."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data') and m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data') and m.bias is not None:
            m.bias.data.fill_(0.0)

def impala_init(module, weight_init, bias_init, gain=1):
    """Custom initialization utility for Impala-style architectures."""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class DrQEncoderV2(nn.Module):  
    def __init__(self, obs_shape=9, feature_dim=50, pretrained=False):
        super().__init__()

        #assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(4, 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(),
                                     nn.Flatten())
        self.linear = nn.Linear(self.repr_dim, feature_dim)
        self.apply(weight_init)

        #if pretrained:
        #    pretrained_agent = torch.load(pretrained)
        #    self.load_state_dict(pretrained_agent.encoder.state_dict())

        num_params = sum(p.numel() for p in self.parameters())
        print(f"Num params of encoder: {num_params}")

    def forward(self, obs):
        obs = obs.float()/ 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        
        h = self.linear(h)
        return F.layer_norm(h, h.size())
    
class SACActor(nn.Module):
    def __init__(self, grayscale=True, action_dim=2):
        super().__init__()

        self.channels = 4 if grayscale else 12
        self.encoder = ImpalaCNN(in_channels=self.channels, feature_dim=256)
        
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)

        self.register_buffer("action_scale", torch.ones(action_dim, dtype=torch.float32))
        self.register_buffer("action_bias", torch.zeros(action_dim, dtype=torch.float32))

    def forward(self, x):
        x = self.encoder(x)
        return self.fc_mean(x), self.fc_logstd(x)

    def get_action(self, x):
        """Only returns the mean action"""
        mean, _ = self.forward(x)
        v = torch.sigmoid(mean[:, 0:1])
        omega = torch.tanh(mean[:, 1:2])
        
        action = torch.cat([v, omega], dim=-1)
        return None, None, action * self.action_scale + self.action_bias
    
class TD3Actor(nn.Module):
    def __init__(self, grayscale=True, action_dim=2):
        super().__init__()
        self.channels = 4 if grayscale else 12

        self.encoder = DrQEncoderV2(obs_shape=9, feature_dim=256)
        
        self.fc_mu = nn.Linear(256, action_dim)
        
        self.register_buffer("action_scale", torch.ones(action_dim, dtype=torch.float32))
        self.register_buffer("action_bias", torch.zeros(action_dim, dtype=torch.float32))

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        v = torch.sigmoid(mu[:, 0:1]) 
        omega = torch.tanh(mu[:, 1:2])
        action = torch.cat([v, omega], dim=-1)
        return action * self.action_scale + self.action_bias