import torch
import torch.nn as nn
import torch.nn.functional as F

class MDNRNN(nn.Module):
    def __init__(self, z_dim, action_dim, hidden_dim, num_mixtures, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures
        self.z_dim = z_dim
        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_size=z_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.mdn_head = nn.Linear(hidden_dim, num_mixtures * (1 + z_dim + z_dim))

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden, cell_state = self.init_hidden(x.size(0), x.device)
        
        out, (hidden, cell_state) = self.rnn(x, (hidden, cell_state))

        mdn_out = self.mdn_head(out)

        return mdn_out, hidden, cell_state

    def split_mdn_params(self, mdn_out):
        K = self.num_mixtures
        Z = self.z_dim

        pi    = mdn_out[..., :K]
        mu    = mdn_out[..., K:K + K*Z]
        sigma = mdn_out[..., K + K*Z:]

        pi    = F.softmax(pi, dim=-1)
        mu    = mu.view(*mu.shape[:-1], K, Z)
        sigma = torch.exp(sigma)
        sigma = sigma.view(*sigma.shape[:-1], K, Z)

        return pi, mu, sigma
    
    def init_hidden(self, batch_size, device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h,c)