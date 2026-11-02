"""DMFC model v2: rank-1 integrator + computation RNN.

Component 1: Rank-1 Integrator RNN
  - W_hh = u @ v^T (hard rank-1 constraint)
  - Receives [startx, starty, dir_onehot(4)] = 6D
  - One sustained integration axis, rest carry identity (start, direction)
  - Auxiliary linear readout → ramp (forces common timing axis)

Component 2: Computation RNN
  - Full recurrent network, activated after fix_off
  - Receives integrator hidden state directly
  - Computes target from integrated progress + identity info
  - Linear readout → target x, y
"""
import torch
import torch.nn as nn
import numpy as np


class Rank1Integrator(nn.Module):
    """Rank-1 RNN: W_hh = u @ v^T, one sustained integration axis."""

    def __init__(self, input_size, hidden_size, alpha, linear=False, flatness=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.linear = linear
        self.flatness = flatness

        self.W_ih = nn.Linear(input_size, hidden_size)
        self.u = nn.Parameter(torch.randn(hidden_size, 1) * 0.1)
        self.v = nn.Parameter(torch.randn(hidden_size, 1) * 0.1)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_seq, h0=None):
        B, T, _ = x_seq.shape
        h = h0 if h0 is not None else torch.zeros(B, self.hidden_size, device=x_seq.device)

        hiddens = []
        for t in range(T):
            # W_hh @ h = u @ v^T @ h = u * (v^T h)
            vTh = (h @ self.v)  # (B, 1)
            rec = self.u.squeeze(1).unsqueeze(0) * vTh  # (B, hidden)
            pre = self.W_ih(x_seq[:, t]) + rec + self.bias
            if self.linear:
                h_new = pre
            else:
                c = self.flatness
                h_new = c * torch.tanh(pre / c)
            h = (1 - self.alpha) * h + self.alpha * h_new
            hiddens.append(h)

        return torch.stack(hiddens, dim=1)


class Rank2Integrator(nn.Module):
    """Rank-2 RNN: W_hh = U @ V^T (U,V are hidden x 2)."""

    def __init__(self, input_size, hidden_size, alpha, linear=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.linear = linear

        self.W_ih = nn.Linear(input_size, hidden_size)
        self.U = nn.Parameter(torch.randn(hidden_size, 2) * 0.1)
        self.V = nn.Parameter(torch.randn(hidden_size, 2) * 0.1)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_seq, h0=None):
        B, T, _ = x_seq.shape
        h = h0 if h0 is not None else torch.zeros(B, self.hidden_size, device=x_seq.device)

        hiddens = []
        for t in range(T):
            VTh = h @ self.V  # (B, 2)
            rec = VTh @ self.U.T  # (B, hidden)
            pre = self.W_ih(x_seq[:, t]) + rec + self.bias
            if self.linear:
                h_new = pre
            else:
                h_new = torch.tanh(pre)
            h = (1 - self.alpha) * h + self.alpha * h_new
            hiddens.append(h)

        return torch.stack(hiddens, dim=1)


class ComputationRNN(nn.Module):
    """Full RNN for delay-period computation.

    Receives integrator hidden state. Only active when fix_on == 0.
    """

    def __init__(self, input_size, hidden_size, alpha):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = alpha

        self.W_ih = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_seq, active_mask, h0=None):
        B, T, _ = x_seq.shape
        h = h0 if h0 is not None else torch.zeros(B, self.hidden_size, device=x_seq.device)

        hiddens = []
        for t in range(T):
            pre = self.W_ih(x_seq[:, t]) + self.W_hh(h) + self.bias
            h_new = torch.tanh(pre)
            h_candidate = (1 - self.alpha) * h + self.alpha * h_new

            mask = active_mask[:, t].unsqueeze(1)
            h = mask * h_candidate + (1 - mask) * h
            hiddens.append(h)

        return torch.stack(hiddens, dim=1)


class Rank2CompRNN(nn.Module):
    """Rank-2 computation RNN: W_hh = U @ V^T, U,V are (hidden, 2).
    Only 2 sustained dimensions. Gated by active_mask."""

    def __init__(self, input_size, hidden_size, alpha):
        super().__init__()
        self.hidden_size = hidden_size
        self.alpha = alpha

        self.W_ih = nn.Linear(input_size, hidden_size)
        self.U = nn.Parameter(torch.randn(hidden_size, 2) * 0.1)
        self.V = nn.Parameter(torch.randn(hidden_size, 2) * 0.1)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_seq, active_mask, h0=None):
        B, T, _ = x_seq.shape
        h = h0 if h0 is not None else torch.zeros(B, self.hidden_size, device=x_seq.device)

        hiddens = []
        for t in range(T):
            VTh = h @ self.V  # (B, 2)
            rec = VTh @ self.U.T  # (B, hidden)
            pre = self.W_ih(x_seq[:, t]) + rec + self.bias
            h_new = torch.tanh(pre)
            h_candidate = (1 - self.alpha) * h + self.alpha * h_new

            mask = active_mask[:, t].unsqueeze(1)
            h = mask * h_candidate + (1 - mask) * h
            hiddens.append(h)

        return torch.stack(hiddens, dim=1)


class DMFCModelV2(nn.Module):
    """Rank-1 integrator + comp RNN + dual readout.

    Input: (batch, T, 7) — [startx, starty, dir_onehot(4), fix_on]
    Output: (batch, T, 2) — [target_x, target_y]
    Aux output: (batch, T, 1) — ramp prediction from integrator
    """

    def __init__(self, config):
        super().__init__()
        dt = config['dt']
        tau_int = config['integrator']['tau']
        tau_comp = config['comp_rnn']['tau']

        self.integrator = Rank1Integrator(
            input_size=config['integrator']['input_size'],
            hidden_size=config['integrator']['hidden_size'],
            alpha=dt / tau_int,
        )

        self.comp_rnn = ComputationRNN(
            input_size=config['integrator']['hidden_size'],
            hidden_size=config['comp_rnn']['hidden_size'],
            alpha=dt / tau_comp,
        )

        self.readout = nn.Linear(
            config['comp_rnn']['hidden_size'],
            config['readout']['output_size'],
        )

        # Auxiliary ramp readout from integrator
        self.ramp_readout = nn.Linear(
            config['integrator']['hidden_size'], 1,
        )

    def forward(self, x):
        """
        x: (batch, T, 7) — [startx, starty, dir(4), fix_on]
        Returns:
            output: (batch, T, 2) — predicted target
            ramp: (batch, T, 1) — auxiliary ramp prediction
            internals: dict of intermediate states
        """
        fix_on = x[:, :, 6]
        active_mask = (fix_on < 0.5).float()

        # Integrator receives [startx, starty, dir(4)] = channels 0:6
        h_int = self.integrator(x[:, :, :6])

        # Aux ramp readout
        ramp = self.ramp_readout(h_int)

        # Comp RNN receives integrator state, gated by fix_off
        h_comp = self.comp_rnn(h_int, active_mask)

        # Behavioral readout
        output = self.readout(h_comp)

        internals = {
            'h_integrator': h_int,
            'h_comp': h_comp,
            'active_mask': active_mask,
            'ramp': ramp,
        }
        return output, ramp, internals


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from src.rnn.configs.dmfc_config_v2 import config

    model = DMFCModelV2(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {n_params}')
    print(f'  Integrator: {sum(p.numel() for p in model.integrator.parameters())}')
    print(f'    W_ih: {model.integrator.W_ih.weight.numel()}')
    print(f'    u:    {model.integrator.u.numel()}')
    print(f'    v:    {model.integrator.v.numel()}')
    print(f'  Comp RNN:   {sum(p.numel() for p in model.comp_rnn.parameters())}')
    print(f'  Readout:    {sum(p.numel() for p in model.readout.parameters())}')
    print(f'  Ramp readout: {sum(p.numel() for p in model.ramp_readout.parameters())}')

    # Test forward pass
    x = torch.randn(4, 81, 7)
    out, ramp, internals = model(x)
    print(f'\nForward pass:')
    print(f'  input:        {x.shape}')
    print(f'  output:       {out.shape}')
    print(f'  ramp:         {ramp.shape}')
    print(f'  h_integrator: {internals["h_integrator"].shape}')
    print(f'  h_comp:       {internals["h_comp"].shape}')
