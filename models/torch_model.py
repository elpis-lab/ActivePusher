import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple MLP model with dropout and uncertainty prediction options.

    in -> hidden -> 2*hidden -> 4*hidden -> 2*hidden -> hidden -> out/logvar
    """

    def __init__(self, in_dim, out_dim, hidden_dim=32, dropout=0):
        """Initialize the MLP model with given dimensions"""
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.in_net = self.block(in_dim, hidden_dim, dropout=0)
        self.block1 = self.block(hidden_dim, 2 * hidden_dim, dropout)
        self.block2 = self.block(2 * hidden_dim, 2 * hidden_dim, dropout)
        self.block3 = self.block(2 * hidden_dim, hidden_dim, dropout)
        self.out_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def block(self, in_dim, out_dim, dropout=0):
        """Simple block with linear layer, leaky relu, and dropout."""
        if dropout > 0:
            fc = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
            )
        else:
            fc = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU(),
            )
        return fc

    def layers(self, x):
        """Forward pass through the layers."""
        x = self.in_net(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x):
        """Forward pass, return pred. Include logvar if include uncertainty."""
        x = self.layers(x)
        pred = self.out_net(x)
        return pred


class MLPVar(MLP):
    def __init__(self, in_dim, out_dim, hidden_dim=32, dropout=0):
        super(MLPVar, self).__init__(in_dim, out_dim, hidden_dim, dropout)
        self.uncertainty_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        """Forward pass, return pred with logvar."""
        x = self.layers(x)
        pred = self.out_net(x)

        # Uncertainty
        uncertainty = self.uncertainty_net(x)
        pred = torch.cat([pred, uncertainty], dim=-1)
        return pred


class MLPEvidential(MLP):
    def __init__(self, in_dim, out_dim, hidden_dim=32, dropout=0):
        super(MLPEvidential, self).__init__(
            in_dim, out_dim, hidden_dim, dropout
        )
        self.gamma_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.nu_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.alpha_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        """Forward pass, return pred with logvar."""
        x = self.layers(x)

        # Uncertainty
        gamma = self.gamma_net(x)  # E[µ]
        nu = F.softplus(self.nu_net(x))  # υ > 0
        alpha = F.softplus(self.alpha_net(x)) + 1  # α > 1
        beta = F.softplus(self.beta_net(x))  # β > 0

        pred = torch.cat([gamma, nu, alpha, beta], dim=-1)
        return pred


class Physics(nn.Module):
    """Physics model wrapper."""

    def __init__(self, equation):
        super(Physics, self).__init__()
        self.equation = equation

    def forward(self, x):
        """Forward pass, return physics prediction."""
        return self.equation(x)


class ResidualPhysics(MLP):
    """
    Residual Physics model.
            ----------------
            |              ↓
    x -> physics -> MLP -> + -> pred
    |                ↑
    ------------------
    """

    def __init__(self, in_dim, out_dim, equation, hidden_dim=32, dropout=0):
        """Initialize the MLP model with given dimensions"""
        super(ResidualPhysics, self).__init__(
            in_dim + out_dim, out_dim, hidden_dim, dropout
        )
        self.equation = equation

    def forward(self, x):
        """Residual physics forward pass"""
        # Physics
        eq = self.equation(x)
        # MLP
        x_p = torch.cat([x, eq], dim=-1)
        x = self.layers(x_p)
        pred = self.out_net(x)
        # Residual
        pred = pred + eq
        return pred


class ResidualPhysicsVar(MLPVar):
    def __init__(self, in_dim, out_dim, equation, hidden_dim=32, dropout=0):
        super(ResidualPhysicsVar, self).__init__(
            in_dim + out_dim, out_dim, hidden_dim, dropout
        )
        self.equation = equation

    def forward(self, x):
        """Forward pass, return pred with logvar."""
        # Physics
        eq = self.equation(x)
        # MLP
        x_p = torch.cat([x, eq], dim=-1)
        x = self.layers(x_p)
        pred = self.out_net(x)
        # Residual
        pred = pred + eq

        # Uncertainty
        uncertainty = self.uncertainty_net(x)
        pred = torch.cat([pred, uncertainty], dim=-1)
        return pred


class ResidualPhysicsEvidential(MLPEvidential):
    def __init__(self, in_dim, out_dim, equation, hidden_dim=32, dropout=0):
        super(ResidualPhysicsEvidential, self).__init__(
            in_dim + out_dim, out_dim, hidden_dim, dropout
        )
        self.equation = equation

    def forward(self, x):
        """Forward pass, return pred with evidential uncertainty."""
        # Physics
        eq = self.equation(x)
        # MLP
        x_p = torch.cat([x, eq], dim=-1)
        x = self.layers(x_p)
        pred = self.out_net(x)
        # Residual
        pred = pred + eq

        # Uncertainty
        gamma = self.gamma_net(x)  # E[µ]
        nu = F.softplus(self.nu_net(x))  # υ > 0
        alpha = F.softplus(self.alpha_net(x)) + 1  # α > 1
        beta = F.softplus(self.beta_net(x))  # β > 0

        pred = torch.cat([gamma, nu, alpha, beta], dim=-1)
        return pred
