import torch
import torch.nn.functional as F


########## Loss functions for SE2 (OMPL Style) ##########
def se2_split_loss(y_pred, y_true, rot_weight=0.2):
    """
    MSE Loss for SE2 Pose, position + weight * rotation.
    Also handle the case of having logvar in output by ignoring it.
    """
    # If y_pred includes log-variance or other extra dims, trim it
    if y_pred.shape[-1] > y_true.shape[-1]:
        y_pred = y_pred[:, : y_true.shape[-1]]

    # Compute MSE loss (as OMPL SE2 distance)
    pos_error = torch.norm(y_true[:, :2] - y_pred[:, :2], dim=1)
    rot_error = torch.abs(
        (y_true[:, 2] - y_pred[:, 2] + torch.pi) % (2 * torch.pi) - torch.pi
    )
    loss = torch.mean(pos_error.mean() + rot_weight * rot_error.mean())
    return loss


########## Loss functions for SE2 ##########
def mse_se2_loss(y_pred, y_true):
    """
    MSE Loss for SE2 Pose.
    Also handle the case of having logvar in output by ignoring it.
    """
    # If y_pred includes log-variance or other extra dims, trim it
    if y_pred.shape[-1] > y_true.shape[-1]:
        y_pred = y_pred[:, : y_true.shape[-1]]

    # Compute MSE loss in tangent space
    delta_err = get_se2_err(y_pred, y_true)
    loss = torch.mean(delta_err**2)
    # # to add weights to rotation error
    # weights = torch.tensor([1.0, 1.0, 0.25], device=delta_err.device)
    # loss = torch.mean(delta_err.pow(2) * weights)
    return loss


def nll_se2_loss(y_pred, y_true):
    """NLL Loss for SE2 Pose."""
    dim = y_true.shape[-1]
    mu, logvar = y_pred[:, :dim], y_pred[:, dim : 2 * dim]
    var = torch.exp(logvar)

    # L = 0.5 * log(var) + 0.5 * ((mu - y)^2 / var) + constant
    delta_err = get_se2_err(mu, y_true)
    nll = 0.5 * (logvar + (delta_err**2 / (var + 1e-8)))
    loss = torch.mean(nll)
    return loss


def beta_nll_se2_loss(y_pred, y_true, beta=0.2):
    """Beta NLL Loss for SE2 Pose."""
    dim = y_true.shape[-1]
    mu, logvar = y_pred[:, :dim], y_pred[:, dim : 2 * dim]
    var = torch.exp(logvar)

    # L = (var * beta) * nll
    delta_err = get_se2_err(mu, y_true)
    weights = var.detach() ** beta
    nll = 0.5 * (logvar + (delta_err**2 / (var + 1e-8)))
    beta_nll = weights * nll
    loss = torch.mean(beta_nll)
    return loss


def evidential_se2_loss(y_pred, y_true, lambda_reg=0.01):
    """
    Evidential Loss function for SE2 Pose
    which predicts both epistemic and aleatoric uncertainty.
    """
    dim = y_true.shape[-1]
    gamma = y_pred[:, :dim]  # E[µ]
    nu = y_pred[:, dim : 2 * dim]  # υ > 0
    alpha = y_pred[:, 2 * dim : 3 * dim]  # α > 1
    beta = y_pred[:, 3 * dim : 4 * dim]  # β > 0

    delta_err = get_se2_err(gamma, y_true)

    # NIG NLL
    omega = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * torch.log(torch.pi / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(nu * delta_err**2 + omega)
        + (torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))
    )
    nll = torch.mean(nll)

    # NIG Regularization
    evidence = 2 * nu + alpha
    # Original Regularization
    # reg = torch.abs(delta_err) * evidence
    # Normalized Regularization
    # https://arxiv.org/pdf/2205.10060
    wst = torch.sqrt(beta * (1 + nu) / alpha / nu + 1e-8)
    z = torch.pow(torch.abs(delta_err) / wst, 2)
    reg = z * evidence
    reg = torch.mean(reg)

    # Total Loss
    loss = nll + lambda_reg * reg
    return loss


########## SE2 Lie Group Functions ##########
def get_se2_err(y_pred, y_true):
    """Get the delta error for SE2 Pose."""
    y_pred_mat = to_se2_transform(y_pred)  # (N, 3, 3)
    y_true_mat = to_se2_transform(y_true)  # (N, 3, 3)
    err_mat = torch.matmul(inv_se2_transform(y_true_mat), y_pred_mat)
    delta_err = log_se2(err_mat)  # (N, 3)
    return delta_err


def log_se2(transforms):
    """
    Logarithm map from SE(2) -> se(2).

    transforms is a Nx3x3 batch homogeneous transformation matrix.
    Returns a Nx3 batch of vectors [delta_x, delta_y, omage] in tangent space.

    omega = arctan2(rot[1, 0], rot[0, 0])
    [dx, dy] = V(w)^-1 * t
    """
    device = transforms.device
    batch_size = transforms.shape[0]

    rot = transforms[:, 0:2, 0:2]  # (N, 2, 2)
    t = transforms[:, 0:2, 2]  # (N, 2)

    # Compute the omega
    omega = torch.atan2(rot[:, 1, 0], rot[:, 0, 0])  # (N,)
    c = torch.cos(omega)
    s = torch.sin(omega)

    # Compute dx and dy
    # V_inv = w / (2 * (1 - cos(w))) *
    #         [[    sin(w)   , 1 - cos(w)]
    #          [-(1 - cos(w)),   sin(w)  ]]
    # Avoid division by zero in V_inv
    eps = 1e-3
    mask = torch.abs(omega) >= eps
    # only compute for elements where omega is not too small
    V_inv = torch.zeros(
        batch_size, 2, 2, device=device, dtype=transforms.dtype
    )  # (N, 2, 2)
    denom = 2 * (1 - c)
    V_inv[mask] = (
        omega[mask].unsqueeze(-1).unsqueeze(-1)
        / denom[mask].unsqueeze(-1).unsqueeze(-1)
    ) * torch.stack(
        [
            torch.stack([s[mask], 1 - c[mask]], dim=1),
            torch.stack([c[mask] - 1, s[mask]], dim=1),
        ],
        dim=1,
    )
    V_inv[~mask] = torch.eye(2, device=device, dtype=transforms.dtype)

    dt = torch.bmm(V_inv, t.unsqueeze(-1)).squeeze(-1)  # (B, 2)
    dx, dy = dt[:, 0], dt[:, 1]

    return torch.stack([dx, dy, omega], dim=1)


def to_se2_transform(params):
    """
    Convert SE2 parameters to SE2 transform matrices.

    params is a Nx3 batch of vectors [x, y, theta].
    This is different from the tangent space vector [dx, dy, omega].
    Returns a Nx3x3 batch of SE2 transform matrices.
    """
    x, y, theta = params[:, 0], params[:, 1], params[:, 2]

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    return torch.stack(
        [
            torch.stack([cos_theta, -sin_theta, x], dim=1),
            torch.stack([sin_theta, cos_theta, y], dim=1),
            torch.stack([zeros, zeros, ones], dim=1),
        ],
        dim=1,
    )


def inv_se2_transform(transforms):
    """Invert a batch of SE2 transform matrices."""
    rot = transforms[:, :2, :2]  # (N, 2, 2)
    t = transforms[:, :2, 2:]  # (N, 2, 1)

    rot_transposed = rot.transpose(1, 2)  # (N, 2, 2)
    t_new = -torch.bmm(rot_transposed, t)  # (N, 2, 1)

    transforms_inv = torch.eye(
        3, device=transforms.device, dtype=transforms.dtype
    ).repeat(transforms.size(0), 1, 1)
    transforms_inv[:, :2, :2] = rot_transposed
    transforms_inv[:, :2, 2] = t_new.squeeze(-1)

    return transforms_inv


########## Loss functions for general use (not SE2 specifically) ##########
def mse_loss(y_pred, y_true):
    """
    A simple wrapper of mse loss.
    Also handle the case of having logvar in output by ignoring it.
    """
    # If y_pred includes log-variance or extra dims, trim it
    if y_pred.shape[-1] > y_true.shape[-1]:
        y_pred = y_pred[:, : y_true.shape[-1]]

    return F.mse_loss(y_pred, y_true)


def nll_loss(y_pred, y_true):
    """NLL Loss function, which includes uncertainty."""
    dim = y_true.shape[-1]
    mu, logvar = y_pred[:, :dim], y_pred[:, dim : 2 * dim]
    var = torch.exp(logvar)

    # L = 0.5 * log(var) + 0.5 * ((mu - y)^2 / var) + constant
    nll = 0.5 * (logvar + ((y_true - mu) ** 2) / (var + 1e-8))
    loss = torch.mean(nll)
    return loss


def beta_nll_loss(y_pred, y_true, beta=0.2):
    """Beta NLL Loss function"""
    dim = y_true.shape[-1]
    mu, logvar = y_pred[:, :dim], y_pred[:, dim : 2 * dim]
    var = torch.exp(logvar)

    # L = (var * beta) * nll
    weights = var.detach() ** beta
    nll = 0.5 * (logvar + ((y_true - mu) ** 2) / (var + 1e-8))
    beta_nll = weights * nll
    loss = torch.mean(beta_nll)
    return loss


def evidential_loss(y_pred, y_true, lambda_reg=0.01):
    """
    Evidential Loss function
    which predicts both epistemic and aleatoric uncertainty.
    """
    dim = y_true.shape[-1]
    gamma = y_pred[:, :dim]  # E[µ]
    nu = y_pred[:, dim : 2 * dim]  # υ > 0
    alpha = y_pred[:, 2 * dim : 3 * dim]  # α > 1
    beta = y_pred[:, 3 * dim : 4 * dim]  # β > 0

    # NIG NLL
    omega = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * torch.log(torch.pi / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(nu * (y_true - gamma) ** 2 + omega)
        + (torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))
    )
    nll = torch.mean(nll)

    # NIG Regularization
    evidence = 2 * nu + alpha
    # Original Regularization
    # reg = torch.abs(y_true - gamma) * evidence
    # Normalized Regularization
    # https://arxiv.org/pdf/2205.10060
    wst = torch.sqrt(beta * (1 + nu) / alpha / nu + 1e-8)
    z = torch.pow(torch.abs(y_true - gamma) / wst, 2)
    reg = z * evidence
    reg = torch.mean(reg)

    loss = nll + lambda_reg * reg
    return loss
