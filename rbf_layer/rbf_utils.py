import torch


# Norm functions
def l_norm(x, p=2):
    return torch.norm(x, p=p, dim=-1)


# Radial basis functions
def rbf_gaussian(x):
    return (-x.pow(2)).exp()


def rbf_linear(x):
    return x


def rbf_multiquadric(x):
    return (1 + x.pow(2)).sqrt()


def rbf_inverse_quadratic(x):
    return 1 / (1 + x.pow(2))


def rbf_inverse_multiquadric(x):
    return 1 / (1 + x.pow(2)).sqrt()