import torch
import torch.nn as nn
from torch import Tensor

# RBF Layer

class RBF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})
    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size
    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.

        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).

        basis_func: the radial basis function used to transform the scaled
            distances.

    Examples: for regression
    import torch.nn as nn
    import torch
    import torch_rbf as rbf
    class Net(nn.Module):
        def __init__(self, num_features, num_target):
            super(Net, self).__init__()

            bias_function = rbf.gaussian

            self.m = nn.Sequential(

                nn.Linear(num_features, 528),
                nn.ELU(),
                nn.BatchNorm1d(528),

                rbf.RBF(528, 1024, basis_func=bias_function),
                nn.BatchNorm1d(1024),

                nn.Linear(1024, 528),
                nn.ELU(),
                nn.BatchNorm1d(528),

                nn.Linear(528, num_target),
            )

        def forward(self, x):
            x_forward = self.m(x)
            return x_forward

    """

    centres: Tensor
    log_sigmas: Tensor

    def __init__(self, in_features: int, out_features: int,
                 basis_func, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.log_sigmas = nn.Parameter(torch.empty((out_features), **factory_kwargs))
        self.basis_func = basis_func
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input: Tensor) -> Tensor:
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        return self.basis_func(distances)


# RBFs

def gaussian(alpha):
    phi = torch.exp(-1 * alpha.pow(2))
    return phi

def linear(alpha):
    phi = alpha
    return phi

def quadratic(alpha):
    phi = alpha.pow(2)
    return phi

def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi

def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi

def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi

def poisson_two(alpha):
    phi = ((alpha - 2 * torch.ones_like(alpha)) / 2 * torch.ones_like(alpha)) \
          * alpha * torch.exp(-alpha)
    return phi

def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3 ** 0.5 * alpha) * torch.exp(-3 ** 0.5 * alpha)
    return phi

def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5 ** 0.5 * alpha + (5 / 3) \
           * alpha.pow(2)) * torch.exp(-5 ** 0.5 * alpha)
    return phi

def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """

    bases = {'gaussian': gaussian,
             'linear': linear,
             'quadratic': quadratic,
             'inverse quadratic': inverse_quadratic,
             'multiquadric': multiquadric,
             'inverse multiquadric': inverse_multiquadric,
             'spline': spline,
             'poisson one': poisson_one,
             'poisson two': poisson_two,
             'matern32': matern32,
             'matern52': matern52}
    return bases
