import torch
import numpy as np


class AbstractDistribution:
    def sample(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()


class DiracDistribution(AbstractDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def mode(self):
        return self.value


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, grain_indices, codebook_mask, deterministic=False):
        self.parameters = parameters
        self.codebook_mask = codebook_mask
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        if grain_indices is not None:
            self.grain_indices = grain_indices.repeat_interleave(2,dim=1).repeat_interleave(2,dim=2).unsqueeze(1).bool()
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def get_noise(self, noise=None):
        noise = torch.randn_like(self.mean)
        noise_coarse = torch.randn(self.mean.shape[0],4,16,16,device=self.mean.device)
        noise_coarse = noise_coarse.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        noise = noise * self.grain_indices + noise_coarse * ~self.grain_indices
        return noise

    def sample(self):
        x = self.mean + self.std * self.get_noise().to(device=self.parameters.device)
        if self.grain_indices is not None:
            x = x * ~self.grain_indices + self.mean * self.grain_indices
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None and self.grain_indices is not None:
                return 0.5 * torch.sum((torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar) *  self.codebook_mask,
                                       dim=[1, 2, 3])

            elif other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
