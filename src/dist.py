import torch
from torch.distributions.multivariate_normal import MultivariateNormal
 
    
class Normal(MultivariateNormal):
    def __init__(self, mean: torch.Tensor, cov: torch.Tensor) -> None:
        super().__init__(mean.squeeze(), cov.squeeze())
        
    def sample(self, shape):
        if type(shape) is int:
            shape = (shape, )
        return super().sample(shape)