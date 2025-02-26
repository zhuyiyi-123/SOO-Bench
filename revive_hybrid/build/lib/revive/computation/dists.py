

import numpy as np
from typing import Tuple

import torch
from torch.functional import F
from torch.distributions import constraints
from torch.distributions import Normal, Categorical, OneHotCategorical

import pyro
from pyro.distributions.torch_distribution import TorchDistributionMixin
from pyro.distributions.kl import register_kl, kl_divergence


def exportable_broadcast(tensor1 : torch.Tensor, tensor2 : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ''' Broadcast tensors to the same shape using onnx exportable operators '''
    if len(tensor1.shape) < len(tensor2.shape):
        tensor2, tensor1 = exportable_broadcast(tensor2, tensor1)
    else:
        shape1 = tensor1.shape
        shape2 = tensor2.shape
        if len(shape1) == len(shape2):
            final_shape = [max(s1, s2) for s1, s2 in zip(shape1, shape2)]
            tensor1 = tensor1.expand(*final_shape)
            tensor2 = tensor2.expand(*final_shape)
        else:
            tensor2 = tensor2.expand(*shape1)
    return tensor1, tensor2

class ReviveDistributionMixin:
    '''Define revive distribution API'''

    @property
    def mode(self,):
        '''return the most likely sample of the distributions'''
        raise NotImplementedError 

    @property
    def std(self):
        '''return the standard deviation of the distributions'''
        raise NotImplementedError

    def sample_with_logprob(self, sample_shape=torch.Size()):
        sample = self.rsample(sample_shape) if self.has_rsample else self.sample(sample_shape)
        return sample, self.log_prob(sample)

class ReviveDistribution(pyro.distributions.TorchDistribution, ReviveDistributionMixin):
    pass

class ExportableNormal(Normal):
    def __init__(self, loc, scale, validate_args):
        self.loc, self.scale = exportable_broadcast(loc, scale)
        batch_shape = self.loc.size()
        super(Normal, self).__init__(batch_shape, validate_args=validate_args)

class ExportableCategorical(Categorical):
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = value.long().unsqueeze(-1)
        value, log_pmf = exportable_broadcast(value, self.logits)
        value = value[..., :1]
        return log_pmf.gather(-1, value).squeeze(-1)

class DiagnalNormal(ReviveDistribution):
    def __init__(self, loc, scale, validate_args=False):
        self.base_dist = ExportableNormal(loc, scale, validate_args)
        batch_shape = torch.Size(loc.shape[:-1])
        event_shape = torch.Size([loc.shape[-1]])
        super(DiagnalNormal, self).__init__(batch_shape, event_shape, validate_args)

    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, sample):
        log_prob = self.base_dist.log_prob(sample)
        return torch.sum(log_prob, dim=-1)

    def entropy(self):
        entropy = self.base_dist.entropy()
        return torch.sum(entropy, dim=-1)

    def shift(self, mu_shift):
        '''shift the distribution, useful in local mode transition'''
        return DiagnalNormal(self.base_dist.loc + mu_shift, self.base_dist.scale)

    @property
    def mode(self):
        return self.base_dist.mean

    @property
    def std(self):
        return self.base_dist.scale

class TransformedDistribution(torch.distributions.TransformedDistribution):
    @property
    def mode(self):
        x = self.base_dist.mode
        for transform in self.transforms:
            x = transform(x)
        return x

    @property
    def std(self):
        raise NotImplementedError # TODO: fix this!
    
    def entropy(self, num=torch.Size([100])):
        # use samples to estimate entropy
        samples = self.rsample(num)
        log_prob = self.log_prob(samples)
        entropy = - torch.mean(log_prob, dim=0)
        return entropy

class DiscreteLogistic(ReviveDistribution):
    r"""
    Model discrete variable with Logistic distribution, inspired from:
    https://github.com/openai/vdvae/blob/main/vae_helpers.py

    As far as I know, the trick was proposed in:
    Salimans, Tim, Andrej Karpathy, Xi Chen, and Diederik P. Kingma
    "Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications." arXiv preprint arXiv:1701.05517 (2017).

    :param loc: Location parameter, assert it have been normalized to [-1, 1]
    :param scale: Scale parameter.
    :param num: Number of possible value for each dimension.
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, num, *, validate_args=False):
        self.loc, self.scale = exportable_broadcast(loc, scale)
        self.num = torch.tensor(num).to(loc)
        batch_shape = torch.Size(loc.shape[:-1])
        event_shape = torch.Size([loc.shape[-1]])
        super(DiscreteLogistic, self).__init__(batch_shape, event_shape, validate_args)

    def log_prob(self, value):
        mid = value - self.loc
        plus = (mid + 1 / (self.num - 1)) / self.scale
        minus = (mid - 1 / (self.num - 1)) / self.scale
        prob = torch.sigmoid(plus) - torch.sigmoid(minus)
        log_prob_left_edge = plus - F.softplus(plus)
        log_prob_right_edge = - F.softplus(minus)
        z = mid / self.scale
        log_prob_extreme = z - torch.log(self.scale) - 2 * F.softplus(z)

        return torch.where(value < - 0.999,
                           log_prob_left_edge,
                           torch.where(value > 0.999,
                                       log_prob_right_edge,
                                       torch.where(prob > 1e-5,
                                                   torch.log(prob + 1e-5),
                                                   log_prob_extreme))).sum(dim=-1)
    
    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = self.loc.new_empty(shape).uniform_()
        value = self.icdf(u)
        round_value = self.round(value)
        return torch.clamp(round_value, -1, 1)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        u = self.loc.new_empty(shape).uniform_()
        value = self.icdf(u)
        round_value = self.round(value)
        return torch.clamp(round_value, -1, 1) + value - value.detach()

    def cdf(self, value):
        z = (value - self.loc) / self.scale
        return torch.sigmoid(z)

    def icdf(self, value):
        return self.loc + self.scale * torch.logit(value, eps=1e-5)

    def round(self, value):
        value = (value + 1) / 2 * (self.num - 1)
        return torch.round(value) / (self.num - 1) * 2 - 1

    @property
    def mode(self):
        return torch.clamp(self.round(self.loc), -1, 1) + self.loc - self.loc.detach()

    @property
    def std(self):
        return self.scale * np.pi / 3 ** 0.5

    def entropy(self):
        return torch.sum(torch.log(self.scale) + 2, dim=-1)

class Onehot(OneHotCategorical, TorchDistributionMixin, ReviveDistributionMixin):
    """Differentiable Onehot Distribution"""

    has_rsample = True
    _validate_args = False

    def __init__(self, logits=None, validate_args=False):
        self._categorical = ExportableCategorical(logits=logits, validate_args=False)
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        super(OneHotCategorical, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        # Implement straight-through estimator
        # Bengio et.al. Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation 
        sample = self.sample(sample_shape)
        return sample + self.probs - self.probs.detach()
    
    @property
    def mode(self):
        index = torch.argmax(self.logits, dim=-1)
        num_classes = self.event_shape[0]
        if torch.is_tensor(num_classes):
            num_classes = num_classes.item()
        sample = F.one_hot(index, num_classes)
        return sample + self.probs - self.probs.detach()

    @property
    def std(self):
        return self.variance

class GaussianMixture(pyro.distributions.MixtureOfDiagNormals, ReviveDistributionMixin):
    def __init__(self, locs, coord_scale, component_logits):
        self.batch_mode = (locs.dim() > 2)
        assert(coord_scale.shape == locs.shape)
        assert(self.batch_mode or locs.dim() == 2), \
            "The locs parameter in MixtureOfDiagNormals should be K x D dimensional (or B x K x D if doing batches)"
        if not self.batch_mode:
            assert(coord_scale.dim() == 2), \
                "The coord_scale parameter in MixtureOfDiagNormals should be K x D dimensional"
            assert(component_logits.dim() == 1), \
                "The component_logits parameter in MixtureOfDiagNormals should be K dimensional"
            assert(component_logits.size(-1) == locs.size(-2))
            batch_shape = ()
        else:
            assert(coord_scale.dim() > 2), \
                "The coord_scale parameter in MixtureOfDiagNormals should be B x K x D dimensional"
            assert(component_logits.dim() > 1), \
                "The component_logits parameter in MixtureOfDiagNormals should be B x K dimensional"
            assert(component_logits.size(-1) == locs.size(-2))
            batch_shape = tuple(locs.shape[:-2])

        self.locs = locs
        self.coord_scale = coord_scale
        self.component_logits = component_logits
        self.dim = locs.size(-1)
        self.categorical = ExportableCategorical(logits=component_logits)
        self.probs = self.categorical.probs
        ReviveDistribution.__init__(self, batch_shape=torch.Size(batch_shape), event_shape=torch.Size((self.dim,)))

    @property
    def mode(self):
        # NOTE: this is only an approximate mode
        which = self.categorical.logits.max(dim=-1)[1]
        which = which.unsqueeze(dim=-1).unsqueeze(dim=-1)
        which_expand = which.expand(tuple(which.shape[:-1] + (self.locs.shape[-1],)))
        loc = torch.gather(self.locs, -2, which_expand).squeeze(-2)
        return loc

    @property
    def std(self):
        p = self.categorical.probs
        return torch.sum(self.coord_scale * p.unsqueeze(-1), dim=-2)

    def shift(self, mu_shift):
        '''shift the distribution, useful in local mode transition'''
        return GaussianMixture(self.locs + mu_shift.unsqueeze(dim=-2), self.coord_scale, self.component_logits)

    def entropy(self):
        p = self.categorical.probs
        normal = DiagnalNormal(self.locs, self.coord_scale)
        entropy = normal.entropy()
        return torch.sum(p * entropy, dim=-1)

class MixDistribution(ReviveDistribution):
    """Collection of multiple distributions"""

    arg_constraints = {}
    
    def __init__(self, dists):
        super().__init__()
        assert len(set([dist.batch_shape for dist in dists])) == 1, "the batch shape of all distributions should be equal"
        assert len(set([len(dist.event_shape) == 1 for dist in dists])) == 1, "the event shape of all distributions should have length 1"
        self.dists = dists
        self.sizes = [dist.event_shape[0] for dist in self.dists]
        batch_shape = self.dists[0].batch_shape
        event_shape = torch.Size((sum(self.sizes),))
        super(MixDistribution, self).__init__(batch_shape, event_shape)

    def sample(self, num=torch.Size()):
        samples = [dist.sample(num) for dist in self.dists]
        return torch.cat(samples, dim=-1)

    def rsample(self, num=torch.Size()): 
        samples = [dist.rsample(num) for dist in self.dists]
        return torch.cat(samples, dim=-1)

    def entropy(self):
        return sum([dist.entropy() for dist in self.dists])  

    def log_prob(self, x):
        if type(x) == list:
            return [self.dists[i].log_prob(x[i]) for i in range(len(x))]
        # manually split the tensor
        x = torch.split(x, self.sizes, dim=-1)
        return sum([self.dists[i].log_prob(x[i]) for i in range(len(x))])

    @property
    def mode(self):
        modes = [dist.mode for dist in self.dists]
        return torch.cat(modes, dim=-1)

    @property
    def std(self):
        stds = [dist.std for dist in self.dists]
        return torch.cat(stds, dim=-1)

    def shift(self, mu_shift):
        '''shift the distribution, useful in local mode transition'''
        assert all([type(dist) in [DiagnalNormal, GaussianMixture] for dist in self.dists]), \
            "all the distributions should have method `shift`"
        return MixDistribution([dist.shift(mu_shift) for dist in self.dists])

@register_kl(DiagnalNormal, DiagnalNormal)
def _kl_diagnalnormal_diagnalnormal(p : DiagnalNormal, q : DiagnalNormal):
    kl = kl_divergence(p.base_dist, q.base_dist)
    kl = torch.sum(kl, dim=-1)
    return kl

@register_kl(Onehot, Onehot)
def _kl_onehot_onehot(p : Onehot, q : Onehot):
    kl = (p.probs * (torch.log(p.probs) - torch.log(q.probs))).sum(dim=-1)
    return kl

@register_kl(GaussianMixture, GaussianMixture)
def _kl_gmm_gmm(p : GaussianMixture, q : GaussianMixture):
    samples = p.rsample()
    log_p = p.log_prob(samples)
    log_q = q.log_prob(samples)
    return log_p - log_q

@register_kl(MixDistribution, MixDistribution)
def _kl_mix_mix(p : MixDistribution, q : MixDistribution):
    assert all([type(_p) == type(_q) for _p, _q in zip(p.dists, q.dists)])
    kl = 0
    for _p, _q in zip(p.dists, q.dists):
        kl = kl + kl_divergence(_p, _q)
    return kl

@register_kl(DiscreteLogistic, DiscreteLogistic)
def _kl_discrete_logistic_discrete_logistic(p : DiscreteLogistic, q : DiscreteLogistic):
    assert torch.all(p.num == q.num)
    # NOTE: Cannot compute the kl divergence in the analysitical form, use 100 samples to estimate.
    samples = p.sample((100,))
    p_log_prob = p.log_prob(samples)
    q_log_prob = q.log_prob(samples)
    return torch.mean(p_log_prob - q_log_prob, dim=0)
     
if __name__ == '__main__':
    print('-' * 50)
    onehot = Onehot(torch.rand(2, 10, requires_grad=True))
    print('onehot batch shape', onehot.batch_shape)
    print('onehot event shape', onehot.event_shape)
    print('onehot sample', onehot.sample())
    print('onehot rsample', onehot.rsample())
    print('onehot log prob', onehot.sample_with_logprob()[1])
    print('onehot mode', onehot.mode)
    print('onehot std', onehot.std)
    print('onehot entropy', onehot.entropy())
    _onehot = Onehot(torch.rand(2, 10, requires_grad=True))
    print('onehot kl', kl_divergence(onehot, _onehot))

    print('-' * 50)
    mixture = GaussianMixture(
        torch.rand(2, 6, 4, requires_grad=True), 
        torch.rand(2, 6, 4, requires_grad=True),
        torch.rand(2, 6, requires_grad=True), 
    )
    print('gmm batch shape', mixture.batch_shape)
    print('gmm event shape', mixture.event_shape)
    print('gmm sample', mixture.sample())
    print('gmm rsample', mixture.rsample())
    print('gmm log prob', mixture.sample_with_logprob()[1])
    print('gmm mode', mixture.mode)
    print('gmm std', mixture.std)
    print('gmm entropy', mixture.entropy())
    _mixture = GaussianMixture(
        torch.rand(2, 6, 4, requires_grad=True), 
        torch.rand(2, 6, 4, requires_grad=True),
        torch.rand(2, 6, requires_grad=True), 
    )
    print('gmm kl', kl_divergence(mixture, _mixture))

    print('-' * 50)
    normal = DiagnalNormal(
        torch.rand(2, 5, requires_grad=True), 
        torch.rand(2, 5, requires_grad=True)
    )
    print('normal batch shape', normal.batch_shape)
    print('normal event shape', normal.event_shape)
    print('normal sample', normal.sample())
    print('normal rsample', normal.rsample())
    print('normal log prob', normal.sample_with_logprob()[1])
    print('normal mode', normal.mode)
    print('normal std', normal.std)
    print('normal entropy', normal.entropy())
    _normal = DiagnalNormal(
        torch.rand(2, 5, requires_grad=True), 
        torch.rand(2, 5, requires_grad=True)
    )
    print('normal kl', kl_divergence(normal, _normal))

    print('-' * 50)
    discrete_logic = DiscreteLogistic(
        torch.rand(2, 5, requires_grad=True) * 2 - 1, 
        torch.rand(2, 5, requires_grad=True),
        [5, 9, 17, 33, 65],
    )
    print('discrete logistic batch shape', discrete_logic.batch_shape)
    print('discrete logistic event shape', discrete_logic.event_shape)
    print('discrete logistic sample', discrete_logic.sample())
    print('discrete logistic rsample', discrete_logic.rsample())
    print('discrete logistic log prob', discrete_logic.sample_with_logprob()[1])
    print('discrete logistic mode', discrete_logic.mode)
    print('discrete logistic std', discrete_logic.std)
    print('discrete logistic entropy', discrete_logic.entropy())
    _discrete_logic = DiscreteLogistic(
        torch.rand(2, 5, requires_grad=True) * 2 - 1, 
        torch.rand(2, 5, requires_grad=True),
        [5, 9, 17, 33, 65],
    )
    print('discrete logistic kl', kl_divergence(discrete_logic, _discrete_logic))

    print('-' * 50)
    mix = MixDistribution([onehot, mixture, normal, discrete_logic])
    print('mix batch shape', mix.batch_shape)
    print('mix event shape', mix.event_shape)
    print('mix sample', mix.sample())
    print('mix rsample', mix.rsample())
    print('mix log prob', mix.sample_with_logprob()[1])
    print('mix mode', mix.mode)
    print('mix std', mix.std)
    print('mix entropy', mix.entropy())
    _mix = MixDistribution([_onehot, _mixture, _normal, _discrete_logic])
    print('mix kl', kl_divergence(mix, _mix))
