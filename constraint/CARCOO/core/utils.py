import numpy as np
import torch
import torch.autograd as autograd
import random
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed=2):
    """setup seed for random procedure

    Args:
        seed (int, optional): Ramdom seed, set default as 2.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def sample_langevin(x_neg, dhs_model, stepsize=0.01, n_steps=50, noise=True):
    for i in range(n_steps):
        if noise:
            x_noise = torch.randn_like(x_neg).detach()
            x_noise.normal_()
            x_neg = x_neg + 0.001 * (n_steps - i - 1) / n_steps * x_noise
        else:
            x_neg = x_neg

        x_neg.requires_grad = True
        out = dhs_model.forward(x_neg)[0]

        x_neg_grad = torch.autograd.grad([out.sum()], [x_neg])[0] 
        x_neg = x_neg + stepsize * x_neg_grad
        x_neg = x_neg.detach()

    return x_neg

class RiskSuppressionFactor(object):
    def __init__(self, energy_min, energy_max, init_m):
        self.min = energy_min
        self.max = energy_max
        self.init_m = init_m

    def normalize(self, energy):
        normalized_energy =  (self.max - energy) /  (self.max - self.min)
        normalized_energy =  self.init_m * np.clip(normalized_energy, 0, 1)

        return normalized_energy

class neg_cons(object):
    def __init__(self, x, y, cons, cons_mean, cons_std):
        self.x = x
        self.y = y 
        self.cons = cons
        self.cons_mean = cons_mean
        self.cons_std = cons_std
    
    def feasible(self):
        feasible_x = []
        feasible_y = []
        self.cons = (self.cons.cpu().numpy())
        for i in range(len(self.x)):
            p = 0
            for j in range(len(self.cons[0])):
                if self.cons[i][j] < 0:
                    p = p + 1
            if p == 0:
                feasible_x.append(self.x[i].cpu().detach().tolist())
                feasible_y.append(self.y[i].cpu().detach().tolist())
        return torch.tensor(feasible_x).to(DEVICE), torch.tensor(feasible_y).to(DEVICE)
    
    def cons_x(self):
        self.cons = (self.cons.cpu().numpy() * self.cons_std + self.cons_mean).astype(np.int64)
        neg_cons = []
        for i in range(len(self.x)):
            p = 1
            for j in range(len(self.cons[0])):
                if self.cons[i][j] < 0:
                    p = p + 1
            if p > 1:
                neg_cons.append(self.x[i].cpu().detach().tolist())
        return torch.tensor(neg_cons).to(DEVICE)