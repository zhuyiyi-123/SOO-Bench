from collections import defaultdict
import torch
import torch.nn as nn
from ARCOO.utils import sample_langevin

class DualHeadSurogateTrainer(object):
    def __init__(self, dhs_model,
                 dhs_model_prediction_opt=torch.optim.Adam,
                 dhs_model_energy_opt=torch.optim.Adam,
                 surrogate_lr=0.001,
                 init_m=0.05, ldk=50):

        self.dhs_model = dhs_model
        self.dhs_model_prediction_opt = dhs_model_prediction_opt(self.dhs_model.parameters(), lr=surrogate_lr)
        self.dhs_model_energy_opt = dhs_model_energy_opt(self.dhs_model.parameters(), lr=surrogate_lr)

        # algorithm hyper parameters
        self.init_m = init_m
        self.ldk = ldk
        self.dhs_model_prediction_loss = nn.MSELoss()

    def train(self, dataloder, e_train=True):
        statistics = defaultdict(list)
        for (x, y) in dataloder:
            if e_train:
                # energy head training
                neg_x = sample_langevin(x, self.dhs_model, self.init_m, self.ldk, noise=True) # @TODO: change the para to train_config
                
                pos_energy = self.dhs_model(x)[1]
                neg_energy = self.dhs_model(neg_x)[1]
                energy_loss = pos_energy.mean() - neg_energy.mean()
                energy_loss += torch.pow(pos_energy, 2).mean() + torch.pow(neg_energy, 2).mean()

                energy_loss = energy_loss.mean()
                self.dhs_model_energy_opt.zero_grad()
                energy_loss.backward()
                self.dhs_model_energy_opt.step()

                statistics[f'train/energy_cdloss'].append(energy_loss)

            # prediction head training            
            self.dhs_model.train()   
            score_pos = self.dhs_model(x)[0]
            
            mse = self.dhs_model_prediction_loss(score_pos, y)
            statistics[f'train/mse'].append(mse)
            
            loss = mse
            self.dhs_model_prediction_opt.zero_grad()
            loss.backward()
            self.dhs_model_prediction_opt.step()

        return statistics

    def validate(self, dataloder):
        statistics = defaultdict(list)
        for (x, y) in dataloder:
            self.dhs_model.eval()
            score_pos = self.dhs_model(x)[0]
            mse = self.dhs_model_prediction_loss(score_pos, y)
            statistics[f'val/mse'].append(mse) 

        return statistics

    def launch(self, train_dl, validate_dl, logger, epochs, e_train=True):
        for e in range(epochs):
            self.train(train_dl, e_train)
            self.validate(validate_dl)