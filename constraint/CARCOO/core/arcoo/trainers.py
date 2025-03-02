from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from core.utils import sample_langevin
import random
import tensorflow as tf
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DualHeadSurogateTrainer(object):
    def __init__(self, dhs_model, 
                 cons, task, cons_mean=0, cons_std=1,
                 dhs_model_prediction_opt=torch.optim.Adam,
                 dhs_model_energy_opt=torch.optim.Adam,
                 surrogate_lr=0.001,
                 init_m=0.05, ldk=50):

        self.dhs_model = dhs_model
        self.dhs_model_prediction_opt = dhs_model_prediction_opt(self.dhs_model.parameters(), lr=surrogate_lr)
        self.dhs_model_energy_ood_opt = dhs_model_energy_opt(self.dhs_model.parameters(), lr=surrogate_lr)
        self.dhs_model_energy_inf_opt = dhs_model_energy_opt(self.dhs_model.parameters(), lr=surrogate_lr)

        self.cons_std = cons_std
        self.cons_mean = cons_mean
        
        # algorithm hyper parameters
        self.init_m = init_m
        self.ldk = ldk
        self.dhs_model_prediction_loss = nn.MSELoss()
        self.cons = cons
        self.task = task
        self.neg_x_cons = []
        self.pos_x = []
        self.pos_y = []
        self.alpha = 0

    def train(self, dataloder, e_train):
        statistics = defaultdict(list)
        for (x, y, cons) in dataloder:
            x = x.to(dtype=torch.float32)
            y = y.to(dtype=torch.float32)
            # prediction head training     
            cons_de = cons.cpu().numpy()
            m = 0
            j = 0
            for i in range(len(x)):
                a = 0
                for e in range(len(cons[0])):
                    if cons_de[i][e] < 0:
                        a = a + 1
                if e == len(cons[0]) - 1 and a > 0:
                    if j == 0:
                        self.neg_x_cons = x[i].view(1, len(x[0]))
                        neg_cons = torch.sqrt(torch.abs(cons[i])).view(1, len(cons[0]))
                        neg_y = y[i].view(1, 1)
                        j = j + 1
                    else:
                        self.neg_x_cons = torch.cat([self.neg_x_cons, x[i - 1].view(1, len(x[0]))])
                        neg_cons = torch.cat([neg_cons, torch.sqrt(torch.abs(cons[i])).view(1, len(cons[0]))],
                                                dim=0)
                        neg_y = torch.cat([neg_y, y[i].view(1, 1)], dim=0)
                        j = j + 1
                elif e == len(cons[0]) - 1 and a == 0:
                    if m == 0:
                        self.pos_x = x[i].view(1, len(x[0]))
                        self.pos_y = y[i].view(1, 1)
                        m = m + 1
                    else:
                        self.pos_x = torch.cat([self.pos_x, x[i].view(1, len(x[0]))], dim=0)
                        self.pos_y = torch.cat([self.pos_y, y[i].view(1, 1)], dim=0)
                        m = m + 1
            
            self.neg_x_cons = torch.tensor(self.neg_x_cons)
            if e_train:
                # energy head training
                neg_x = sample_langevin(x, self.dhs_model, self.init_m, self.ldk, noise=True) # @TODO: change the para to train_config
                pos_energy = self.dhs_model(self.pos_x)[1]
                neg_energy = self.dhs_model(neg_x)[1]
                energy_loss = pos_energy.mean() - neg_energy.mean()
                energy_loss += torch.pow(pos_energy, 2).mean() + torch.pow(neg_energy, 2).mean()

                energy_loss = energy_loss.mean()
                self.dhs_model_energy_ood_opt.zero_grad()
                energy_loss.backward()
                self.dhs_model_energy_ood_opt.step()

                statistics[f'train/energy_cdloss'] = energy_loss.clone().detach()

                pos_energy = self.dhs_model(self.pos_x)[2]
                neg_energy_cons = self.dhs_model(self.neg_x_cons)[2]
                energy_inf_loss = pos_energy.mean() - (self.alpha * neg_energy_cons).mean()
                energy_inf_loss += torch.pow(pos_energy, 2).mean() + torch.pow(neg_energy_cons, 2).mean()
                energy_inf_loss = energy_inf_loss.mean()
                self.dhs_model_energy_inf_opt.zero_grad()
                energy_inf_loss.backward()
                self.dhs_model_energy_inf_opt.step()
                energy_inf_loss = energy_loss.clone().detach()

                statistics[f'train/energy_cdloss'] = energy_inf_loss.clone().detach().requires_grad_(True)
            
            # prediction head training
            self.dhs_model.train()
            score_pos = self.dhs_model(self.pos_x)[0].to(dtype=torch.float32)
            score_neg = self.dhs_model(self.neg_x_cons)[0].to(dtype=torch.float32)
            mse = self.dhs_model_prediction_loss(score_pos.reshape(len(score_pos), ), self.pos_y.reshape(len(self.pos_y), )).to(dtype=torch.float32)
            mse_neg = score_neg.reshape(len(score_neg), ).mean()
            mse1 = mse.clone().detach().requires_grad_(True)
            mse2 = mse_neg.clone().detach().requires_grad_(True)
            cons_mse = score_neg.mean()
            tmp = torch.sum(neg_cons.to(dtype=torch.float32), dim=1)
            self.alpha = torch.softmax(tmp, dim=0).to(DEVICE)
            statistics[f'oracle{i}/train/mse'] = mse1
            statistics[f'oracle{i}/train/cons_mse'] = mse2
            loss = mse.to(dtype=torch.float32) + (self.alpha * score_neg.reshape(len(score_neg), )).mean().to(dtype=torch.float32)
            statistics[f'oracle{i}/train/total_mse'] = loss.clone().detach()
            self.dhs_model_prediction_opt.zero_grad()
            loss.backward()
            self.dhs_model_prediction_opt.step()

        return statistics

    def validate(self, dataloder):
        i = 0
        statistics = defaultdict(list)
        for (x, y, cons) in dataloder:
            i = i + 1
            self.dhs_model.eval()
            j = 0
            cons_de = (cons.cpu().numpy() * self.cons_std + self.cons_mean)
            for i in range(len(x)):
                for e in range(len(cons[0])):
                    if cons_de[i][e] >= 0:
                        if j == 0:
                            pos_x = x[i].view(1, len(x[0]))
                            pos_y = y[i].view(1, 1)
                            j = j + 1
                        else:
                            pos_x = torch.cat([pos_x, x[i].view(1, len(x[0]))], dim=0)
                            pos_y = torch.cat([pos_y, y[i].view(1, 1)], dim=0)
                            j = j + 1

            score_pos = self.dhs_model(pos_x)[0]
            mse = self.dhs_model_prediction_loss(score_pos, pos_y)
            statistics[f'oracle{i}/val/mse'] = mse

        return statistics

    def launch(self, train_dl, validate_dl, epochs, e_train=True):
        for e in range(epochs):
            self.train(train_dl, e_train)
            self.validate(validate_dl)
