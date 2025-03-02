import numpy as np
import torch
import torch.autograd as autograd
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Optim(object):
    def __init__(self, config, logger,
                 trainer, init_xt, init_yt,
                 pre_model=None, dhs_model=None):

        self.config = config
        self.logger = logger
        self.trainer = trainer
        self.init_xt = init_xt
        self.init_yt = init_yt
        self.predictive_model = pre_model
        self.dhs_model = dhs_model

    def ackley(self, x):
        dim = len(x[0])
        sum_sq = 0
        sum_cos = 0
        y = []
        for j in range(len(x)):
            for i in range(dim):
                sum_sq += np.sum(x[j][i] ** 2)
                sum_cos += np.sum(np.cos(2 * np.pi * x[j][i]))
            data_y = -20 * np.exp(-0.2 * np.sqrt(sum_sq / dim)) - np.exp(sum_cos / dim) + 20 + np.exp(1)
            y.append(-data_y)
        return y

    def rastrigin02(self, x):  # x:[-5.12,5.12]
        x = np.array(x)
        if len(x.shape) < 2:
            x = x[np.newaxis, :]
        sum = [0.0] * x.shape[0]
        dimension = x.shape[1]
        cons = np.zeros((x.shape[0], 2))
        for i in range(len(x)):
            for j in range(dimension):
                sum[i] += (np.square(x[i][j]) - 10 * np.cos(2 * np.pi * x[i][j]) + 10)
            cons[i][0] = np.sum(np.square(x[i] + 1)) - 10 * dimension  # >=0
            cons[i][1] = np.sum(np.square(x[i] - 1)) - 10 * dimension  # >=0
        sum = np.array(sum)
        return sum, cons

    def ellipsoid01(self, x):  # x:[-5.12,5.12]
        x = np.array(x)
        if len(x.shape) < 2:
            x = x[np.newaxis, :]
        sum = [0.0] * x.shape[0]
        dimension = x.shape[1]
        cons = np.zeros((x.shape[0], 2))
        for i in range(len(x)):
            for j in range(dimension):
                sum[i] += (j + 1) * np.square(x[i][j])
            cons[i][0] = np.mean(x[i]) - 5 / dimension  # >=0
            cons[i][1] = np.sum(x[i][int(dimension / 2):]) / (dimension - int(dimension / 2)) - np.sum(
                x[i][:int(dimension / 2)]) / int(dimension / 2)  # >=0
        sum = np.array(sum)
        return sum, cons

    def add_gaussian_noise(self, data, mean=0, std_dev=0.01):
        noisy_data = np.zeros((100, len(data)))
        for i in range(100):
            noise = np.random.normal(mean, std_dev, data.shape)
            noisy_data[i] = data.cpu().detach().numpy() + noise
        return noisy_data

    def optimize(self, uc, fun, min_x, max_x, ori_x):
        self.uc = uc
        self.dhs_model.eval()
        max_n = 0
        xt = self.init_xt
        solution = xt * (max_x - min_x) + min_x
        score, cons = fun(solution.detach().cpu().numpy())
        # if self.task.is_normalized_y:
        #     init_yt = self.task.denormalize_y(self.init_yt.detach().cpu().numpy())
        #     score = self.task.denormalize_y(score)
        self.logger.record(f"offline_best", torch.tensor(-np.array(score)), 0)
        self.logger.record(f"score", torch.tensor(-np.array(score)), 0)
        score_2 = 0
        for step in range(1, 1 + 1000):
            prediction, energy = self.dhs_model(xt)
            uc_e = self.uc.normalize(energy.detach().cpu().numpy())
            xt = self.optimize_step(xt, 1, uc_e, energy_opt=True, stt=step)
            xtt = (xt * (max_x - min_x) + min_x).cpu().numpy()
            score, cons = fun(xtt)
            q = 0
            score = -np.array(score)
            score_2 = np.zeros((len(score), 1)) * 1000
            score_3 = np.zeros((len(score), 1)) * 1000
            for m in range(len(cons)):
                k = 0
                for e in range(len(cons[0])):
                    if cons[m][e] < 0:
                        k = k + 1
                if k == 0:
                    q = q + 1
                    score_2[q] = score[m]
                if score_2 != []:
                    score_3 = score_2
                    if max_n > np.array(max(score_3)[0]):
                        max_n = max(score_3)
                    for i in range(len(score_3)):
                        if score_3[i] == max_n:
                            best_x = xt[i]

            self.logger.record(f"opt/energy", energy, step - 1)
            self.logger.record(f"opt/risk suppression factor", torch.tensor(uc_e), step - 1)
            self.logger.record(f"opt/prediction", prediction, step)
            self.logger.record(f"feasible rate", torch.tensor(1 - q / len(cons)), step)
            self.logger.record(f"best", torch.tensor(max_n), step)

    def optimize_step(self, xt, steps, uc_e, energy_opt=False, stt=1):
        self.dhs_model.eval()

        for step in range(steps):
            if energy_opt:
                uc_e = torch.tensor(uc_e).to(DEVICE)
                if len(xt.shape) > 2:
                    uc_e = uc_e.expand(xt.shape[0], xt.shape[1] * xt.shape[2])
                    uc_e = torch.reshape(uc_e, xt.shape)
                xt.requires_grad = True
                loss = self.dhs_model.forward(xt)[0]
                grad = autograd.grad(loss.sum(), xt)[0]
                xt = xt + uc_e * grad
            else:
                xt.requires_grad = True
                loss = self.dhs_model(xt)[0]
                grad = autograd.grad(loss.sum(), xt)[0]
                xt = xt + self.init_m * grad

        return xt.detach()

    def optimize_tr(self, xt, steps, grad_scale=1):
        self.dhs_model.eval()
        xt.requires_grad = True

        for step in range(steps):
            loss_p = self.dhs_model(xt)[0]
            grad_p = autograd.grad(loss_p.sum(), xt, retain_graph=True)[0]
            xt_p = xt + grad_scale * grad_p

            loss = loss_p - 0.9 * self.dhs_model(xt_p)[0]
            grad = autograd.grad(loss.sum(), xt)[0]
            xt = xt + grad_scale * grad

        return xt.detach()
