import numpy as np
import torch
import torch.autograd as autograd


class Optimizer(object):
    def __init__(self, config, task,
                 trainer, init_xt, init_yt,
                 pre_model=None, dhs_model=None):
        
        self.config = config
        self.task = task
        self.trainer = trainer
        self.init_xt = init_xt
        self.init_yt= init_yt
        self.predictive_model = pre_model
        self.dhs_model = dhs_model

    def optimize(self, uc):
        self.uc = uc
        self.dhs_model.eval()
        
        xt = self.init_xt
        solution = xt
        score = np.array(self.task.predict(solution.detach().cpu().numpy())[0])
        scores = [np.max(score)]
        # if self.task.is_normalized_y:
        #     init_yt = self.task.denormalize_y(self.init_yt.detach().cpu().numpy())
        #     score = self.task.denormalize_y(score)
        from tqdm import tqdm

        for step in tqdm(range(1, 1 + self.config['opt_steps']), desc='Optimization'):
            result = self.dhs_model(xt)
            prediction, energy_out, energy_inf_out = result
            energy = energy_out + energy_inf_out
            uc_e = self.uc.normalize(energy.detach().cpu().numpy())
            xt = self.optimize_step(xt, 1, uc_e, self.config['energy_opt'])
            score = np.array(self.task.predict(xt.detach().cpu().numpy())[0])
            # evaluate the solutions found by the model
            # if self.task.is_normalized_y:
            #     score = self.task.denormalize_y(score)
            #     prediction = self.task.denormalize_y(prediction.detach().cpu().numpy())
            scores.append(max(score))
        #     print("score:", min(score), max(score))
        # print(scores)
        return scores

    def optimize_step(self, xt, steps, uc_e, energy_opt=False):
        self.dhs_model.eval()
        
        for step in range(steps):
            if energy_opt:
                uc_e = torch.tensor(uc_e).cuda()
                if len(xt.shape) > 2:
                    uc_e = uc_e.expand(xt.shape[0], xt.shape[1]*xt.shape[2])
                    uc_e = torch.reshape(uc_e, xt.shape)
                xt.requires_grad = True

                loss = self.dhs_model(xt)[0]
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