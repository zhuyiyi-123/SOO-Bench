from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import numpy as np


class Logger(object):

    def __init__(self,
                 log_dir):
        self.img_dir = os.path.join(log_dir, 'images')
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir)

    def record(self, key, value, step):
        if value.size == 1:
            self.writer.add_scalar(key, np.mean(value), global_step=step)
        else:
            self.writer.add_scalar(key + '/max', np.max(value.detach().cpu().numpy()), global_step=step)

    def record_hist(self,
               key,
               value,
               step):
        if value.size == 1:
            self.writer.add_scalar(key, value, global_step=step)
        else:
            self.writer.add_histogram(key, value, global_step=step)

    def record_img_dist(self, x, y, step):
        plt.plot(x.detach().cpu().numpy()[:, 0], y.detach().cpu().numpy()[:, 0],'ro-')
        plt.title('Distribution in Step {}'.format(step))
        plt.xlabel('dis')
        plt.ylabel('value')
        plt.savefig(os.path.join(self.img_dir, 'dist_step_{}.png'.format(step)))
        plt.cla()