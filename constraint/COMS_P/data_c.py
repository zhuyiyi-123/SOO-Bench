# from design_bench.task import Task
# from design_bench import make

import torch.utils.data
import torch

class TaskDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, cons):
        self.x = x
        self.y = y
        self.cons = cons
        
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.cons[index]
    
    def __len__(self):
        return len(self.y)

# class DesignBenchTask(Task):
#     def __init__(self,  task_name, **task_kwargs):
#         # use the design_bench registry to make a task
#          self.dbtask = make(task_name, **task_kwargs)

#     @property
#     def is_discrete(self):
#         return self.dbtask.is_discrete

#     @property
#     def dataset_size(self):
#         return self.dbtask.dataset_size
    
#     @property
#     def x(self):
#         return self.dbtask.x

#     @property
#     def y(self):
#         return self.dbtask.y

#     @property
#     def is_normalized_y(self):
#         return self.dbtask.is_normalized_y

#     def map_normalize_x(self):
#         self.dbtask.map_normalize_x()

#     def map_normalize_y(self):
#         self.dbtask.map_normalize_y()

#     def map_denormalize_x(self):
#         self.dbtask.map_denormalize_x()

#     def map_denormalize_y(self):
#         self.dbtask.map_denormalize_y()

#     def map_to_logits(self):
#         self.dbtask.map_to_logits()

#     def normalize_x(self, x):
#         return self.dbtask.normalize_x(x)

#     def normalize_y(self, y):
#         return self.dbtask.normalize_y(y)

#     def denormalize_x(self, x):
#         return self.dbtask.denormalize_x(x)

#     def denormalize_y(self, y):
#         return self.dbtask.denormalize_y(y)

#     def to_logits(self, x):
#         return self.dbtask.to_logits(x)

#     def predict(self, x_batch):
#         return self.dbtask.predict(x_batch)