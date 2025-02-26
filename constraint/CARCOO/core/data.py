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
