import torch.nn as nn
import torch.nn.functional as F

class MLPSurogateModel(nn.Module):
    def __init__(self, n_i, n_h, n_o):
        super(MLPSurogateModel, self).__init__()
        self.fc1 = nn.Linear(n_i, n_h)
        self.do1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(n_h, n_h)
        self.do2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(n_h, n_h)
        self.do3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(n_h, n_o)

    def forward(self, x):
        out1 = self.fc1(x)
        out1 = self.do1(out1)
        out1 = F.relu(out1)
        
        out2 = self.fc2(out1)
        out2 = self.do2(out2)
        out2 = F.relu(out2) + out1
        
        out3 = self.fc3(out2)
        out3 = self.do3(out3)
        out3 = F.relu(out3) + out2

        out4 = self.fc4(out3)
        
        return out4

class DualHeadSurogateModel(nn.Module):
    def __init__(self, n_i, n_h, n_o):
        super(DualHeadSurogateModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_i, n_h)
        self.do1 = nn.Dropout(0.2)
        
        
        self.predict_fc3 = nn.Linear(n_h, n_h)
        self.predict_do3 = nn.Dropout(0.2)
        self.predict_fc5 = nn.Linear(n_h, n_o)
        
        self.energy_fc3 = nn.Linear(n_h, n_h)
        self.energy_do3 = nn.Dropout(0.2)
        self.energy_fc4 = nn.Linear(n_h, 1)


    def forward(self, x):
        x = self.flatten(x)
        
        out1 = self.fc1(x)
        out1 = self.do1(out1)
        out1 = F.relu(out1)
        out2 = out1

        # the prediction head
        predict_out3 = self.predict_fc3(out2)
        predict_out3 = self.predict_do3(predict_out3)
        predict_out3 = F.relu(predict_out3) + out2

        predict_out = self.predict_fc5(predict_out3) 

        # the energy head
        energy_out3 = self.energy_fc3(out2)
        energy_out3 = self.energy_do3(energy_out3)
        energy_out3 = F.relu(energy_out3) + out2

        energy_out = self.energy_fc4(energy_out3)
        
        return predict_out, energy_out