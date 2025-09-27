import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_main_linear_embedding(nn.Module):
    def __init__(self):
        super(Model_main_linear_embedding,self).__init__()
        self.fc1 = nn.Linear(1,128)
        self.fc2 = nn.Linear(1,128)
        self.fc3 = nn.Linear(1,128)
        self.fc4 = nn.Linear(1,128)

    def forward(self,x1,x2,x3,x4):

        x1_fc = self.fc1(x1)
        x2_fc = self.fc2(x2)
        x3_fc = self.fc3(x3)
        x4_fc = self.fc4(x4)

        x_fc = torch.cat((x1_fc,x2_fc,x3_fc,x4_fc),dim=1)
        return x_fc





    


        


if __name__ == '__main__':
    pass