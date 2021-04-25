import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.convNet = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, 5, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    
        self.FC = nn.Sequential(
            nn.Linear(50*6*12, 500),
            nn.ReLU(inplace=True)
        )

        self.output = nn.Linear(502, 2)

    def forward(self, x_in):
        feature = self.convNet(x_in['eye'])
        feature = torch.flatten(feature, start_dim=1)
        feature = self.FC(feature)

        feature = torch.cat((feature, x_in['head_pose']), 1)
        gaze = self.output(feature)

        return gaze

if __name__ == '__main__':
    m = model().cuda()
    '''feature = {"face":torch.zeros(10, 3, 224, 224).cuda(),
                "left":torch.zeros(10,1, 36,60).cuda(),
                "right":torch.zeros(10,1, 36,60).cuda()
              }'''
    feature = {"head_pose": torch.zeros(10, 2).cuda(),
               "eye": torch.zeros(10, 1, 36, 60).cuda(),
               "right": torch.zeros(10, 3, 36, 60).cuda()
               }
    a = m(feature)
    print(m)

