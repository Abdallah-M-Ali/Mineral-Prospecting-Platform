import torch.nn as nn
import math


class ANNWsNet(nn.Module):
    '''The architecture of WsNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(ANNWsNet, self).__init__()
        self.classifier = nn.Linear(20, outputs)

        self.features = nn.Sequential(
            nn.Linear(inputs, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 20),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

        for modules in [self.features, self.classifier]:
            for l in modules.modules():
                if isinstance(l, nn.Linear):
                    nn.init.normal_(l.weight, std=0.01)
                    nn.init.constant_(l.bias, 0)


    def forward(self, x):
        """Forward pass """
        x = self.features(x)
        x = self.classifier(x)
        return x