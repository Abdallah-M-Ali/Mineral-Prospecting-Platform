import torch.nn as nn
import math
from ANN.utils.BBBlayers import BBBConv2d, BBBLinearFactorial


class BBBWsNet(nn.Module):
    '''The architecture of WsNet with Bayesian Layers'''

    def __init__(self, outputs, inputs):
        super(BBBWsNet, self).__init__()

        self.q_logvar_init = 0.05
        self.p_logvar_init = math.log(0.05)
        self.classifier = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init, 50, outputs)

        self.fc1 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init, inputs, 100)
        self.soft1 = nn.Softplus()

        self.fc2 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init, 100, 50)
        self.soft2 = nn.Softplus()

        # self.fc3 = BBBLinearFactorial(self.q_logvar_init, self.p_logvar_init, 50, 20)
        # self.soft3 = nn.Softplus()

        layers = [self.fc1, self.soft1, self.fc2, self.soft2, self.classifier]
        # layers = [self.fc1, self.soft1, self.fc2, self.soft2, self.fc3, self.soft3, self.classifier]

        self.layers = nn.ModuleList(layers)

    def probforward(self, x):
        'Forward pass with Bayesian weights'
        kl = 0
        for layer in self.layers:
            if hasattr(layer, 'convprobforward') and callable(layer.convprobforward):
                x, _kl, = layer.convprobforward(x)
                kl += _kl

            elif hasattr(layer, 'fcprobforward') and callable(layer.fcprobforward):
                x, _kl, = layer.fcprobforward(x)
                kl += _kl
            else:
                x = layer(x)
        logits = x
        return logits, kl
