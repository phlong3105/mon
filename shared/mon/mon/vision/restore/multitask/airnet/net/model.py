from net.DGRN import DGRN
from net.encoder import CBDE
from torch import nn


class AirNet(nn.Module):
    def __init__(self, opt):
        super(AirNet, self).__init__()

        # Restorer
        self.R = DGRN(opt)

        # Encoder
        self.E = CBDE(opt)

    def forward(self, x_query, x_key):
        if self.training:
            fea, logits, labels, inter = self.E(x_query, x_key)

            restored = self.R(x_query, inter)

            return restored, logits, labels
        else:
            fea, inter = self.E(x_query, x_query)

            restored = self.R(x_query, inter)

            return restored
