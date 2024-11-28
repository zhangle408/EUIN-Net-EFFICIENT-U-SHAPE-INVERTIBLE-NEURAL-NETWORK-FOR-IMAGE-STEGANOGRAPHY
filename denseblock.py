from rrdb_denselayer_deepmih import *
import config as c


class Dense(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, input, output):
        super(Dense, self).__init__()

        self.dense = ResidualDenseBlock_out(input, output, nf=c.nf, gc=c.gc)

    def forward(self, x):
        out = self.dense(x)

        return out


