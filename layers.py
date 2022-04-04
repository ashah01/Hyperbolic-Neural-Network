import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.modules.module import Module
from math_utils import PoincareBall
import torch.nn.functional as F


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = PoincareBall().mobius_matvec(drop_weight, x, self.c)
        res = PoincareBall().proj(mv, self.c)
        if self.use_bias:
            bias = self.bias.view(1, -1)
            hyp_bias = PoincareBall().expmap0(bias, self.c)
            hyp_bias = PoincareBall().proj(hyp_bias, self.c)
            res = PoincareBall().mobius_add(res, hyp_bias, self.c)
            res = PoincareBall().proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )

class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        xt = PoincareBall().logmap0(x, c=self.c_in)
        xt = PoincareBall().proj_tan0(xt, c=self.c_out)
        return PoincareBall().proj(PoincareBall().expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

class HNNLayer(nn.Module):
    def __init__(self, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h

class HNN(nn.Module):

    def __init__(self):
        super(HNN, self).__init__()
        self.layers = nn.Sequential(
            HNNLayer(500, 16, torch.tensor([1.]), 0.5, [lambda x: x], 1)
        )

    def encode(self, x, adj):
        x_hyp = PoincareBall().proj(PoincareBall().expmap0(x, c=torch.tensor([1.])), c=torch.tensor([1.]))
        return self.layers.forward(x_hyp)

class Linear(Module):
    """
    Simple Linear layer with dropout.
    """

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act

    def forward(self, x):
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.act(hidden)
        return out

class LinearDecoder(nn.Module):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c):
        super(LinearDecoder, self).__init__()
        self.c = c
        self.input_dim = 16
        self.output_dim = 3 
        self.bias = 1 
        self.cls = Linear(self.input_dim, self.output_dim, 0.5, lambda x: x, self.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = PoincareBall().logmap0(x, self.c)
        return self.cls.forward(h)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )