
import sys, os
import argparse
import time
import torch
import torch.nn as nn

sys.path.append('..')
import nnutils

sys.path.append('..\\..\\..\\wingoal_utils')
from common import (
    set_log_file,
    log,
    logs
)

set_log_file(os.path.split(__file__)[-1], timestamp=True)


class MyModel(nn.Module):
    def __init__(self, d_model=512, n_heads=8, dropout=0.):
        super(MyModel, self).__init__()
        d_k = d_v = d_model // n_heads
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)


def main():
    # test_get_net_detail()
    test_check_net()


def test_get_net_detail():
    net = MyModel()
    param_count, detail = nnutils.get_net_detail(net)
    print(f'parameters count: {param_count}')
    print(detail)
    assert param_count == 1049600, 'Incorrect parameter count got!'

    print('\n ==========more detail============')
    param_count, detail = nnutils.get_net_detail(net, model_name='my_model', show_param_shape=True)
    print(f'parameters count: {param_count}')
    print(detail)
    assert param_count == 1049600, 'Incorrect parameter count got!'


def test_check_net():
    net = MyModel()
    x = torch.rand(512)

    out, detail = nnutils.check_net(net, x, model_name='my_model')
    print(detail)
    assert out.shape[0] == 512, 'Incorrect output of net got!'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("params", nargs="*")
    args = parser.parse_args()
    if len(args.params) == 0:
        log('executing function [main] ...')
        main()
    else:
        func = args.params[0]
        if func != 'main':
            set_log_file(os.path.split(__file__)[-1], suffix=func, timestamp=True)
        param_list = args.params[1:]
        log('executing function [%s] ...' % func)
        eval(func)(*param_list)
    log('finish executing function!')

