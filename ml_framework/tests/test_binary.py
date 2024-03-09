'''
Tests binary operations
- +, -, *, /, **, dot, matmul
'''

import pytest
import random
from math import log10
from torch import zeros
from torch import Tensor as tTensor
from ..src.tensor import Tensor

# testing parameters
mindims = 2
maxdims = 2
minshape = 40
maxshape = 40
multiplier = 50

min_single_sigfig = 2
min_avg_sigfig = 5

class TestTensor:

    def setup_method(self, method):
        ndims = random.randint(mindims,maxdims)
        shape = [random.randint(minshape,maxshape) for i in range(ndims)]
        self.a1 = Tensor.rand(shape, -1, 1) * multiplier
        self.a2 = Tensor.rand(shape, -1, 1) * multiplier
        self.b1 = tTensor(self.a1.data).reshape(self.a1.shape)
        self.b2 = tTensor(self.a2.data).reshape(self.a2.shape)

    def tensors_eq(self, a, b, single_sig_threshold=min_single_sigfig):
        if len(b.shape) == 0: b = b.reshape([1]) # this needs to be here because pytorch handles scalar tensors as 0 dimensional, whereas i say they are 1d
        assert len(a.shape) == len(b.shape), f"ndim mismatch: a.ndim = {len(a.shape)}, b.ndim = {len(b.shape)}"
        assert a.shape == list(b.shape), f"Shape mismatch: a.shape = {a.shape}, b.shape = {b.shape}"
        sig_history = []
        for val_a, val_b in zip(a.data, b.flatten()):
            error = abs(val_a - val_b)
            if error <= 1e-9: continue
            correct_sigfigs = int(log10(abs(val_a/error)))
            if correct_sigfigs < single_sig_threshold:
                return correct_sigfigs, False
            sig_history.append(correct_sigfigs)
        if len(sig_history) == 0: return 1e9, True
        avg_sigs = sum(sig_history)/len(sig_history)
        return avg_sigs, True

    def generic_forward(self, f, name):
        sigs, result = self.tensors_eq(f(self.a1, self.a2), f(self.b1, self.b2))
        assert result, f"{name} forward failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{name} forward failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"
    
    def generic_backward(self, f, name):
        self.a1.init_grad()
        self.a2.init_grad()
        self.b1.requires_grad=True
        self.b2.requires_grad=True
        f(self.a1, self.a2).sum().backward()
        f(self.b1, self.b2).sum().backward()
        sigs, result = self.tensors_eq(self.a1.grad, self.b1.grad) and self.tensors_eq(self.a2.grad, self.b2.grad)  
        assert result, f"{name} backward failed: some element agreed on only {sigs} / {min_single_sigfig} sigs"
        assert sigs >= min_avg_sigfig, f"{name} backward failed: on average agreed on only {sigs} / {min_avg_sigfig} sigs"
    
    basic_ops = [
        (lambda x, y: x + y, "add"),
        (lambda x, y: x - y, "sub"),
        (lambda x, y: x * y, "mul"),
        (lambda x, y: x / y, "div"),
        (lambda x, y: x * x * y + x + y * y / x,                    "combo1"),
        (lambda x, y: (x.relu()+1e-9) ** (y / multiplier).relu(),   "pow non-neg^non-neg"),
        (lambda x, y: (x.relu()+1e-9) ** (y / multiplier),          "pow pos^real"),
        (lambda x, y: (x.flatten()).dot((y.flatten())),             "dot"),
        (lambda x, y: x @ y,                                        "matmul")
    ]

    @pytest.mark.parametrize("f, name", basic_ops)
    def test_binary_forward(self, f, name):
        self.generic_forward(f, name)
    
    @pytest.mark.parametrize("f, name", basic_ops)
    def test_binary_backward(self, f, name):
        self.generic_backward(f, name)

    def test_pow_zero_nonneg_forward(self):
        self.a1 = Tensor.fill(self.a1.shape, 0)
        self.b1 = zeros(self.b1.shape)
        f = lambda x, y: x ** (y.relu())
        self.generic_forward(f, "pow zero^nonneg")
        # not doing the backward version here, because pytorch does weird things

    def test_pow_zero_neg(self):
        self.a1 = Tensor.fill(self.a1.shape, 0)
        with pytest.raises(ZeroDivisionError):
            self.a1 ** (-(self.a2.relu()))

    def test_pow_neg_frac(self):
        self.a1 = Tensor.fill(self.a1.shape, -1)
        self.a2 = Tensor.fill(self.a1.shape, 0.5)
        with pytest.raises(ValueError):
            self.a1 ** self.a2