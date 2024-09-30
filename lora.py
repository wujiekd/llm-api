import math
from typing import Optional

import loralib as lora
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LoraLinear(lora.LoRALayer, nn.Module):
    """Replace in-place ops to out-of-place ops to fit gemini. Convert a torch.nn.Linear to LoraLinear.
    """

    def __init__(
        self,
        weight: nn.Parameter,
        bias: Optional[nn.Parameter],
        r: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        fan_in_fan_out: bool = False,    # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
    ):
        nn.Module.__init__(self)
        lora.LoRALayer.__init__(self,
                                r=r,
                                lora_alpha=lora_alpha,
                                lora_dropout=lora_dropout,
                                merge_weights=merge_weights)
        print('[lora] lora_rank:', r)
        print('[lora] lora_alpha:', lora_alpha)
        print('[lora] lora_dropout:', lora_dropout)
        self.weight = weight
        self.bias = bias

        out_features, in_features = weight.shape
        self.in_features = in_features
        self.out_features = out_features

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            # self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        self.infer_with_lora = True

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):

        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Module.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):

        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Module.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                delattr(self, 'lora_A')
                delattr(self, 'lora_B')
            self.merged = True

    def set_infer_lora(self, infer_with_lora=True):
        self.infer_with_lora = infer_with_lora

    def forward(self, x: torch.Tensor):

        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged and self.infer_with_lora:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result = result + (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
        


class LoraLinear_merge(lora.LoRALayer, nn.Module):
    """Replace in-place ops to out-of-place ops to fit gemini. Convert a torch.nn.Linear to LoraLinear.
    """

    def __init__(
        self,
        weight: nn.Parameter,
        bias: Optional[nn.Parameter],
        r: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        fan_in_fan_out: bool = False,    # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
    ):
        nn.Module.__init__(self)
        lora.LoRALayer.__init__(self,
                                r=r,
                                lora_alpha=lora_alpha,
                                lora_dropout=lora_dropout,
                                merge_weights=merge_weights)
        print('[lora] lora_rank:', r)
        print('[lora] lora_alpha:', lora_alpha)
        print('[lora] lora_dropout:', lora_dropout)
        self.weight = weight
        self.bias = bias

        out_features, in_features = weight.shape
        self.in_features = in_features
        self.out_features = out_features

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r

            self.lora_A_mr = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B_mr = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling_mr = self.lora_alpha / self.r

            self.lora_A_rd = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B_rd = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling_rd = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            # self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        self.infer_with_lora = True

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        if hasattr(self, 'lora_A_mr'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A_mr, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_mr)

        if hasattr(self, 'lora_A_rd'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A_rd, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_rd)

    def train(self, mode: bool = True):

        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Module.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):

        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Module.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                delattr(self, 'lora_A')
                delattr(self, 'lora_B')
            self.merged = True

    def set_infer_lora(self, infer_with_lora=True):
        self.infer_with_lora = infer_with_lora

    def set_lora_mode(self, lora_mode=None):
        self.lora_mode = lora_mode

    def forward(self, x: torch.Tensor):

        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged and self.infer_with_lora:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.lora_mode == 'route':
                result = result + (self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
            elif self.lora_mode == 'refine':
                result = result + (self.lora_dropout(x) @ self.lora_A_mr.t() @ self.lora_B_mr.t()) * self.scaling_mr
            elif self.lora_mode == 'read':
                result = result + (self.lora_dropout(x) @ self.lora_A_rd.t() @ self.lora_B_rd.t()) * self.scaling_rd
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)



def lora_linear_wrapper(linear: nn.Linear, lora_rank: int, lora_alpha: float, lora_dropout: float) -> LoraLinear:
    assert lora_rank <= linear.in_features, f'LoRA rank ({lora_rank}) must be less than or equal to in features ({linear.in_features})'
    lora_linear = LoraLinear(linear.weight, linear.bias, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
    return lora_linear


def lora_linear_wrapper_merge(linear: nn.Linear, lora_rank: int, lora_alpha: float, lora_dropout: float) -> LoraLinear:
    assert lora_rank <= linear.in_features, f'LoRA rank ({lora_rank}) must be less than or equal to in features ({linear.in_features})'
    lora_linear = LoraLinear_merge(linear.weight, linear.bias, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=False)
    return lora_linear

def convert_to_lora_recursively(module: nn.Module, lora_rank: int, lora_alpha: float, lora_dropout: float) -> None:
    import pdb
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # pdb.set_trace()
            setattr(module, name, lora_linear_wrapper(child, lora_rank, lora_alpha, lora_dropout))
        else:

            convert_to_lora_recursively(child, lora_rank, lora_alpha, lora_dropout)
def merge_to_lora_recursively(module: nn.Module, lora_rank: int, lora_alpha: float, lora_dropout: float) -> None:
    import pdb
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # pdb.set_trace()
            setattr(module, name, lora_linear_wrapper_merge(child, lora_rank, lora_alpha, lora_dropout))
        else:
            merge_to_lora_recursively(child, lora_rank, lora_alpha, lora_dropout)

def merge_lora(model):
    for module in model.modules():
        if isinstance(module, LoraLinear):
            module.merge_weights = True
            module.eval()

def merge_lora_merge(model):
    for module in model.modules():
        if isinstance(module, LoraLinear_merge):
            module.merge_weights = True
            module.eval()

def print_trainable_parameters(model):
    trainable_model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = sum([np.prod(p.size()) for p in trainable_model_parameters])

    non_trainable_model_parameters = filter(lambda p: not p.requires_grad, model.parameters())
    non_trainable_params = sum([np.prod(p.size()) for p in non_trainable_model_parameters])

    print('############ trainable_params:{} ({:.2f}%), non_trainable_params:{}'.format(trainable_params, trainable_params/non_trainable_params*100,non_trainable_params))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('[trainable layer]:', name)

def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n and 'x_embeddings' not in n and 'y_embeddings' not in n and 'w_embeddings' not in n and 'h_embeddings' not in n:
        # if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, lora.LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError

class LoRAModule(nn.Module):
    """A LoRA module base class. All derived classes should call `convert_to_lora()` at the bottom of `__init__()`.
    This calss will convert all torch.nn.Linear layer to LoraLinear layer.

    Args:
        lora_rank (int, optional): LoRA rank. 0 means LoRA is not applied. Defaults to 0.
        lora_train_bias (str, optional): Whether LoRA train biases.
            'none' means it doesn't train biases. 'all' means it trains all biases. 'lora_only' means it only trains biases of LoRA layers.
            Defaults to 'none'.
    """

    def __init__(self, lora_rank: int = 0, lora_train_bias: str = 'none') -> None:
        super().__init__()
        self.lora_rank = lora_rank
        self.lora_train_bias = lora_train_bias

    def convert_to_lora(self) -> None:
        if self.lora_rank <= 0:
            return
        convert_to_lora_recursively(self, self.lora_rank)
        lora.mark_only_lora_as_trainable(self, self.lora_train_bias)
