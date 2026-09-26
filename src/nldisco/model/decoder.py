"""Full-window and shift-equivariant SED decoders."""

from typing import Literal

import torch as t
from torch import Tensor, nn
from torch.nn import functional as F


def _activate(x: Tensor, activation: str) -> Tensor:
    if activation == "relu":
        return F.relu(x)
    if activation == "softplus":
        return F.softplus(x)
    if activation == "none":
        return x
    raise ValueError(f"Unknown output activation: {activation}")


class LinearWindowDecoder(nn.Module):
    """Decode one global sparse vector into a complete spike-count window."""

    code_layout = "global"

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        n_neurons: int,
        output_activation: str,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_neurons = n_neurons
        self.output_activation = output_activation
        self.weight = nn.Parameter(t.empty(seq_len, n_neurons, n_features))
        self.bias = nn.Parameter(t.zeros(seq_len, n_neurons))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        self.normalize_dictionary()

    def forward(
        self,
        sparse_acts: Tensor,
        *,
        apply_output_activation: bool = True,
        include_bias: bool = True,
    ) -> Tensor:
        d_sed = sparse_acts.shape[-1]
        reconstruction = t.einsum("bd,snd->bsn", sparse_acts, self.weight[..., :d_sed])
        if include_bias:
            reconstruction = reconstruction + self.bias
        if apply_output_activation:
            reconstruction = _activate(reconstruction, self.output_activation)
        return reconstruction

    @t.no_grad()
    def normalize_dictionary(self) -> None:
        norms = self.weight.norm(dim=(0, 1), keepdim=True).clamp_min(1e-12)
        self.weight.div_(norms)

    @t.no_grad()
    def project_dictionary_grad(self) -> None:
        if self.weight.grad is None:
            return
        dot = (self.weight.grad * self.weight).sum(dim=(0, 1), keepdim=True)
        self.weight.grad.sub_(dot * self.weight)


class TemporalConvDecoder(nn.Module):
    """Decode sparse feature occurrences using shared temporal motif kernels."""

    code_layout = "temporal"

    def __init__(
        self,
        n_features: int,
        n_neurons: int,
        kernel_len: int,
        alignment: Literal["center", "causal"],
        output_activation: str,
    ) -> None:
        super().__init__()
        self.kernel_len = kernel_len
        self.alignment = alignment
        self.output_activation = output_activation
        self.weight = nn.Parameter(t.empty(n_features, n_neurons, kernel_len))
        self.bias = nn.Parameter(t.zeros(n_neurons))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        self.normalize_dictionary()

    def forward(
        self,
        sparse_acts: Tensor,
        *,
        apply_output_activation: bool = True,
        include_bias: bool = True,
    ) -> Tensor:
        seq_len = sparse_acts.shape[1]
        d_sed = sparse_acts.shape[-1]
        code = sparse_acts.transpose(1, 2)
        reconstruction = F.conv_transpose1d(
            code,
            self.weight[:d_sed],
            bias=self.bias if include_bias else None,
        )
        offset = self.kernel_len - 1 if self.alignment == "causal" else self.kernel_len // 2
        reconstruction = reconstruction[..., offset : offset + seq_len]
        reconstruction = reconstruction.transpose(1, 2)
        if apply_output_activation:
            reconstruction = _activate(reconstruction, self.output_activation)
        return reconstruction

    @t.no_grad()
    def normalize_dictionary(self) -> None:
        norms = self.weight.norm(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        self.weight.div_(norms)

    @t.no_grad()
    def project_dictionary_grad(self) -> None:
        if self.weight.grad is None:
            return
        dot = (self.weight.grad * self.weight).sum(dim=(1, 2), keepdim=True)
        self.weight.grad.sub_(dot * self.weight)
