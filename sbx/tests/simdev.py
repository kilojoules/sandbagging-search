"""CPU simulation of a model sharded over two CUDA devices, to exercise sbx.core's device handling without a GPU.

Fake devices are real `torch.device("cuda:N")` objects (constructible on a CPU build).  Inside `SimMode` every tensor carries a fake device tag
(class DT) and every torch op follows PyTorch's rule: all tensor operands must be on ONE device (a 0-dim CPU tensor and a CPU INDEX tensor are the
two exceptions PyTorch allows), otherwise a RuntimeError like the real one is raised; `.numpy()` of a "cuda" tensor raises; `.to(device)` / factory
functions with device= retag.  Real arithmetic runs on the CPU underneath.  ShardedModel wraps a tiny HF model and returns hidden states / logits
tagged the way accelerate's dispatch would leave them (see `io_same_device`)."""
from __future__ import annotations

import types

import torch
from torch.overrides import TorchFunctionMode

CPU = torch.device("cpu")
PASSTHROUGH = [0]          # > 0: run ops untouched (the real forward of the wrapped tiny model)


def _is_cuda(d):
    return isinstance(d, torch.device) and d.type == "cuda"


class DT(torch.Tensor):
    """A tensor living on a FAKE cuda device `fdev` (the data is on the CPU)."""
    __torch_function__ = torch._C._disabled_torch_function_impl

    @property
    def device(self):
        return self.fdev


def tag(t, dev):
    if not isinstance(t, torch.Tensor) or not _is_cuda(dev):
        return t
    x = torch.Tensor._make_subclass(DT, t.detach())
    x.fdev = dev
    return x


def fdev(t):
    return getattr(t, "fdev", CPU)


def _flat(x):
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _flat(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from _flat(y)


def _split_device(args, kwargs):
    dev, new_args, new_kwargs = None, [], {}
    for a in args:
        if _is_cuda(a):
            dev = a
            continue
        if isinstance(a, str) and a.startswith("cuda"):
            dev = torch.device(a)
            continue
        new_args.append(a)
    for k, v in kwargs.items():
        if k == "device":
            if isinstance(v, str):
                v = torch.device(v)
            if _is_cuda(v):
                dev = v
                continue
        new_kwargs[k] = v
    return dev, tuple(new_args), new_kwargs


def _retag(res, dev):
    if isinstance(res, torch.Tensor):
        return tag(res, dev) if _is_cuda(dev) else (res.as_subclass(torch.Tensor) if isinstance(res, DT) else res)
    if isinstance(res, (tuple, list)):
        return type(res)(_retag(r, dev) for r in res) if not hasattr(res, "_fields") else type(res)(*[_retag(r, dev) for r in res])
    return res


class SimMode(TorchFunctionMode):
    log: list = []

    def __torch_function__(self, func, types_, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if PASSTHROUGH[0] or func is torch.device:
            return func(*args, **kwargs)
        name = getattr(func, "__name__", str(func))
        want, a2, k2 = _split_device(args, kwargs)           # a target fake device (.to(dev), factory device=dev)
        tens = list(_flat(a2)) + list(_flat(list(k2.values())))
        if name == "__getitem__":
            self_t, idx = a2[0], a2[1]
            for it in _flat(idx):
                if fdev(it) != CPU and fdev(it) != fdev(self_t):
                    raise RuntimeError(f"indices should be either on cpu or on the same device as the indexed tensor ({fdev(self_t)}), got {fdev(it)}")
            res = func(*a2, **k2)
            return _retag(res, fdev(self_t))
        if name == "numpy":
            if _is_cuda(fdev(a2[0])):
                raise TypeError(f"can't convert {fdev(a2[0])} device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.")
        devs = {fdev(t) for t in tens if not (fdev(t) == CPU and t.dim() == 0)}
        if len(devs) > 1:
            raise RuntimeError(f"Expected all tensors to be on the same device, but found at least two devices, {sorted(map(str, devs))}! (in {name})")
        out_dev = next(iter(devs)) if devs else CPU
        if name == "cpu":
            res = func(*a2, **k2)
            return _retag(res.clone() if isinstance(res, torch.Tensor) else res, CPU)
        if want is not None:                                  # .to(fake dev) / factory(device=fake dev)
            res = func(*a2, **k2)
            if isinstance(res, torch.Tensor) and tens and res is a2[0]:
                res = res.clone()
            return _retag(res, want)
        res = func(*a2, **k2)
        if isinstance(res, torch.Tensor) and tens and any(res is t for t in tens):     # in-place op returned its (already tagged) operand
            return res
        return _retag(res, out_dev)


class ShardedModel:
    """Wraps a tiny HF causal LM.  Layer i (1-based hidden_states index) lives on cuda:0 for i <= split, else cuda:1; embeddings (index 0) on
    cuda:0; the last entry (post-final-norm) and the logits on cuda:1.  io_same_device=True mimics accelerate's top-level
    AlignDevicesHook(io_same_device=True): every returned tensor is moved back to the input device cuda:0.  Inputs are moved to cuda:0 (accelerate's
    pre_forward does that); a tensor on cuda:1 is moved silently, as the hook would."""

    def __init__(self, model, split, io_same_device):
        object.__setattr__(self, "_m", model)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "io_same_device", io_same_device)
        object.__setattr__(self, "calls", [])

    def __getattr__(self, k):
        return getattr(self._m, k)

    @property
    def device(self):
        return torch.device("cuda:0")

    def __call__(self, **enc):
        PASSTHROUGH[0] += 1
        try:
            plain = {k: (v.as_subclass(torch.Tensor) if isinstance(v, DT) else v) for k, v in enc.items()}
            out = self._m(**plain)
        finally:
            PASSTHROUGH[0] -= 1
        nh = len(out.hidden_states) if out.hidden_states is not None else 0
        hs = None
        if nh:
            hs = []
            for i, h in enumerate(out.hidden_states):
                d = torch.device("cuda:0") if (i <= self.split and i != nh - 1) else torch.device("cuda:1")
                hs.append(tag(h, torch.device("cuda:0") if self.io_same_device else d))
            hs = tuple(hs)
        lg = tag(out.logits, torch.device("cuda:0") if self.io_same_device else torch.device("cuda:1"))
        return types.SimpleNamespace(hidden_states=hs, logits=lg)
