"""Adapter registry with auto-discovery: every module in this package that defines Adapter subclasses with
non-empty `keys` is registered. To add a model family, just add a module here (no edits to this file)."""
import importlib
import pkgutil

from .base import Adapter

_REG = {}
_LOADED = False


def _discover():
    global _LOADED
    if _LOADED:
        return
    for m in pkgutil.iter_modules(__path__):
        if m.name == "base":
            continue
        mod = importlib.import_module(f"{__name__}.{m.name}")
        for obj in vars(mod).values():
            if isinstance(obj, type) and issubclass(obj, Adapter) and obj is not Adapter and obj.keys:
                for k in obj.keys:
                    _REG[k.lower()] = obj
    _LOADED = True


def get_adapter(key):
    _discover()
    k = key.lower()
    if k not in _REG:
        raise KeyError(f"unknown model key {key!r}; known: {sorted(_REG)}")
    return _REG[k](k)


def list_keys():
    _discover()
    return sorted(_REG)
