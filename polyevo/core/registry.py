from functools import partial
from importlib import import_module
from inspect import signature
from typing import Callable


_REGISTRY: dict[str, dict[str, Callable]] = {}


def register(stage: str, name: str | None = None):
    def decorate(fn: Callable):
        key = name or fn.__name__
        entries = _REGISTRY.setdefault(stage, {})
        if key in entries:
            raise ValueError(f"Already registered: {stage}:{key}")
        entries[key] = fn
        return fn

    return decorate


def resolve(stage: str, spec):
    if callable(spec):
        return spec
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, dict) and isinstance(spec.get("name"), str):
        params = dict(spec)
        name = params.pop("name")
    else:
        raise ValueError(f"{stage}: expected a registered name or {{name, parameters}}")
    if name.startswith("python:"):
        target = name.removeprefix("python:").strip()
        try:
            module, attr = target.rsplit(":", 1)
            fn = getattr(import_module(module), attr)
        except (ValueError, ImportError, AttributeError) as exc:
            raise ValueError(f"Cannot load hook {name}") from exc
    else:
        try:
            fn = _REGISTRY[stage][name]
        except KeyError as exc:
            raise ValueError(f"Unknown {stage}: {name}") from exc
    if not callable(fn):
        raise ValueError(f"{stage}: {name} is not callable")
    try:
        signature(fn).bind_partial(**params)
    except TypeError as exc:
        raise ValueError(f"{stage}:{name}: {exc}") from exc
    return partial(fn, **params)


def list_ops(stage: str | None = None):
    return {key: sorted(value) for key, value in sorted(_REGISTRY.items())
            if stage is None or key == stage}
