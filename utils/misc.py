from typing import Any, List, Tuple, Union


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax. From stylegan2-ADA"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]
