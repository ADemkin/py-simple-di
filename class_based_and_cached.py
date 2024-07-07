from __future__ import annotations

from typing import Any, Type, Generic, TypeVar, ParamSpecArgs, ParamSpecKwargs, Iterable
from dataclasses import dataclass

T = TypeVar("T", bound=Any)


def name(obj: Any):
    if not hasattr(obj, '__name__'):
        obj = obj.__class__
    return obj.__name__


def annotations(cls: Any) -> Iterable[str, Type]:
    return cls.__annotations__.items()


class Factory(Generic[T]):
    # keep all factories in a class variable
    _registry: dict[str, Factory[T]] = {}

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()

    _cls: Type[T]

    def __init__(self, cls: Type[T]) -> None:
        self._cls = cls
        self._registry[name(cls)] = self

    def _inject(self) -> dict[str, Any]:
        kwargs = {}
        for arg, cls_name in annotations(self._cls):
            if factory := self._registry.get(cls_name):
                kwargs[arg] = factory()
        return kwargs

    def __call__(self, *args: ParamSpecArgs, **kwargs: ParamSpecKwargs) -> T:
        dependencies = self._inject()
        dependencies.update(**kwargs)
        return self._cls(*args, **dependencies)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self._cls.__name__}>"


class Singleton(Factory, Generic[T]):
    # keep all created instances in a class variable
    _instances: dict[str, T] = {}

    @classmethod
    def clear(cls) -> None:
        cls._instances.clear()

    def __call__(self, *args: ParamSpecArgs, **kwargs: ParamSpecKwargs) -> T:
        if (instance := self._instances.get(name(self._cls))) is not None:
            return instance
        # TODO: check if cached version has same arguments
        self._instances[name(self._cls)] = instance = super().__call__(*args, **kwargs)
        return instance


if __name__ == '__main__':
    @dataclass
    class Client:
        host: str = "localhost"
        port: int = 8080

    @dataclass
    class Repo:
        client: Client

    @dataclass
    class Service:
        repo: Repo

    # test Factory
    client_factory = Factory(Client)
    client = client_factory()
    assert isinstance(client, Client)
    assert id(client_factory()) != id(client)
    print("Factory test passed")
    Factory.clear()

    # test Singleton
    client_factory = Singleton(Client)
    client = client_factory()
    assert isinstance(client, Client)
    assert id(client_factory()) == id(client)
    print("Singleton test passed")
    Singleton.clear()

    # test injection
    client_factory = Singleton(Client)
    repo_factory = Factory(Repo)
    service_factory = Factory(Service)
    service = service_factory()
    assert isinstance(service, Service)
    repo = service.repo
    assert isinstance(repo, Repo)
    client = repo.client
    assert isinstance(client, Client)
    print("Injection test passed")
