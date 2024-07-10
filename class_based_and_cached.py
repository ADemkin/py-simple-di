from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Generic
from typing import Iterable
from typing import ParamSpecArgs
from typing import ParamSpecKwargs
from typing import Type
from typing import TypeVar

T = TypeVar("T", bound=Any)


class DependencyInjectionError(Exception):
    pass


def name(obj: Any) -> str:
    if not hasattr(obj, "__name__"):
        obj = obj.__class__
    return obj.__name__


def annotations(cls: Any) -> Iterable[str, Type]:
    return cls.__annotations__.items()


class Registry(Generic[T]):
    # keep all factories in a class variable
    _registry: dict[str, Callable[T]] = {}

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()


class Instance(Generic[T], Registry[T]):
    _instance: T

    def __init__(self, instance: T) -> None:
        self._instance = instance
        self._registry[name(instance)] = self

    def __call__(self) -> T:
        return self._instance

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self._instance.__class__.__name__}"


class Factory(Instance[T], Generic[T]):
    _cls: Type[T]
    _args: ParamSpecArgs
    _kwargs: ParamSpecKwargs

    def __init__(
        self,
        cls: Type[T],
        *args: ParamSpecArgs,
        **kwargs: ParamSpecKwargs,
    ) -> None:
        self._cls = cls
        self._args = args
        self._kwargs = kwargs
        self._registry[name(cls)] = self

    def _get_deps_kwargs(self) -> dict[str, Any]:
        kwargs = {}
        for arg, cls_name in annotations(self._cls):
            if factory := self._registry.get(cls_name):
                kwargs[arg] = factory()
        return kwargs

    def __call__(self, *args: ParamSpecArgs, **kwargs: ParamSpecKwargs) -> T:
        dependencies = self._get_deps_kwargs()
        resolved_args = (*self._args, *args)
        resolved_kwargs = {**dependencies, **self._kwargs, **kwargs}
        try:
            return self._cls(*resolved_args, **resolved_kwargs)
        except TypeError as err:
            raise DependencyInjectionError(err) from err

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self._cls.__name__}>"


class Singleton(Factory, Generic[T]):
    # keep all created instances in a class variable
    _instances: dict[str, T] = {}

    @classmethod
    def clear(cls) -> None:
        cls._instances.clear()

    def __call__(self, *args: ParamSpecArgs, **kwargs: ParamSpecKwargs) -> T:
        key = (name(self._cls), args, tuple(sorted(kwargs.items())))
        is_unhashable_args = False
        try:
            if (instance := self._instances.get(key)) is not None:
                return instance
        except TypeError:
            is_unhashable_args = True
        instance = super().__call__(*args, **kwargs)
        if not is_unhashable_args:
            self._instances[key] = instance
        return instance


if __name__ == "__main__":

    @dataclass
    class Logger:
        level: str = "INFO"

    @dataclass
    class Client:
        logger: Logger
        host: str = "localhost"
        port: int = 8080

    @dataclass
    class Repo:
        client: Client
        logger: Logger

    @dataclass
    class ApiClient:
        host: str = "external"
        port: int = 80

    @dataclass
    class Service:
        repo: Repo
        logger: Logger
        api: ApiClient

    # test Instance
    api_client = ApiClient()
    api_instance_provider = Instance(api_client)
    assert api_instance_provider() == api_client
    Instance.clear()
    print("Instance test passed")

    # test Factory
    logger_factory = Factory(Logger)
    logger = logger_factory()
    assert isinstance(logger, Logger)
    assert id(logger_factory()) != id(logger)
    print("Factory test passed")
    Factory.clear()

    # test Singleton
    Singleton(Logger)
    client_factory = Singleton(Client)
    client = client_factory()
    assert isinstance(client, Client)
    assert id(client_factory()) == id(client)
    print("Singleton test passed")
    Singleton.clear()

    # test Singleton with factory args
    logger_factory = Singleton(Logger)
    default_logger = logger_factory()
    error_logger = logger_factory(level="ERROR")
    assert id(error_logger) != id(default_logger)
    print("Singleton with factory args test passed")
    Singleton.clear()

    # test Singleton with default args
    debug_logger_factory = Singleton(Logger, level="DEBUG")
    debug_logger = debug_logger_factory()
    assert debug_logger.level == "DEBUG"
    print("Singleton with default args test passed")
    Singleton.clear()

    # test Singleton default args overriding injection
    Singleton(Logger)
    logger = Logger()
    client_factory = Singleton(Client, logger=logger)
    client = client_factory()
    assert id(client.logger) == id(logger)
    Singleton.clear()
    print("Singleton default args overriding injection test passed")

    # test Singleton with factory args overriding injection
    logger_factory = Singleton(Logger)
    debug_logger = Logger(level="DEBUG")
    client_factory = Singleton(Client)
    client = client_factory(logger=debug_logger)
    assert id(client.logger) == id(debug_logger)
    Singleton.clear()
    print("Singleton with factory args overriding injection test passed")

    # test ingleton with factory args overriding default args
    debug_logger_factory = Singleton(Logger, level="DEBUG")
    debug_logger = debug_logger_factory(level="ERROR")
    assert debug_logger.level == "ERROR"
    print("Singleton with factory args overriding default args test passed")
    Singleton.clear()

    # test injection
    api_client = ApiClient()
    Instance(api_client)
    Singleton(Logger)
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
    Factory.clear()
    Singleton.clear()

    # test cached injection
    api_client = ApiClient()
    Instance(api_client)
    logger_factory = Singleton(Logger)
    client_factory = Singleton(Client)
    repo_factory = Factory(Repo)
    service_factory = Factory(Service)
    service = service_factory()
    assert isinstance(service.logger, Logger)
    repo = repo_factory()
    assert isinstance(repo.logger, Logger)
    assert id(service.logger) == id(repo.logger)
    Factory.clear()
    Singleton.clear()
    print("Cached injection test passed")

    # test missing dependency
    try:
        Factory(Client)()
    except DependencyInjectionError as err:
        assert "missing 1 required positional argument: 'logger'" in str(err)
    else:
        assert False, "did not raise"
    Factory.clear()
    print("Missing dependency test passed")
