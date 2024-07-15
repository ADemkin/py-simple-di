import asyncio
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from functools import partialmethod
from typing import Any
from typing import Mapping
from typing import Callable
from typing import Generic
from typing import Iterable
from typing import ParamSpec
from typing import Type
from typing import TypeVar
from typing import Self


T = TypeVar("T", bound=Any)
P = ParamSpec("P")


class DependencyInjectionError(Exception):
    pass


def name(obj: Any) -> str:
    if not hasattr(obj, "__name__"):
        obj = obj.__class__
    return obj.__name__


def annotations(cls: Any) -> Iterable[tuple[str, Type]]:
    return cls.__annotations__.items()


@dataclass(slots=True, frozen=True)
class Provider(Generic[T]):
    _registry: dict[str, partial[T]] = field(default_factory=dict)
    _instances: dict[str, T] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        factories: Iterable[Type] = (),
        instances: Iterable[T] = (),
    ) -> Self:
        provider = cls()
        for factory in factories:
            provider.register_factory(factory)
        for instance in instances:
            provider.register_instance(instance)
        return provider

    def register_instance(self, instance: T) -> None:
        self._instances[name(instance)] = instance

    def get_instance(self, cls: Type[T]) -> T | None:
        return self._instances.get(name(cls))

    def get_factory(self, cls: Type[T]) -> partial[T] | None:
        return self._registry.get(name(cls))

    def register_factory(self, cls: Type[T], *args: P.args, **kwargs: P.kwargs) -> None:
        self._registry[name(cls)] = partial(cls, *args, **kwargs)

    def _gather_deps(self, cls: Type[T] | Callable[P, Any]) -> Mapping[str, T]:
        kwargs = {}
        for arg, dep in annotations(cls):
            if instance := self.get_instance(dep):
                kwargs[arg] = instance
                continue
            if factory := self.get_factory(dep):
                dependencies = self._gather_deps(dep)
                kwargs[arg] = factory(**dependencies)
        return kwargs

    def provide(self, cls: Type[T]) -> T:
        if instance := self.get_instance(cls):
            return instance
        if factory := self.get_factory(cls):
            dependencies = self._gather_deps(cls)
            return factory(**dependencies)
        raise DependencyInjectionError(f"{cls} is not registered")

    def inject(self, func: Callable[P, T]) -> Callable[P, T]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            dependencies = self._gather_deps(func)
            return func(*args, **kwargs, **dependencies)

        return wrapper


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
    logger = Logger()
    provider: Provider = Provider()
    provider.register_instance(logger)
    assert id(logger) == id(provider.provide(Logger))
    print("Instance test passed")

    # test Factory
    provider = Provider()
    provider.register_factory(Logger)
    logger = provider.provide(Logger)
    assert id(logger) != id(provider.provide(Logger))
    print("Factory test passed")

    # test Factory with args
    provider = Provider()
    provider.register_factory(Logger, level="TRACE")  # type: ignore
    logger = provider.provide(Logger)
    assert logger.level == "TRACE"
    print("Factory with args test passed")

    # test provide with level 1 dependency
    logger = Logger()
    provider = Provider()
    provider.register_instance(logger)
    provider.register_factory(Client)
    client = provider.provide(Client)
    assert isinstance(client, Client)
    print("Level 1 dependency test passed")

    # test provide with level 2 dependency
    provider = Provider()
    provider.register_factory(Logger)
    provider.register_factory(Client)
    provider.register_factory(Repo)
    repo = provider.provide(Repo)
    assert isinstance(repo, Repo)
    print("Level 2 dependency test passed")

    # test provide with level 3 dependency
    provider = Provider()
    provider.register_factory(Logger)
    provider.register_factory(Client)
    provider.register_factory(Repo)
    provider.register_factory(ApiClient)
    provider.register_factory(Service)
    service = provider.provide(Service)
    assert isinstance(service, Service)
    print("Level 3 dependency test passed")

    # test provide instance to multiple factories
    logger = Logger()
    provider = Provider()
    provider.register_instance(logger)
    provider.register_factory(Client)
    provider.register_factory(Repo)
    provider.register_factory(ApiClient)
    provider.register_factory(Service)
    service = provider.provide(Service)
    assert id(service.logger) == id(logger)
    repo = provider.provide(Repo)
    assert id(repo.logger) == id(logger)
    client = provider.provide(Client)
    assert id(client.logger) == id(logger)
    print("Provide instance to multiple factories test passed")

    # test inject decorator
    provider = Provider()
    provider.register_factory(Logger)

    @provider.inject
    def ensure_logger(logger: Logger) -> None:
        assert isinstance(logger, Logger)

    ensure_logger()  # type: ignore
    print("Inject decorator test passed")

    # test inject async decorator
    provider = Provider()
    provider.register_factory(Logger)

    @provider.inject
    async def ensure_logger_async(logger: Logger) -> None:
        await asyncio.sleep(0)
        assert isinstance(logger, Logger)
        print('executed')

    asyncio.run(ensure_logger_async())  # type: ignore
    print("Inject async decorator test passed")

    # test provider factory method
    logger = Logger()
    provider = Provider.create(
        factories=[Client, Repo, ApiClient, Service],
        instances=[logger],
    )
    service = provider.provide(Service)
    assert isinstance(service, Service)
    print("Provider factory method test passed")
