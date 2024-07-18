from dataclasses import dataclass
from dataclasses import field
from contextlib import contextmanager
from functools import wraps
from typing import Any
from typing import Mapping
from typing import Callable
from typing import Generic
from typing import Iterable
from typing import ParamSpec
from typing import Type
from typing import TypeVar
from typing import Self
from typing import Generator
from typing import get_type_hints
from typing import Protocol
from typing import runtime_checkable
from typing import NamedTuple

import pytest


T = TypeVar("T", bound=Any)
P = ParamSpec("P")


class DependencyInjectionError(Exception):
    pass


class CircularDependencyError(DependencyInjectionError):
    def __init__(self, obj: Any, dep: Any) -> None:
        message = f"circular dependency between {name(dep)!r} & {name(obj)!r}"
        super().__init__(message)


class DependencyResolutionError(DependencyInjectionError):
    def __init__(self, dep_name: str) -> None:
        message = f"unable to resolve dependency: {dep_name!r}"
        super().__init__(message)


def name(obj: Any) -> str:
    if not hasattr(obj, "__name__"):
        obj = obj.__class__
    return obj.__name__


@runtime_checkable
class Injectable(Protocol):
    __singletone__: bool = False


class Singletone(Injectable):
    __singletone__: bool = True


def is_injectable(obj: Any) -> bool:
    return isinstance(obj, Injectable)


def is_singletone(obj: Any) -> bool:
    if params := getattr(obj, "__dataclass_params__", None):
        if params.frozen:
            return True
    if isinstance(obj, tuple):
        return True
    return getattr(obj, "__singletone__", False)


def set_singletone(obj: Any, value: bool) -> None:
    setattr(obj, "__singletone__", False)


def is_all_singletones(deps: Mapping[str, Any]) -> bool:
    for dep in deps.values():
        if not is_singletone(dep):
            return False
    return True


@dataclass(slots=True, frozen=True)
class Provider(Generic[T]):
    _instances: dict[str, T] = field(default_factory=dict)
    _stack: set[str] = field(default_factory=set)

    @classmethod
    def create(cls, instances: Iterable[T] = ()) -> Self:
        provider = cls()
        for instance in instances:
            provider.register_instance(instance)
        return provider

    def register_instance(self, instance: T) -> None:
        self._instances[name(instance)] = instance

    @contextmanager
    def _circular_dependency_protection(
        self,
        obj: Type[T] | Callable[P, T],
        dep: Type[T],
    ) -> Generator[None, None, None]:
        dep_name = name(dep)
        if dep_name in self._stack:
            raise CircularDependencyError(obj, dep)
        self._stack.add(dep_name)
        yield
        self._stack.discard(dep_name)

    def gather_dependencies(
        self, obj: Type[T] | Callable[P, T]
    ) -> Mapping[str, T]:
        try:
            type_hints = get_type_hints(obj)
        except NameError as err:
            raise DependencyResolutionError(err.name) from err
        kwargs = {}
        for arg, dep in type_hints.items():
            if not is_injectable(dep):
                continue
            with self._circular_dependency_protection(obj, dep):
                kwargs[arg] = self.build(dep)
        return kwargs

    def build(self, cls: Type[T]) -> T:
        if (instance := self._instances.get(name(cls))) is not None:
            return instance
        deps = self.gather_dependencies(cls)
        instance = cls(**deps)
        if is_singletone(instance) and is_all_singletones(deps):
            self.register_instance(instance)
        return instance

    def inject(self, func: Callable[P, Any]) -> Callable[P, Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            deps = self.gather_dependencies(func)
            return func(*args, **kwargs, **deps)

        return wrapper


@dataclass
class Logger(Singletone):
    level: str = "INFO"
    formatter: Callable | None = None


@dataclass
class Client(Injectable):
    logger: Logger
    host: str = "localhost"
    port: int = 8080


@dataclass
class Repo(Injectable):
    client: Client
    logger: Logger


@dataclass
class ApiClient(Injectable):
    host: str = field(default="external")
    port: int = field(default=80)


@dataclass
class Service(Injectable):
    repo: Repo
    logger: Logger
    api: ApiClient


@pytest.fixture
def provider() -> Provider:
    return Provider()


def test_build(provider: Provider) -> None:
    logger = provider.build(Logger)
    assert isinstance(logger, Logger)


def test_build_with_default_values(provider: Provider) -> None:
    logger = provider.build(Logger)
    assert isinstance(logger, Logger)
    assert logger.level == "INFO"


def test_build_with_injection_keeps_all_default_values(
    provider: Provider,
) -> None:
    client = provider.build(Client)
    assert client.host == "localhost"
    assert client.port == 8080


def test_build_dataclass_with_default_factory(provider: Provider) -> None:
    @dataclass
    class ClientWithDefaultLogger(Injectable):
        logger: Logger = field(default_factory=Logger)

    client = provider.build(ClientWithDefaultLogger)
    assert client.logger is provider.build(Logger)


def test_build_with_instance(provider: Provider) -> None:
    logger = Logger()
    provider.register_instance(logger)
    service = provider.build(Service)
    assert isinstance(service, Service)
    assert service.logger is logger


def test_build_raises_if_dependency_missing(provider: Provider) -> None:
    @dataclass
    class WithMissingDependency:
        logger: Logger
        undefined: "Undefined"  # type: ignore # noqa: F821

    with pytest.raises(DependencyResolutionError) as err:
        provider.build(WithMissingDependency)

    assert "unable to resolve dependency: 'Undefined'" in str(err.value)


def test_build_raises_if_circular_dependency(provider: Provider) -> None:
    @dataclass
    class A(Injectable):
        b: "B"

    @dataclass
    class B(Injectable):
        a: A

    A.__annotations__["b"] = B

    with pytest.raises(CircularDependencyError) as err:
        provider.build(A)

    assert "circular dependency between" in str(err.value)
    assert "'A'" in str(err.value)
    assert "'B'" in str(err.value)


def test_build_caches_immutable_instane(provider: Provider) -> None:
    logger = provider.build(Logger)
    assert logger is provider.build(Logger)


def test_build_do_not_cache_mutable_instance(provider: Provider) -> None:
    client = provider.build(Client)
    assert client is not provider.build(Client)


def test_build_do_not_cache_singletone_with_direct_mutable_deps(
    provider: Provider,
) -> None:
    @dataclass
    class ImmutableClient(Singletone):
        logger: Logger
        host: str = "localhost"
        port: int = 8080

    @dataclass
    class MutableRepo(Injectable):
        client: ImmutableClient
        logger: Logger

    @dataclass
    class ImmutableService(Singletone):
        """Singletone has mutable dependency- repo and MUST not be cached."""

        repo: MutableRepo
        logger: Logger

    service = provider.build(ImmutableService)
    assert service is not provider.build(ImmutableService)


def test_build_will_cache_singletone_with_immutable_deps(
    provider: Provider,
) -> None:
    @dataclass
    class ImmutableClient(Singletone):
        logger: Logger
        host: str = "localhost"
        port: int = 8080

    @dataclass
    class ImmutableRepo(Singletone):
        client: ImmutableClient
        logger: Logger

    @dataclass
    class ImmutableService(Singletone):
        repo: ImmutableRepo
        logger: Logger

    service = provider.build(ImmutableService)
    assert service is provider.build(ImmutableService)


def test_build_instance_with_protocol(provider: Provider) -> None:
    @dataclass
    class ProtocolService:
        logger: Logger
        api: ApiClient
        __singletone__: bool = False

    service = provider.build(ProtocolService)
    assert isinstance(service, ProtocolService)


def test_build_will_cache_frozen_dataclass(provider: Provider) -> None:
    @dataclass(frozen=True)
    class FrozenClient:
        logger: Logger

    client = provider.build(FrozenClient)
    assert client is provider.build(FrozenClient)


def test_build_will_cache_namedtuple(provider: Provider) -> None:
    class NamedTupleClient(NamedTuple):
        logger: Logger

    provider = Provider()
    client = provider.build(NamedTupleClient)
    assert client is provider.build(NamedTupleClient)


def test_inject_provides_dependencies(provider: Provider) -> None:
    @provider.inject
    def handler(logger: Logger, service: Service) -> bool:
        assert isinstance(logger, Logger)
        assert isinstance(service, Service)
        return True

    assert handler()  # type: ignore


def test_inject_combines_kwargs(provider: Provider) -> None:
    @provider.inject
    def handler(a: int, b: int, logger: Logger, service: Service) -> int:
        assert isinstance(logger, Logger)
        assert isinstance(service, Service)
        return a + b

    assert handler(a=1, b=2) == 3  # type: ignore


def test_inject_combines_args(provider: Provider) -> None:
    @provider.inject
    def handler(a: int, b: int, logger: Logger, service: Service) -> int:
        assert isinstance(logger, Logger)
        assert isinstance(service, Service)
        return a + b

    assert handler(1, 2) == 3  # type: ignore


def test_inject_provides_depenencies_for_async_function(
    provider: Provider,
) -> None:
    import asyncio

    @provider.inject
    async def handler(
        a: int, b: int, logger: Logger, service: Service
    ) -> int:
        assert isinstance(logger, Logger)
        assert isinstance(service, Service)
        return a + b

    assert asyncio.run(handler(1, 2)) == 3  # type: ignore
