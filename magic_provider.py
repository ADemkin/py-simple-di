from dataclasses import dataclass
from dataclasses import field
from contextlib import contextmanager
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


T = TypeVar("T", bound=Any)
P = ParamSpec("P")


class DependencyInjectionError(Exception):
    pass


class CircularDependencyError(DependencyInjectionError):
    def __init__(self, obj: Type[T], dep: Type[T]) -> None:
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


class Injectable:
    __is_immutable__ = False


class Singletone(Injectable):
    __is_immutable__ = True


def is_injectable(obj: Any) -> bool:
    try:
        return issubclass(obj, Injectable)
    except TypeError:
        return False


def set_not_mutable(obj: Any) -> None:
    setattr(obj, "__is_immutable__", False)


def set_mutable(obj: Any) -> None:
    setattr(obj, "__is_immutable__", True)


def is_immutable(obj: Any) -> bool:
    return getattr(obj, "__is_immutable__", False)


def can_be_cached(obj: Any, deps: Mapping[str, Any]) -> bool:
    if not is_immutable(obj):
        return False
    for dep in deps.values():
        if not is_immutable(dep):
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
        if instance := self._instances.get(name(cls)):
            return instance
        deps = self.gather_dependencies(cls)
        instance = cls(**deps)
        if can_be_cached(cls, deps):
            set_not_mutable(instance)
            self.register_instance(instance)
        else:
            set_mutable(instance)
        return instance


if __name__ == "__main__":

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

    # # test build
    # provider = Provider()
    # logger = provider.build(Logger)
    # assert isinstance(logger, Logger)
    # assert logger.level == "INFO"
    # print("build test passed")

    # # test build keeps default values
    # provider = Provider()
    # client = provider.build(Client)
    # assert client.host == "localhost"
    # assert client.port == 8080
    # logger = client.logger
    # assert logger.level == "INFO"
    # print("build keeps default values test passed")

    # test build dataclass with default factory field
    @dataclass
    class ClientWithDefaultLogger(Injectable):
        logger: Logger = field(default_factory=Logger)

    provider = Provider()
    client = provider.build(ClientWithDefaultLogger)
    assert client.logger is provider.build(Logger)
    print("build with default field test passed")

    # Test injection
    logger = Logger()
    provider = Provider.create([logger])
    service = provider.build(Service)
    assert isinstance(service, Service)
    print("service test passed")

    # Test missing dependency
    @dataclass
    class Missing:
        logger: Logger
        converter: "Converter"  # noqa: F821

    try:
        provider.build(Missing)
    except DependencyInjectionError as err:
        assert "unable to resolve dependency: 'Converter'" in str(err)
    else:
        raise AssertionError(
            "expected DependencyInjectionError - unable to resolve dependency: 'Converter'"
        )
    print("missing dependency test passed")

    # test circular depedency
    @dataclass
    class A(Injectable):
        b: "B"

    @dataclass
    class B(Injectable):
        a: A

    provider = Provider()
    try:
        provider.build(A)
    except DependencyInjectionError as err:
        assert "circular dependency between" in str(err)
        assert "'A'" in str(err)
        assert "'B'" in str(err)
    else:
        raise AssertionError(
            "expected DependencyInjectionError - "
            "circular dependency between 'A' & 'B'"
        )
    print("circular dependency test passed")

    # test cache immutable instance
    provider = Provider()
    logger = provider.build(Logger)
    assert id(logger) == id(provider.build(Logger))
    print("cache immutable instance test passed")

    # test do not cache mutable instance
    provider = Provider()
    client = provider.build(Client)
    assert id(client) != id(provider.build(Client))
    print("do not cache mutable instance test passed")

    # test mutable chain
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
        repo: MutableRepo
        logger: Logger

    provider = Provider()
    service = provider.build(ImmutableService)
    assert id(service) != id(provider.build(ImmutableService))
    print("mutable chain test passed")
