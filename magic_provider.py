import asyncio
from dataclasses import dataclass
from dataclasses import field
from functools import partial
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
from typing import get_type_hints


T = TypeVar("T", bound=Any)
P = ParamSpec("P")


class DependencyInjectionError(Exception):
    pass


def name(obj: Any) -> str:
    if not hasattr(obj, "__name__"):
        obj = obj.__class__
    return obj.__name__


@dataclass(slots=True, frozen=True)
class Provider(Generic[T]):
    _instances: dict[str, T] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        instances: Iterable[T] = (),
    ) -> Self:
        provider = cls()
        for instance in instances:
            provider.register_instance(instance)
        return provider

    def register_instance(self, instance: T) -> None:
        self._instances[name(instance)] = instance

    def gather_dependencies(self, obj: Type[T] | Callable[P, Any]) -> Mapping[str, T]:
        try:
            type_hints = get_type_hints(obj)
        except NameError as err:
            message = f"unable to resolve dependency: {err.name!r}"
            raise DependencyInjectionError(message) from err
        return {arg: self.build(dep) for arg, dep in type_hints.items()}

    def build(self, cls: Type[T], *args, **kwargs) -> T:
        if instance := self._instances.get(name(cls)):
            return instance
        dependencies = self.gather_dependencies(cls)
        return cls(*args, **dependencies, **kwargs)


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
        converter: 'Converter'

    try:
        provider.build(Missing)
    except DependencyInjectionError as err:
        assert "unable to resolve dependency: 'Converter'" in str(err)
    else:
        raise AssertionError("expected DependencyInjectionError")
    print("missing dependency test passed")
