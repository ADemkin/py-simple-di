from dataclasses import dataclass

_registry = {}
_instances = {}


def inject(cls, cached: bool):
    def new_with_cache(_cls):
        return _instances.get(cls.__name__) or cls.__new__(_cls)

    def init_with_injection(self):
        kwargs = {}
        for arg_name, dep in cls.__annotations__.items():
            if factory := _registry.get(dep.__name__):
                kwargs[arg_name] = factory()
        cls.__init__(self, **kwargs)
        if cached:
            _instances[cls.__name__] = self

    return type(
        cls.__name__,
        (cls,),
        {
            "__new__": new_with_cache,
            "__init__": init_with_injection,
        },
    )


def register(cached: bool) -> type:
    def wrapper(cls) -> type:
        cls_injected = _registry[cls.__name__] = inject(cls, cached)
        return cls_injected
    return wrapper


if __name__ == "__main__":
    print(_instances)

    @register(cached=True)
    @dataclass(slots=True, frozen=True, eq=False)
    class Logger:
        level: str = "debug"

    print(_instances)

    @register(cached=True)
    @dataclass(slots=True, frozen=True, eq=False)
    class Client:
        logger: Logger
        host: str = "localhost"
        port: int = 8080

    print(_instances)

    @register(cached=False)
    @dataclass(slots=True, frozen=True, eq=False)
    class Repo:
        logger: Logger
        client: Client

    print(_instances)

    @register(cached=False)
    @dataclass(slots=True, frozen=True, eq=False)
    class Service:
        logger: Logger
        repo: Repo

    print(_instances)
    print("registration done")

    print("service creating")
    service = Service()
    assert isinstance(service, Service)
    print("service created")
    print(_instances)
    repo = service.repo
    assert isinstance(repo, Repo)
    client = repo.client
    assert isinstance(client, Client)
    assert id(service) != id(Service()), f"{id(service)=} != {id(Service())=}"
    assert id(repo) != id(Repo()), f"{id(repo)=} != {id(Repo())=}"
    assert id(client) == id(Client()), f"{id(client)=} != {id(Client())=}"
    assert id(service.logger) == id(Logger())
    print(_instances)
