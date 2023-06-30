import dataclasses
from typing import Union, Iterable
import catalogue
from confection import registry, Config
from pydantic import BaseModel

# Create a new registry.
registry.optimizers = catalogue.create("confection", "optimizers", entry_points=False)


# Define a dummy optimizer class.

@dataclasses.dataclass
class MyCoolOptimizer:
    learn_rate: float
    gamma: float


@registry.optimizers.register("my_cool_optimizer.v1")
def make_my_optimizer(learn_rate: Union[float, Iterable[float]], gamma: float):
    return MyCoolOptimizer(learn_rate=learn_rate, gamma=gamma)


if __name__ == "__main__":
    # Load the config file from disk, resolve it and fetch the instantiated optimizer object.
    cfg_str = """
[optimizer]
@optimizers = "my_cool_optimizer.v1"
learn_rate = 0.001
gamma = 1e-8
    """
    config = Config().from_str(cfg_str)
    resolved = registry.resolve(config)
    optimizer = resolved["optimizer"]  # MyCoolOptimizer(learn_rate=0.001, gamma=1e-08)

    print(config, resolved, optimizer)
