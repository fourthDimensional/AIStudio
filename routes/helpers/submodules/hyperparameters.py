from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Union

from skopt.space import Real, Integer, Categorical

class Hyperparameter(ABC):
    """
    Base class for hyperparameters.
    """
    def __init__(self, name: str, value: Any, value_range: Union[Tuple[float, float], List[Any]]) -> None:
        """
        Args:
            name: Name of the hyperparameter.
            value: The current value of the hyperparameter.
            value_range: The allowed range for this hyperparameter.
                For Real/Integer parameters, this should be a tuple (min, max).
                For categorical parameters, this should be a list of options.
        """
        self.name = name
        self.value = value
        self.range = value_range
        self.default = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, value={self.value})"

    def __str__(self) -> str:
        return f"{self.name}: {self.value}"

    def get_value(self) -> Any:
        return self.value

    def set_value(self, value: Any) -> None:
        if not self.range[0] <= value <= self.range[1]:
            raise ValueError(f"Value {value} is out of range for {self.name}")
        self.value = value

    def restore_default(self) -> None:
        self.value = self.default

    @abstractmethod
    def to_skopt_space(self) -> Any:
        """
        Convert the hyperparameter to a scikit-opt space object.
        Must be implemented by subclasses.
        """
        pass

class RealHyperparameter(Hyperparameter):
    def __init__(self, name: str, value: float, value_range: Tuple[float, float]) -> None:
        super().__init__(name, value, value_range)

    def to_skopt_space(self) -> Real:
        return Real(self.range[0], self.range[1])

class IntegerHyperparameter(Hyperparameter):
    def __init__(self, name: str, value: int, value_range: Tuple[int, int]) -> None:
        super().__init__(name, value, value_range)

    def to_skopt_space(self) -> Integer:
        return Integer(self.range[0], self.range[1])

class CategoricalHyperparameter(Hyperparameter):
    def __init__(self, name: str, value: Any, value_range: List[Any]) -> None:
        super().__init__(name, value, value_range)

    def to_skopt_space(self) -> Categorical:
        return Categorical(self.range)
