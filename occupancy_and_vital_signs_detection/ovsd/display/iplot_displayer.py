from abc import ABC, abstractmethod


class IPlotDisplayer(ABC):
    @abstractmethod
    def display(self) -> None: ...
