from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from MCVGraph.EventType import EventType

class GraphBase(ABC):
    """
    Abstract base class for all graph types in MCVGraph.
    Defines the minimal interface to integrate with Canvas and GraphBus.
    """

    def handle_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if event_type == EventType.SUBSET_INDICES:
            indices = np.array(payload["indices"], dtype=int)
            source = payload["source"]
            color = payload.get("color", getattr(self, "color", 'r'))
            self._highlight_indices[source] = (indices, color)
            self._update_plot()

        elif event_type == EventType.CLEAR_HIGHLIGHT:
            source = payload["source"]
            if source in self._highlight_indices:
                del self._highlight_indices[source]
                self._update_plot()

    @abstractmethod
    def clone(self) -> "GraphBase":
        ...

    @abstractmethod
    def add_to(self, plot_item: Any, view_box: Any) -> None:
        ...

    @abstractmethod
    def remove_from(self, plot_item: Any) -> None:
        ...

    @abstractmethod
    def _update_plot(self) -> None:
        ...

    def close(self) -> None:
        if hasattr(self, "graph") and hasattr(self.graph, "disconnect"):
            self.graph.disconnect()

    @abstractmethod
    def _source_name(self) -> str:
        ...
