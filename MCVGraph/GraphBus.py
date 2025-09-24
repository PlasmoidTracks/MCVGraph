# GraphBus.py

from PyQt5.QtCore import QObject, pyqtSignal
from typing import Any, Dict, List
from MCVGraph.EventType import EventType

class GraphEvent:
    def __init__(self, source_type: str, event_type: str, payload: Any) -> None:
        self.source_type: str = source_type
        self.event_type: str = event_type
        self.payload: Any = payload

class GraphBus:
    _instance: "GraphBus" = None

    def __new__(cls) -> "GraphBus":
        """
        Singleton constructor for GraphBus.
        Ensures only one central event bus exists in the application.
        Each client that creates/uses GraphBus gets the same instance,
        with a shared `nodes` list for event distribution.
        """
        if cls._instance is None:
            cls._instance = super(GraphBus, cls).__new__(cls)
            cls._instance.nodes: List[Any] = []
        return cls._instance

    def register(self, handler: Any) -> None:
        print(f"[GraphBus] Registering: {handler.node_type}")
        self.nodes.append(handler)

    def unregister(self, handler: Any) -> None:
        if handler in self.nodes:
            print(f"[GraphBus] Unregistering: {handler.node_type}")
            self.nodes.remove(handler)

    def emit_event(self, source_type: str, event_type: str, payload: Any) -> None:
        """
        Broadcast an event to all registered nodes.
        - Wraps the raw input into a `GraphEvent`.
        - Iterates through the current list of nodes (copied to avoid
          modification during iteration).
        - Calls each node's `on_graph_event` method, letting them react
          to selection, highlight, or other Canvas/Graph events.
        """
        event = GraphEvent(source_type, event_type, payload)
        for node in list(self.nodes):
            node.on_graph_event(event)

class GraphEventClient(QObject):
    signal = pyqtSignal(str, object)
    
    def __init__(self, owner: Any, node_type: str) -> None:
        super().__init__()
        self.owner: Any = owner
        self.node_type: str = node_type
        self.bus: GraphBus = GraphBus()
        self.bus.register(self)
        self.signal.connect(owner.handle_event)

    def disconnect(self) -> None:
        print(f"[GraphBus] Disconnecting: {self.node_type}")
        self.signal.disconnect()
        self.bus.unregister(self)

    def emit_broadcast(self, event_type: str, payload: Any) -> None:
        if event_type == EventType.SUBSET_INDICES and isinstance(payload, dict):
            if "color" not in payload and hasattr(self.owner, "get_color"):
                payload = payload.copy()
                payload["color"] = self.owner.get_color()
        self.bus.emit_event(self.node_type, event_type, payload)

    def on_graph_event(self, event: GraphEvent) -> None:
        self.signal.emit(event.event_type, event.payload)
