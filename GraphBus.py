# GraphBus.py

from PyQt5.QtCore import QObject, pyqtSignal

class GraphEvent:
    def __init__(self, source_type, target_type, event_type, payload):
        self.source_type = source_type
        self.target_type = target_type
        self.event_type = event_type
        self.payload = payload

class GraphBus:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GraphBus, cls).__new__(cls)
            cls._instance.nodes = []
        return cls._instance

    def register(self, handler):
        print(f"[GraphBus] Registering: {handler.node_type}")
        self.nodes.append(handler)

    def unregister(self, handler):
        if handler in self.nodes:
            print(f"[GraphBus] Unregistering: {handler.node_type}")
            self.nodes.remove(handler)

    def emit_event(self, source_type, target_type, event_type, payload):
        event = GraphEvent(source_type, target_type, event_type, payload)
        for node in list(self.nodes):
            node.on_graph_event(event)

class GraphEventClient(QObject):
    signal = pyqtSignal(str, object)

    def __init__(self, owner, node_type):
        super().__init__()
        self.owner = owner
        self.node_type = node_type
        self.pinned = False
        self.bus = GraphBus()
        self.bus.register(self)
        self.signal.connect(owner.handle_event)

    def disconnect(self):
        self.signal.disconnect()
        self.bus.unregister(self)

    def emit(self, target_type, event_type, payload):
        if not self.pinned:
            if event_type == "subset_indices" and isinstance(payload, dict):
                if "color" not in payload and hasattr(self.owner, "get_color"):
                    payload = payload.copy()
                    payload["color"] = self.owner.get_color()
            self.bus.emit_event(self.node_type, target_type, event_type, payload)

    def on_graph_event(self, event):
        if self.pinned:
            return
        if event.target_type == self.node_type:
            self.signal.emit(event.event_type, event.payload)

    def set_pinned(self, state: bool):
        self.pinned = state
