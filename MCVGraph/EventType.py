from enum import Enum

class EventType(str, Enum):
    SUBSET_INDICES = "subset_indices"
    CLEAR_HIGHLIGHT = "clear_highlight"
