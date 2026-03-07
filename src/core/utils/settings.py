from .interfaces import SharedMemoryObject


class DebugSettings(SharedMemoryObject):
    """The in-memory version of our settings."""

    mode: str          # "boxes" | "mask"
    debug_enabled: bool
    render_bboxes: bool

    def __init__(self, *, render_bboxes: bool, debug_enabled: bool = False, mode: str = "boxes", _dict_proxy=None):
        super().__init__(render_bboxes=render_bboxes, debug_enabled=debug_enabled, mode=mode, _dict_proxy=_dict_proxy)


class AppSettings(SharedMemoryObject):
    debug_settings: DebugSettings

    def __init__(self, *, debug_settings: DebugSettings, _dict_proxy=None):
        super().__init__(debug_settings=debug_settings, _dict_proxy=_dict_proxy)
