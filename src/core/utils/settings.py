from .interfaces import SharedMemoryObject


class DebugSettings(SharedMemoryObject):
    """The in-memory version of our settings."""

    debug_enabled: bool
    render_bboxes: bool

    def __init__(self, *, render_bboxes: bool, _dict_proxy=None):
        super().__init__(render_bboxes=render_bboxes, _dict_proxy=_dict_proxy)


class AppSettings(SharedMemoryObject):
    debug_settings: DebugSettings

    def __init__(self, *, debug_settings: DebugSettings, _dict_proxy=None):
        super().__init__(debug_settings=debug_settings, _dict_proxy=_dict_proxy)
