from .interfaces import SharedMemoryObject


class DebugSettings(SharedMemoryObject):
    """The in-memory version of our display settings."""

    show_boxes: bool

    def __init__(self, *, show_boxes: bool = True, _dict_proxy=None):
        super().__init__(show_boxes=show_boxes, _dict_proxy=_dict_proxy)


class AimSettings(SharedMemoryObject):
    """Servo aim configuration — which YOLO classes trigger aiming and whether it's active."""

    target_classes: list   # list of YOLO class name strings (stripped)
    servo_enabled: bool
    target_lock_duration: float  # seconds to hold target before allowing a switch
    aim_confidence: float        # minimum confidence to lock onto a target (0-1)
    pan_invert: bool
    tilt_invert: bool

    def __init__(self, *, target_classes: list, servo_enabled: bool = False,
                 target_lock_duration: float = 3.0, aim_confidence: float = 0.4,
                 pan_invert: bool = False, tilt_invert: bool = False,
                 _dict_proxy=None):
        super().__init__(target_classes=target_classes, servo_enabled=servo_enabled,
                         target_lock_duration=target_lock_duration,
                         aim_confidence=aim_confidence,
                         pan_invert=pan_invert, tilt_invert=tilt_invert,
                         _dict_proxy=_dict_proxy)


class AppSettings(SharedMemoryObject):
    debug_settings: DebugSettings
    aim_settings: AimSettings

    def __init__(self, *, debug_settings: DebugSettings, aim_settings: AimSettings, _dict_proxy=None):
        super().__init__(debug_settings=debug_settings, aim_settings=aim_settings, _dict_proxy=_dict_proxy)
