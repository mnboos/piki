"""Xbox 360 controller input daemon for manual servo/pump control.

Reads the Xbox 360 wired controller (xpad kernel driver) via evdev,
providing relative-rate joystick control of pan/tilt servos and pump.
"""

import atexit
import logging
import os
import select
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration via environment variables ──────────────────────────────────

GAMEPAD_NAME: str = os.environ.get("GAMEPAD_NAME", "Microsoft X-Box 360 pad")
GAMEPAD_MAX_RATE: float = float(os.environ.get("GAMEPAD_MAX_RATE", "60.0"))
GAMEPAD_DEAD_ZONE: float = float(os.environ.get("GAMEPAD_DEAD_ZONE", "0.55"))
GAMEPAD_STICK_HALF_RANGE: float = float(os.environ.get("GAMEPAD_STICK_HALF_RANGE", "15000.0"))
GAMEPAD_POLL_S: float = float(os.environ.get("GAMEPAD_POLL_S", "0.05"))
GAMEPAD_PUMP_DUTY: float = float(os.environ.get("GAMEPAD_PUMP_DUTY", "100.0"))
GAMEPAD_PUMP_BURST_BTN: int = int(os.environ.get("GAMEPAD_PUMP_BURST_BTN", "304"))  # BTN_SOUTH = A
GAMEPAD_PUMP_BURST_S: float = float(os.environ.get("GAMEPAD_PUMP_BURST_S", "0.5"))
GAMEPAD_TOGGLE_BTN: int = int(os.environ.get("GAMEPAD_TOGGLE_BTN", "315"))  # BTN_START
GAMEPAD_DEBUG: bool = os.environ.get("GAMEPAD_DEBUG", "0") == "1"

# ── Shared state ─────────────────────────────────────────────────────────────

_gamepad_thread: Optional[threading.Thread] = None
_gamepad_stop = threading.Event()

gamepad_connected = threading.Event()
gamepad_enabled = threading.Event()

gamepad_pan = 0.0
gamepad_tilt = 0.0
_gamepad_state_lock = threading.Lock()

# Per-axis calibration queried from the device on connect.
# Dict of code -> (center, half_range)
_axis_cal: dict[int, tuple[float, float]] = {}


def _stick_to_rate(raw_value: int, code: int) -> float:
    """Convert an ABS axis value to rate in deg/s with dead zone."""
    cal = _axis_cal.get(code)
    if cal is None:
        return 0.0
    center, half_range = cal
    if half_range <= 0:
        return 0.0
    deflection = (raw_value - center) / half_range
    ad = abs(deflection)
    if ad < GAMEPAD_DEAD_ZONE:
        return 0.0
    sign = 1.0 if deflection > 0 else -1.0
    scaled = (ad - GAMEPAD_DEAD_ZONE) / (1.0 - GAMEPAD_DEAD_ZONE)
    return sign * scaled * GAMEPAD_MAX_RATE


def _query_axis_calibration(dev) -> dict[int, tuple[float, float]]:
    """Read absinfo for all axes from the device, return code→(center, half_range)."""
    import evdev.ecodes as e  # noqa: PLC0415

    cal: dict[int, tuple[float, float]] = {}
    axes_of_interest = {
        e.ABS_X: "ABS_X (left stick X / pan)",
        e.ABS_Y: "ABS_Y (left stick Y / tilt)",
        e.ABS_RZ: "ABS_RZ (right trigger)",
    }
    for code, label in axes_of_interest.items():
        try:
            absinfo = dev.absinfo(code)
            if absinfo is not None:
                center = (absinfo.min + absinfo.max) / 2.0
                half_range = (absinfo.max - absinfo.min) / 2.0
                cal[code] = (center, half_range)
                logger.info("Axis %s: min=%d max=%d center=%.0f half_range=%.0f",
                            label, absinfo.min, absinfo.max, center, half_range)
        except Exception:
            logger.warning("Axis %s: absinfo not available", label)
    return cal


def _discover_device():
    """Find the Xbox 360 controller by scanning /dev/input/event*."""
    try:
        import evdev  # noqa: PLC0415
    except ImportError:
        logger.warning("evdev not installed — gamepad support unavailable")
        return None

    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
            if GAMEPAD_NAME in dev.name:
                logger.info("Gamepad found: %s at %s", dev.name, path)
                return dev
        except Exception:
            continue
    return None


def _build_status_payload() -> dict:
    with _gamepad_state_lock:
        pan = round(gamepad_pan, 1)
        tilt = round(gamepad_tilt, 1)
    return {
        "connected": gamepad_connected.is_set(),
        "enabled": gamepad_enabled.is_set(),
        "pan": pan,
        "tilt": tilt,
    }


def _gamepad_loop() -> None:
    """Daemon thread: discover controller, read events, command servos/pump."""
    import evdev.ecodes as e  # noqa: PLC0415
    from .engine import activate_pump, deactivate_pump, move_to  # noqa: PLC0415

    global gamepad_pan, gamepad_tilt, _axis_cal  # noqa: PLW0603

    pan_angle = 0.0
    tilt_angle = 0.0
    pump_active = False
    last_tick = time.monotonic()
    dev = None

    # Current axis values — updated by events
    abs_x = 0
    abs_y = 0
    abs_rz = 0

    # Last commanded angles to skip redundant move_to() calls
    _last_pan: Optional[float] = None
    _last_tilt: Optional[float] = None

    while not _gamepad_stop.is_set():
        # ── (Re)connect ──
        if dev is None:
            gamepad_connected.clear()
            _axis_cal.clear()
            dev = _discover_device()
            if dev is None:
                _gamepad_stop.wait(timeout=2.0)
                continue
            _axis_cal = _query_axis_calibration(dev)
            gamepad_connected.set()
            abs_x = 0
            abs_y = 0
            abs_rz = 0
            _last_pan = None
            _last_tilt = None
            last_tick = time.monotonic()
            _publish_status()

        # ── Read events with select() for robust non-blocking poll ──
        events_list: list = []
        try:
            readable, _, _ = select.select([dev.fileno()], [], [], GAMEPAD_POLL_S)
            if readable:
                events_list = list(dev.read())
        except (OSError, IOError) as exc:
            logger.warning("Gamepad read error: %s", exc)
            try:
                dev.close()
            except Exception:
                pass
            dev = None
            _axis_cal.clear()
            gamepad_connected.clear()
            if gamepad_enabled.is_set():
                gamepad_enabled.clear()
                logger.info("Gamepad control DISABLED (device lost)")
            continue

        now = time.monotonic()
        dt = now - last_tick
        last_tick = now

        # ── Process events ──
        if GAMEPAD_DEBUG and events_list:
            logger.info("gamepad raw events: %d events", len(events_list))
        for event in events_list:
            if GAMEPAD_DEBUG:
                logger.info("gamepad event: type=%d code=%d value=%d",
                            event.type, event.code, event.value)

            # Toggle button
            if event.type == e.EV_KEY and event.code == GAMEPAD_TOGGLE_BTN:
                if event.value == 1:
                    if gamepad_enabled.is_set():
                        gamepad_enabled.clear()
                        logger.info("Gamepad control DISABLED (AI tracking resumes)")
                    else:
                        gamepad_enabled.set()
                        from .shared import servo_pan, servo_tilt  # noqa: PLC0415
                        pan_angle = float(servo_pan.value)
                        tilt_angle = float(servo_tilt.value)
                        # Calibrate rest center from current stick position
                        _axis_cal[e.ABS_X] = (float(abs_x), GAMEPAD_STICK_HALF_RANGE)
                        _axis_cal[e.ABS_Y] = (float(abs_y), GAMEPAD_STICK_HALF_RANGE)
                        logger.info("Rest cal: ABS_X center=%d, ABS_Y center=%d", abs_x, abs_y)
                        move_to(pan_angle, tilt_angle)  # immediately disable AI tracking
                        _last_pan = pan_angle
                        _last_tilt = tilt_angle
                        logger.info("Gamepad control ENABLED (pan=%.1f, tilt=%.1f)", pan_angle, tilt_angle)
                    _publish_status()

            # Axis updates
            if event.type == e.EV_ABS:
                if event.code == e.ABS_X:
                    abs_x = event.value
                elif event.code == e.ABS_Y:
                    abs_y = event.value
                elif event.code == e.ABS_RZ:
                    abs_rz = event.value

            # Pump burst button
            if event.type == e.EV_KEY and event.code == GAMEPAD_PUMP_BURST_BTN and event.value == 1:
                if gamepad_enabled.is_set():
                    logger.info("Gamepad: pump burst %.1fs", GAMEPAD_PUMP_BURST_S)
                    activate_pump(GAMEPAD_PUMP_BURST_S, GAMEPAD_PUMP_DUTY)
                elif GAMEPAD_DEBUG:
                    logger.info("Gamepad: pump burst ignored (gamepad not enabled)")

        # ── Apply control if enabled ──
        if gamepad_enabled.is_set():
            pan_rate = _stick_to_rate(abs_x, e.ABS_X)
            tilt_rate = _stick_to_rate(abs_y, e.ABS_Y)

            if GAMEPAD_DEBUG and (pan_rate != 0.0 or tilt_rate != 0.0):
                logger.info("gamepad rates: pan=%.1f deg/s, tilt=%.1f deg/s (abs_x=%d, abs_y=%d)",
                            pan_rate, tilt_rate, abs_x, abs_y)

            pan_angle += pan_rate * dt
            tilt_angle += tilt_rate * dt
            pan_angle = max(-90.0, min(90.0, pan_angle))
            tilt_angle = max(-90.0, min(90.0, tilt_angle))

            if (pan_rate != 0.0 or tilt_rate != 0.0):
                if _last_pan != pan_angle or _last_tilt != tilt_angle:
                    move_to(pan_angle, tilt_angle)
                    _last_pan = pan_angle
                    _last_tilt = tilt_angle

            with _gamepad_state_lock:
                gamepad_pan = pan_angle
                gamepad_tilt = tilt_angle

            # Right trigger = continuous pump
            # Use calibration to determine if trigger is pressed (>10% of range)
            rz_cal = _axis_cal.get(e.ABS_RZ)
            if rz_cal is not None:
                _, rz_half = rz_cal
                trigger_thresh = rz_cal[0] + rz_half * 0.1
            else:
                trigger_thresh = 10  # fallback
            if abs_rz > trigger_thresh:
                if not pump_active:
                    pump_active = True
                    # Scale duty 0-100% across the trigger range
                    if rz_cal is not None:
                        rz_center, rz_half = rz_cal
                        frac = max(0.0, min(1.0, (abs_rz - rz_center) / rz_half))
                    else:
                        frac = abs_rz / 255.0
                    duty = frac * GAMEPAD_PUMP_DUTY
                    activate_pump(0, duty)
            else:
                if pump_active:
                    pump_active = False
                    deactivate_pump()

    # ── Cleanup ──
    if dev is not None:
        try:
            dev.close()
        except Exception:
            pass
    _axis_cal.clear()
    gamepad_connected.clear()
    gamepad_enabled.clear()
    logger.info("Gamepad loop stopped.")


def _publish_status():
    """Publish gamepad status to the WebSocket events channel."""
    try:
        from . import events  # noqa: PLC0415
        events.publish("gamepad_status", _build_status_payload())
        if GAMEPAD_DEBUG:
            logger.debug("gamepad_status published OK")
    except Exception:
        logger.exception("Failed to publish gamepad_status")


def start_gamepad_loop() -> None:
    global _gamepad_thread  # noqa: PLW0603
    if _gamepad_thread is not None and _gamepad_thread.is_alive():
        return
    _gamepad_stop.clear()
    _gamepad_thread = threading.Thread(target=_gamepad_loop, daemon=True, name="piki-gamepad")
    _gamepad_thread.start()
    logger.info("Gamepad loop started.")


def stop_gamepad_loop() -> None:
    global _gamepad_thread  # noqa: PLW0603
    if _gamepad_thread is None:
        return
    _gamepad_stop.set()
    _gamepad_thread.join(timeout=1.5)
    _gamepad_thread = None


@atexit.register
def _cleanup_gamepad() -> None:
    stop_gamepad_loop()
