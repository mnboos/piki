from django.shortcuts import redirect, render

from .utils import clamp, shared
from .utils.shared import (
    MotionDetector,
    settings,
)


def index(request):
    request.session["prob_threshold"] = int(round(shared.prob_threshold.value, 2) * 100)
    return render(request, "core/index.html", {"servo_range": list(range(-90, 90))})


def config(request):
    request.session.setdefault("mask_transparency", shared.mask_transparency.value * 100)
    request.session.setdefault("mog2_history", settings.foreground_mask_options.mog2_history.value)
    request.session.setdefault(
        "mog2_var_threshold",
        settings.foreground_mask_options.mog2_var_threshold.value,
    )
    request.session.setdefault(
        "denoise_strength",
        settings.foreground_mask_options.denoise_kernelsize.value,
    )

    if request.method == "POST":
        mask_transparency = int(request.POST.get("mask_transparency"))
        shared.mask_transparency.value = mask_transparency / 100
        request.session["mask_transparency"] = mask_transparency

        mog2_history = int(request.POST.get("mog2_history"))
        settings.foreground_mask_options.mog2_history.value = mog2_history
        request.session["mog2_history"] = mog2_history

        mog2_var_threshold = int(request.POST.get("mog2_var_threshold"))
        settings.foreground_mask_options.mog2_var_threshold.value = mog2_var_threshold
        request.session["mog2_var_threshold"] = mog2_var_threshold

        denoise_strength = int(request.POST.get("denoise_strength"))
        if denoise_strength != 0 and not denoise_strength % 2:
            denoise_strength += 1

        settings.foreground_mask_options.denoise_kernelsize.value = denoise_strength
        request.session["denoise_strength"] = denoise_strength

        shared.motion_detector = MotionDetector()

    return render(request, "core/config.html", {})


def config_ai(request):
    if request.method == "POST":
        threshold = float(request.POST.get("prob_threshold"))
        shared.prob_threshold.value = threshold / 100.0
    return redirect("index")


def move_servo(request):
    """Route to handle servo movement from form submission."""
    if request.method == "POST":
        angle = int(request.POST.get("tilt_position", 0))
        request.session["tilt_position"] = clamp(angle, minimum=-90, maximum=90)

        angle = int(request.POST.get("pan_position", 0))
        request.session["pan_position"] = clamp(angle, minimum=-90, maximum=90)
    return redirect("index")
