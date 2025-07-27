def clamp(value, *, minimum, maximum):
    return max(minimum, min(value, maximum))
