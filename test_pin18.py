"""Quick test: toggle physical pin 18 HIGH for 2 seconds, then LOW."""
import time

import Hobot.GPIO as GPIO

PIN = 18

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

try:
    GPIO.cleanup([PIN])
except Exception:
    pass

GPIO.setup(PIN, GPIO.OUT)
GPIO.output(PIN, GPIO.LOW)
print(f"Pin {PIN} set LOW — measure now (expect ~0V)")
time.sleep(2)

GPIO.output(PIN, GPIO.HIGH)
print(f"Pin {PIN} set HIGH — measure now (expect ~3.3V)")
time.sleep(5)

GPIO.output(PIN, GPIO.LOW)
print(f"Pin {PIN} set LOW again")

GPIO.cleanup([PIN])
print("Done.")
