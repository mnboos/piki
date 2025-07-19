from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device, AngularServo
from time import sleep

#Device.pin_factory = NativeFactory()
#Device.pin_factory = LGPIOFactory()
Device.pin_factory = PiGPIOFactory()
#Device.pin_factory = RPiGPIOFactory()

min_angle = -170
max_angle = 170

servo_configs = {
	"hd1900mg": {
		"min_angle": -180,
		"max_angle": 180,
		"min_pulse_width": 0.6/1000,
		"max_pulse_width": 2.45/1000,
	}
}

servo = "hd1900mg"
servo_config = servo_configs[servo]

servo_tilt = AngularServo(12, **servo_config)
servo_pan = AngularServo(13, **servo_config)

while True:
    servo_tilt.min()
    servo_pan.min()
    sleep(1)
    servo_tilt.mid()
    servo_pan.mid()
    sleep(1)
    servo_tilt.max()
    servo_pan.max()
    sleep(1)
