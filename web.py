from flask import Flask, render_template, Response, request, redirect, url_for
from gpiozero import Servo
from picamera2.encoders import JpegEncoder, MJPEGEncoder, Quality
from picamera2 import Picamera2
from picamera2.outputs import FileOutput
import io
from threading import Condition
import time

# Initialize Flask
app = Flask(__name__)

# --- Servo Setup ---
# It's recommended to use a try/except block to ensure the GPIO pins are cleaned up
try:
    # Create a Servo object for GPIO pin 17
    servo = Servo(17, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

    # --- Camera Setup ---
    picam2 = Picamera2()
    # Configure the camera for video streaming
    camera_config = picam2.create_video_configuration(main={"size": (640, 480)}, controls={"ColourGains": (1,1)})
    picam2.configure(camera_config)

    class StreamingOutput(io.BufferedIOBase):
        def __init__(self):
            self.frame = None
            self.condition = Condition()

        def write(self, buf):
            with self.condition:
                self.frame = buf
                self.condition.notify_all()

    
    output = StreamingOutput()
    encoder = MJPEGEncoder()
    picam2.start_recording(JpegEncoder(), FileOutput(output))
    #picam2.start_recording(MJPEGEncoder(), FileOutput(output))

    # --- Web Routes ---
    @app.route('/')
    def index():
        """Video streaming home page."""
        return render_template('index.html')

    def gen_frames():
        """Video streaming generator function."""
        while True:
            with output.condition:
                output.condition.wait()
                frame = output.frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    @app.route('/video_feed')
    def video_feed():
        """Video streaming route. Put this in the 'src' attribute of an <img> tag."""
        return Response(gen_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/move_servo', methods=['POST'])
    def move_servo():
        """Route to handle servo movement from form submission."""
        slider_value = request.form['slider']
        
        # Convert slider value (-100 to 100) to servo value (-1 to 1)
        servo_value = int(slider_value) / 100.0
        servo.value = servo_value
        
        return redirect(url_for('index'))

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, threaded=True)

finally:
    print("Stopping camera and cleaning up GPIO")
    picam2.stop_recording()
    # Any other cleanup can go here
