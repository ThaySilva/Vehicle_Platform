import time
import RPi.GPIO as GPIO

class ObstacleDetector():

    def __init__(self):
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        self.FRONT_TRIGGER = 13
        self.BACK_TRIGGER = 31
        self.FRONT_ECHO = 15
        self.BACK_ECHO = 33

    def detect_obstacle_front(self):
        GPIO.setup(self.FRONT_TRIGGER, GPIO.OUT)
        GPIO.setup(self.BACK_ECHO, GPIO.IN)
        GPIO.output(self.BACK_TRIGGER, False)

        GPIO.output(self.FRONT_TRIGGER, True)
        time.sleep(0.00001)
        GPIO.output(self.FRONT_TRIGGER, False)
        start_echo = time.time()

        while GPIO.input(self.FRONT_ECHO) == 0:
            start_echo = time.time()

        while GPIO.input(self.FRONT_ECHO) == 1:
            stop_echo = time.time()

        time_elapsed = stop_echo - start_echo
        obstacle_distance = (time_elapsed * 34300) / 2

        return obstacle_distance

    def detect_obstacle_back(self):
        GPIO.setup(self.BACK_TRIGGER, GPIO.OUT)
        GPIO.setup(self.BACK_ECHO, GPIO.IN)
        GPIO.output(self.BACK_TRIGGER, False)

        GPIO.output(self.BACK_TRIGGER, True)
        time.sleep(self.BACK_TRIGGER, True)
        time.sleep(0.00001)
        GPIO.output(self.BACK_TRIGGER, False)
        start_echo = time.time()

        while GPIO.input(self.BACK_ECHO) == 0:
            start_echo = time.time()

        while GPIO.input(self.BACK_ECHO) == 1:
            stop_echo = time.time()

        time_elapsed = stop_echo - start_echo
        obstacle_distance = (time_elapsed * 34300) / 2

        return obstacle_distance