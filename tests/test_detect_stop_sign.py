import cv2
import picamera
from picamera.array import PiRGBArray
import time
import math

class StopSignDetection(object):

    def __init__(self):
        self.stop_sign = False
        self.cam_y_coordinate = 0

    def detect(self, cascade, grayscale_image, image):

        cascade_object = cascade.detectMultiScale(grayscale_image, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in cascade_object:
            cv2.rectangle(image, (x + 5, y + 5), (x + w - 5, y + h - 5), (255, 255, 255), 2)
            self.cam_y_coordinate = y + h - 5

            if w / h == 1:
                cv2.putText(image, 'STOP', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 2, 255, 2)
                self.stop_sign = True

        return self.cam_y_coordinate

class DistanceToCamera(object):

    def __init__(self):
        self.alpha = 8.0 * math.pi / 180
        self.v0 = 119.865631204
        self.ay = 332.262498472

    def calculate_distance(self, y, h, x_shift, image):
        distance = h / math.tan(self.alpha + math.atan((y - self.v0) / self.ay))
        if distance > 0:
            cv2.putText(image, "%.1fcm" % distance, (image.shape[1] - x_shift, image.shape[0] - 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

        return distance


stop_sign_detection = StopSignDetection()
stop_cascade = cv2.CascadeClassifier('stop_sign_cascade.xml')
distance_to_camera = DistanceToCamera()

stop_sign = 15.5 - 10
distance_stop_sign = 25

camera = picamera.PiCamera()
camera.resolution = (640,360)
camera.framerate = 10
time.sleep(1)
rawCapture = PiRGBArray(camera)

for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    image = rawCapture.array
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cam_y_coordinate = stop_sign_detection.detect(stop_cascade, grayscale_image, image)

    if cam_y_coordinate > 0:
        distance = distance_to_camera.calculate_distance(cam_y_coordinate, stop_sign, 300, image)
        distance_stop_sign = distance

    cv2.imshow('image', image)
    key = cv2.waitKey(1) & 0xFF

    rawCapture.truncate(0)

    if key == ord("q"):
        break

cv2.destroyAllWindows()