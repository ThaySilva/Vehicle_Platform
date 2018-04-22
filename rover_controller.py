#!/usr/bin/env python
__author__ = "Thaynara Silva"
__copyright__ = "Copyright 2018, Software Development Final Year Project"
__version__ = "1.0"
__date__ = "19/04/2018"

import cv2
import distance_sensor_data_collector as collector
import math
import numpy as np
import picamera
from picamera.array import PiRGBArray
import re
import RPi.GPIO as GPIO
import serial
import socket
import sys
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import threading
import time

# Setup the GPIO output pins wired to the motors
GPIO.setwarnings(False)
GPIO.cleanup()
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT)
GPIO.setup(11, GPIO.OUT)
GPIO.setup(29, GPIO.OUT)
GPIO.setup(35, GPIO.OUT)

# Keep the power of the motors off, waiting for command
GPIO.output(7, False)
GPIO.output(11, False)
GPIO.output(29, False)
GPIO.output(35, False)

class RoverController(object):
    
    def __init__(self):
        self.BACK_MOTOR_FORWARD = 7
        self.BACK_MOTOR_REVERSE = 11
        self.FRONT_MOTOR_LEFT = 29
        self.FRONT_MOTOR_RIGHT = 35

    def steer_rover(self, direction):
        if np.array_equal(direction, [0,1,0]):
            GPIO.output(self.BACK_MOTOR_REVERSE, False)
            GPIO.output(self.BACK_MOTOR_FORWARD, True)
        elif np.array_equal(direction, [1,0,0]):
            GPIO.output(self.FRONT_MOTOR_RIGHT, False)
            GPIO.output(self.FRONT_MOTOR_LEFT, True)
            GPIO.output(self.BACK_MOTOR_FORWARD, True)
        elif np.array_equal(direction, [0,0,1]):
            GPIO.output(self.FRONT_MOTOR_LEFT, False)
            GPIO.output(self.FRONT_MOTOR_RIGHT, True)
            GPIO.output(self.BACK_MOTOR_FORWARD, True)
        else:
            self.stop_rover()

    def stop_rover(self):
        GPIO.output(self.BACK_MOTOR_REVERSE, False)
        GPIO.output(self.BACK_MOTOR_FORWARD, False)
        GPIO.output(self.FRONT_MOTOR_LEFT, False)
        GPIO.output(self.FRONT_MOTOR_RIGHT, False)

class NeuralNetwork(object):

    def __init__(self):
        conv_network = input_data(shape=[None, 28, 28, 1], name='input')
        
        conv_network = conv_2d(conv_network, 32, 2, activation='relu')
        conv_network = max_pool_2d(conv_network, 2)
        
        conv_network = conv_2d(conv_network, 64, 2, activation='relu')
        conv_network = max_pool_2d(conv_network, 2)
        
        conv_network = fully_connected(conv_network, 28, activation='relu')
        conv_network = dropout(conv_network, 0.8)
        
        conv_network = fully_connected(conv_network, 3, activation='softmax')
        conv_network = regression(conv_network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')
        
        self.model = tflearn.DNN(conv_network)

    def load_model(self):
        self.model.load('model/cnn_model.model')

    def predict_direction(self, image):
        prediction = np.round(self.model.predict(image)[0])
        return prediction

class ObjectDetector(object):

    def __init__(self):
        self.cam_y_coordinate = 0

    def detect_object(self, classifier, grayscale_image, image):
        classifier_object = classifier.detectMultiScale(grayscale_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in classifier_object:
            cv2.rectangle(image, (x + 5, y + 5), (x + w - 5, y + h - 5), (255, 255, 255), 2)
            self.cam_y_coordinate = y + h - 5

            if w / h == 1:
                cv2.putText(image, 'STOP', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 2, 255, 2)
            
        return self.cam_y_coordinate

class ObjectDistanceCollector(object):

    def __init__(self):
        self.alpha = 8.0 * math.pi / 180
        self.v0 = 119.865631204
        self.ay = 332.262498472

    def collect_distance(self, y, h, x_shift, image):
        distance = h / math.tan(self.alpha + math.atan((y - self.v0) / self.ay))
        if distance > 0:
            cv2.putText(image, "%.1fcm" % distance, (image.shape[1] - x_shift, image.shape[0] - 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)

        return distance

class GenuinoDataHandler(object):

    def handle(self, connection):
        while True:
            ser = serial.Serial('/dev/ttyACM0', 9600, 8, 'N', 1, timeout=5)
            while True:
                if ser.inWaiting() > 0:
                    data = ser.readline()
                    data = re.sub('\r$', '', data)
                    data = re.sub('\n$', '', data)
                    metrics = ""

                    for info in data.split(','):
                        key, value = info.split(':')
                        if key == "orientation":
                            metrics += ",orientation:" + value
                        elif key == "impact":
                            metrics += ",impact:" + value
                        elif key == "latitude":
                            metrics += ",latitude" + value
                        elif key == "longitude":
                            metrics += ",longitude" + value
                    
                    connection.send(metrics)

class CameraFeedHandler(object):

    rover_controller = RoverController()
    neural_network = NeuralNetwork()
    neural_network.load_model()
    object_detector = ObjectDetector()
    stop_sign_classifier = cv2.CascadeClassifier('classifier/stop_sign_cascade.xml')
    stop_sign = 15.5 - 10
    distance_collector = ObjectDistanceCollector()
    distance_to_stop_sign = 30
    start_stop_time = 0
    stop_stop_time = 0
    stop_duration = 0
    stop_sign_active = False
    stop_sign_flag = False
    vertices = np.array([[0,240],[0,140],[140,120],[180,120],[320,140],[320,240]], np.int32)
    obstacle_detector = collector.ObstacleDetector()
    distance_to_obstacle = 35

    def mask_image(self, image, vertices):
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, vertices, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
    
    def handle(self, connection):
        # Setup camera variable
        camera = picamera.PiCamera()
        camera.resolution = (320, 240)
        camera.framerate = 10
        rawCapture = PiRGBArray(camera)

        while True:
            for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
                image = rawCapture.array
                grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                canny_image = cv2.Canny(grayscale_image, threshold1=200, threshold2=300)
                roi_image = self.mask_image(canny_image, [self.vertices])
                resized_image = cv2.resize(roi_image, (28, 28))
                prediction_image = resized_image.reshape([-1, 28, 28, 1])

                # Uncomment these lines to watch the live camera feed for debugging purposes
#                cv2.imshow('image', image)
#                cv2.waitKey(0) & 0xFF

                cam_y_coordinate = self.object_detector.detect_object(self.stop_sign_classifier, grayscale_image, image)
                
                if cam_y_coordinate > 0:
                    print 'Stop Sign detected!'
                    distance = self.distance_collector.collect_distance(cam_y_coordinate, self.stop_sign, 300, image)
                    self.distance_to_stop_sign = distance
                    
                self.distance_to_obstacle = self.obstacle_detector.detect_obstacle_front()

                prediction = self.neural_network.predict_direction(prediction_image)

                if self.distance_to_obstacle < 30:
                    print 'Stopping Rover'
                    self.rover_controller.stop_rover()
                elif 0 < self.distance_to_stop_sign < 25 and self.stop_sign_active:
                    print 'Stopping Rover'
                    self.rover_controller.stop_rover()
                    if self.stop_sign_flag is False:
                        self.start_stop_time = cv2.getTickCount()
                        self.stop_sign_flag = True
                    self.stop_stop_time = cv2.getTickCount()

                    self.stop_duration = (self.start_stop_time - self.start_stop_time) / cv2.getTickFrequency()
                    if self.stop_duration > 5:
                        self.stop_sign_flag = False
                        self.stop_sign_active = False
                else:
                    self.rover_controller.steer_rover(prediction)
                    self.distance_to_stop_sign = 30
                
                rawCapture.truncate(0)
            
            # Uncomment this line if the cv2.imshow line above is uncommented
#            cv2.destroyAllWindows()

class StartServerThread(object):

    def __init__(self):
        self.HOST = 'ip_address'
        self.PORT = 8888
        self.METRICS_PORT = 8887
        self.main_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.main_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.main_socket.bind((self.HOST, self.PORT))
        self.metrics_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.metrics_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.metrics_socket.bind((self.HOST, self.METRICS_PORT))

    def listen(self):
        self.main_socket.listen(5)
        self.metrics_socket.listen(5)
        print 'Socket is now listening for connections!'
        while True:
            main_client, address = self.main_socket.accept()
            metrics_client, address = self.metrics_socket.accept()
            threading.Thread(target= CameraFeedHandler().handle, args= (main_client,)).start()
            threading.Thread(target= GenuinoDataHandler().handle, args= (metrics_client,)).start()

if __name__ == "__main__":
    StartServerThread().listen()