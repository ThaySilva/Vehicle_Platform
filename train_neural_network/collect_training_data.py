import socket
import sys
from thread import *
import RPi.GPIO as GPIO
import picamera
from picamera.array import PiRGBArray
import time
import cv2
import numpy as np

# Setup the GPIO output pins wired to the motors
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

# Setup host and port to use in the connection with the Rover HQ
HOST = 'ip_address'
PORT = 8888

# Create a socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print 'Socket Created!'

# Bind the socket to the chosen host and port
try:
    server_socket.bind((HOST, PORT))
except socket.error, message:
    print 'Bind failed! Error Code: ' + str(message[0]) + '. Message: ' + message[1]
    sys.exit()
# If bind successful
print 'Socket Bind Completed!'

# Start listening for connections
server_socket.listen(10)
print 'Socket is now listening for connections!'

def mask_image(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def collect_data(connection):
    # Setup camera variables
    camera = picamera.PiCamera()
    camera.resolution = (320, 240)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera)

    # Setup variables
    saved_frames = 0
    total_frames = 0
    num_frame = 1
    image_array = np.zeros((1, 784), np.float32)
    label_array = np.zeros((1, 3), np.float32)
    quit = False
    vertices = np.array([[0,240],[0,140],[140,120],[180,120],[320,140],[320,240]], np.int32)

    print 'Start collection of training data...'
    collection_start_time = cv2.getTickCount()

    while True:
        for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
            image = rawCapture.array
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            canny_image = cv2.Canny(grayscale_image, threshold1=200, threshold2=300)
            roi_image = mask_image(canny_image, [vertices])
            resized_image = cv2.resize(roi_image, (28, 28))
            training_image = resized_image.reshape(1, 784).astype(np.float32)

            cv2.imshow('image', image)
            cv2.waitKey(1) & 0xFF

            training_label = [0,0,0]

            command = connection.recv(1024)
            if not command:
                break
            if command == "disconnect":
                print 'Quit collection!'
                quit = True
                break
            elif command == "forwardOn":
                print 'Forward'
                training_label[1] = 1
                image_array = np.vstack((image_array, training_image))
                label_array = np.vstack((label_array, training_label))
                saved_frames += 1
                GPIO.output(11, False)
                GPIO.output(7, True)
            elif command == "forwardOff":
                GPIO.output(7, False)
            elif command == "leftOn":
                print 'Left'
                training_label[0] = 1
                image_array = np.vstack((image_array, training_image))
                label_array = np.vstack((label_array, training_label))
                saved_frames += 1
                GPIO.output(35, False)
                GPIO.output(29, True)
            elif command == "leftOff":
                GPIO.output(29, False)
            elif command == "rightOn":
                print 'Right'
                training_label[2] = 1
                image_array = np.vstack((image_array, training_image))
                label_array = np.vstack((label_array, training_label))
                saved_frames += 1
                GPIO.output(29, False)
                GPIO.output(35, True)
            elif command == "rightOff":
                GPIO.output(35, False)

            cv2.imwrite('training_images/frame{:>05}.jpg'.format(num_frame), roi_image)

            num_frame += 1
            total_frames += 1
            rawCapture.truncate(0)

            if quit == True:
                connection.close()
            
        train = image_array[1:, :]
        train_labels = label_array[1:, :]

        filename = str(int(time.time()))
        np.savez("training_data/"+filename+".npz", train=train, train_labels=train_labels)
        
        collection_stop_time = cv2.getTickCount()
        collection_time = (collection_stop_time - collection_start_time) / cv2.getTickFrequency()
        print 'Collection Duration: ', collection_time

        print 'Total Frames: ', total_frames
        print 'Saved Frames: ', saved_frames
        print 'Dropped Frames: ', total_frames - saved_frames

        cv2.destroyAllWindows()
        server_socket.close()
        sys.exit()

while 1:
    connection, address = server_socket.accept()
    print 'Connected with ' + address[0] + " : " + str(address[1])
    start_new_thread(collect_data, (connection,))
