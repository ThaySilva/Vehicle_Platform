import socket
import sys
from thread import *
import RPi.GPIO as GPIO

# Set up the GPIO output pins wired to the wheels
GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT)
GPIO.setup(11, GPIO.OUT)
GPIO.setup(29, GPIO.OUT)
GPIO.setup(35, GPIO.OUT)

# Keep the power on the wheels off waiting for command
GPIO.output(7, False)
GPIO.output(11, False)
GPIO.output(29, False)
GPIO.output(35, False)

# Set up host and port to use in the connection
HOST = 'ip_address'
PORT = 8888

# Create socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print 'Socket Created!'

# Bind the socker to the chosen host and port
try:
    server_socket.bind((HOST, PORT))
except socket.error, message:
    print 'Bind failed! Error Code : ' + str(message[0]) + '. Message : ' + message[1]
    sys.exit()
print 'Socket Bind Complete!'

# Start listening for connections
server_socket.listen(10)
print 'Socket Is Now Listening For Connections!'

# Set up thread that will receive commands from the client
def client_thread(connection):
    connection.send('Welcome to Raspberry Pi!')

    # While connected to the client keep listening for commands
    while True:
        command = connection.recv(1024)
        if not command:
            break
        elif command == "forwardOn":
            # Rotate back wheels forwards
            GPIO.output(11, False)
            GPIO.output(7, True)
        elif command == "forwardOff":
            # Stop the back wheels from moving
            GPIO.output(7, False)
        elif command == "reverseOn":
            # Rotate back wheels backwards
            GPIO.output(7, False)
            GPIO.output(11, True)
        elif command == "reverseOff":
            # Stop the back wheels from moving
            GPIO.output(11, False)
        elif command == "leftOn":
            # Rotate front wheels to the left
            GPIO.output(35, False)
            GPIO.output(29, True)
        elif command == "leftOff":
            # Move front wheels back to the middle position
            GPIO.output(29, False)
        elif command == "rightOn":
            # Rotate front wheels to the right
            GPIO.output(29, False)
            GPIO.output(35, True)
        elif command == "rightOff":
            GPIO.output(35, False)

    # Close connection if lost
    connection.close()

# Start client thread once the host and the client have successfully connected
while 1:
    connection, address = server_socket.accept()
    print 'Connected with ' + address[0] + ' : ' + str(address[1])

    start_new_thread(client_thread, (connection,))

# Close socket when done
server_socket.close()