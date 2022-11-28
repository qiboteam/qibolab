class tii_rfsoc4x2:

    def setup()


""""
import socket
import sys
import struct
HOST = "192.168.1.69"  # Serverinterface address
PORT = 6000

#data = " ".join(sys.argv[1:])
data = struct.pack('BH', 3, 1000)
# Create a socket (SOCK_STREAM means a TCP socket)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    # Connect to server and send data
    sock.connect((HOST, PORT))
    sock.sendall(bytes(data))
#    sock.sendall(bytes(data , "utf-8"))

    # Receive data from the server and shut down
    received = str(sock.recv(1024))
#    received = str(sock.recv(1024), "utf-8")

print("Sent:     {}".format(data))
print("Received: {}".format(received))
