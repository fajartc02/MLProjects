import cv2
import socket
import pickle
import struct

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('10.65.129.69', 8888))
server_socket.listen(3000)
print("Server is listening...")
client_socket, client_address = server_socket.accept()
print(f"Connection from {client_address} accepted")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame_data = pickle.dumps(frame)
    client_socket.sendall(struct.pack("Q", len(frame_data)))
    print(client_socket)
    client_socket.sendall(frame_data)
    print(frame_data)
    cv2.imshow('Server', frame)
    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()