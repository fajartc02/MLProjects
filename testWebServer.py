import cv2
import socket
import pickle
import struct

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('10.65.129.69', 8888))  # Binding to all available interfaces
server_socket.listen(5)

print("Server is listening...")

while True:
    client_socket, client_address = server_socket.accept()
    print(f"Connection from {client_address} accepted")
    cap = cv2.VideoCapture(0)  # Capture video from default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_data = pickle.dumps(frame)
        client_socket.sendall(struct.pack("Q", len(frame_data)))
        client_socket.sendall(frame_data)

        if cv2.waitKey(1) == 27:  # Press 'Esc' to stop streaming
            client_socket.close()
            break

    cap.release()

server_socket.close()