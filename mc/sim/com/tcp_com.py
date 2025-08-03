import socket
import time
import logging

class TCPClient:
    def __init__(self, ip, port, timeout=0.1):
        self.ip = ip
        self.port = port
        self.client_socket = None
        self.timeout = timeout  # Timeout in seconds (0.1 seconds = 100 ms)

    def connect(self):
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.ip, self.port))
            self.client_socket.settimeout(self.timeout)  
            logging.info("Connected to the device.")

            self.send_test_message()
        except Exception as e:
            logging.error(f"Failed to connect: {e}")
            self.client_socket = None
    
    def send_test_message(self):
        try:
            message = "test"
            self.client_socket.sendall(message.encode('ascii'))
            logging.info(f"Sent message: {message}")
        except Exception as e:
            logging.error(f"Failed to send message: {e}")

    def receive_data(self):
        buffer = []
        last_received_time = time.time()

        while True:
            try:
                data = self.client_socket.recv(4096)
                if data:
                    buffer.append(data.decode('ascii'))
                    last_received_time = time.time()  # Reset the timer upon receiving data
                else:
                    break  # No data received, possibly end of transmission

            except socket.timeout:
                if (time.time() - last_received_time) > self.timeout:
                    logging.info("Timeout reached, considering the current filling episode as complete.")
                    return None  
                continue  

        return ''.join(buffer) if buffer else None

    def close_connection(self):
        if self.client_socket:
            self.client_socket.close()
            logging.info("TCP connection closed.")
