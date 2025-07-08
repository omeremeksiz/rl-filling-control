import logging
from pymodbus.client import ModbusTcpClient

class ModbusClient:
    def __init__(self, device_ip, device_port):
        self.device_ip = device_ip
        self.device_port = device_port
        self.client = ModbusTcpClient(self.device_ip, port=self.device_port)

    def connect(self):
        if self.client.connect():
            logging.info(f"Connected to Modbus device at {self.device_ip}:{self.device_port}")
        else:
            raise ConnectionError(f"Failed to connect to Modbus device at {self.device_ip}:{self.device_port}")

    def send_weight(self, register, weight):
        try:
            scaled_weight = int(weight * 1000)  
            zero_based_register = register - 40001  # Adjust for Modbus addressing

            # Write to the register
            result = self.client.write_register(zero_based_register, scaled_weight)
            if result.isError():
                logging.error(f"Error writing weight {weight} (scaled {scaled_weight}) to register {register}")
            else:
                logging.info(f"Weight {weight} (scaled {scaled_weight}) successfully written to register {register}")
        except Exception as e:
            logging.error(f"Error in send_weight: {e}")

    def close(self):
        self.client.close()
        logging.info("Connection to Modbus device closed.")
