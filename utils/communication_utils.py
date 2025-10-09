# utils/communication_utils.py
from __future__ import annotations

from typing import Any, List, Optional
import socket
import logging

try:
    from pymodbus.client import ModbusTcpClient  # type: ignore
    PYMODBUS_AVAILABLE = True
except Exception:
    PYMODBUS_AVAILABLE = False


class ModbusClient:
    def __init__(self, host: str, port: int, register: int = 40010) -> None:
        if not PYMODBUS_AVAILABLE:
            raise ImportError("pymodbus package required for Modbus communication")
        self.host = host
        self.port = port
        self.register = register
        self.client = None

    def connect(self) -> bool:
        try:
            self.client = ModbusTcpClient(self.host, port=self.port)
            ok = self.client.connect()
            if ok:
                logging.info(f"Connected to Modbus device at {self.host}:{self.port}")
            else:
                logging.error(f"Failed to connect Modbus {self.host}:{self.port}")
            return bool(ok)
        except Exception as e:
            logging.error(f"Error connecting to Modbus device: {e}")
            return False

    def send_switch_point(self, switch_point: float) -> bool:
        if not self.client:
            logging.error("Modbus client not connected")
            return False
        try:
            scaled = int(round(switch_point))
            if scaled < 0:
                scaled = 0
            if scaled > 0xFFFF:
                scaled = int(round(switch_point * 1000))
                scaled = max(0, min(0xFFFF, scaled))

            zero_based_register = self.register - 40001
            result = self.client.write_register(zero_based_register, scaled)
            if hasattr(result, 'isError') and result.isError():
                logging.error("Error writing switching point")
                return False
            logging.info(f"Sent switching point {switch_point} (scaled {scaled}) to register {self.register}")
            return True
        except Exception as e:
            logging.error(f"Error sending switching point: {e}")
            return False

    def close(self) -> None:
        if self.client:
            self.client.close()
            self.client = None


class TCPClient:
    def __init__(self, host: str, port: int, timeout: float = 2.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client_socket = None

    def connect(self) -> bool:
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))
            self.client_socket.settimeout(self.timeout)
            logging.info(f"Connected to TCP device at {self.host}:{self.port}")
            self._send_test_message()
            return True
        except Exception as e:
            logging.error(f"Failed to connect to TCP device: {e}")
            self.client_socket = None
            return False

    def _send_test_message(self) -> None:
        try:
            message = "test"
            self.client_socket.sendall(message.encode('ascii'))
            logging.info("Sent test message: test")
        except Exception as e:
            logging.error(f"Failed to send test message: {e}")

    def receive_data(self) -> Optional[str]:
        if not self.client_socket:
            logging.error("TCP client not connected")
            return None
        try:
            data = self.client_socket.recv(4096)
            return data.decode('ascii') if data else None
        except Exception as e:
            logging.error(f"Error receiving TCP data: {e}")
            return None

    def close(self) -> None:
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None


def create_modbus_client(host: str, port: int, register: Optional[int] = None) -> ModbusClient:
    return ModbusClient(host, port, register=register or 40010)


def create_tcp_client(host: str, port: int, timeout: Optional[float] = None) -> TCPClient:
    return TCPClient(host, port, timeout=timeout or 2.0)


def parse_live_payload_to_floats(payload: str) -> List[float]:
    """
    Parse a live data payload into a list of floats.
    Assumes payload contains numbers separated by commas or whitespace.
    """
    parts = payload.replace("\n", " ").replace("\r", " ").replace(",", " ").split()
    values: List[float] = []
    for p in parts:
        try:
            values.append(float(p))
        except ValueError:
            continue
    return values
