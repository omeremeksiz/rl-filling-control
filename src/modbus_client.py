"""
Modbus client for sending switching points to physical device.
"""

import logging
from typing import Optional
from config import DEFAULT_MODBUS_REGISTER

try:
    from pymodbus.client import ModbusTcpClient
    PYMODBUS_AVAILABLE = True
except ImportError:
    PYMODBUS_AVAILABLE = False
    logging.warning("pymodbus not available. Install with: pip install pymodbus")


class ModbusClient:
    """Modbus client for sending switching points to physical device."""
    
    def __init__(self, ip: str, port: int, register: int = DEFAULT_MODBUS_REGISTER):
        """
        Initialize Modbus client.
        
        Args:
            ip: Device IP address
            port: Device Modbus port
            register: Modbus register address for writing switching points
        """
        self.ip = ip
        self.port = port
        self.register = register
        self.client = None
        
        if not PYMODBUS_AVAILABLE:
            raise ImportError("pymodbus package required for Modbus communication")
            
    def connect(self) -> bool:
        """
        Connect to Modbus device.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client = ModbusTcpClient(self.ip, port=self.port)
            if self.client.connect():
                logging.info(f"Connected to Modbus device at {self.ip}:{self.port}")
                return True
            else:
                logging.error(f"Failed to connect to Modbus device at {self.ip}:{self.port}")
                return False
                
        except Exception as e:
            logging.error(f"Error connecting to Modbus device: {e}")
            return False
    
    def send_switch_point(self, switch_point: float) -> bool:
        """
        Send switching point to device via Modbus.
        
        Args:
            switching_point: Switching point weight value
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            logging.error("Modbus client not connected")
            return False
            
        try:
            # Scale the switching point for device (multiply by 1000)
            scaled_weight = int(switch_point * 1000)
            
            # Adjust register address (Modbus addressing convention)
            zero_based_register = self.register - 40001
            
            # Write to the register
            result = self.client.write_register(zero_based_register, scaled_weight)
            
            if result.isError():
                logging.error(f"Error writing switching point {switch_point} "
                            f"(scaled {scaled_weight}) to register {self.register}")
                return False
            else:
                logging.info(f"Switching point {switch_point} "
                           f"(scaled {scaled_weight}) sent to register {self.register}")
                return True
                
        except Exception as e:
            logging.error(f"Error sending switching point: {e}")
            return False
    
    def close(self) -> None:
        """Close Modbus connection."""
        if self.client:
            self.client.close()
            self.client = None
            logging.info("Modbus connection closed")