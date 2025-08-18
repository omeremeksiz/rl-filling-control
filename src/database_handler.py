"""
Database handler for saving real-world filling episodes.
"""

import logging
import json
from typing import Dict, Any, Optional, List
from config import DEFAULT_DB_CONFIG

try:
    import mysql.connector
    from mysql.connector import Error
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    logging.warning("mysql-connector-python not available. Install with: pip install mysql-connector-python")


class DatabaseHandler:
    """Handles database operations for real-world filling episodes."""
    
    def __init__(self, db_config: Dict[str, Any] = None):
        """
        Initialize database handler.
        
        Args:
            db_config: Database configuration dictionary
        """
        self.config = db_config or DEFAULT_DB_CONFIG.copy()
        self.connection = None
        
        if not MYSQL_AVAILABLE:
            logging.warning("Database functionality disabled - mysql-connector-python not available")
    
    def connect(self) -> bool:
        """
        Connect to database.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not MYSQL_AVAILABLE:
            return False
            
        try:
            self.connection = mysql.connector.connect(
                database=self.config['name'],
                user=self.config['user'],
                password=self.config['password'],
                host=self.config['host'],
                port=self.config['port']
            )
            logging.info("Database connection successful")
            self._create_tables_if_not_exist()
            return True
            
        except Error as e:
            logging.error(f"Error connecting to database: {e}")
            self.connection = None
            return False
    
    def _create_tables_if_not_exist(self) -> None:
        """Create database tables if they don't exist."""
        if not self.connection:
            return
            
        try:
            cursor = self.connection.cursor()
            
            # Create original data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS filling_data_table (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    raw_data TEXT,
                    coarse_time INT,
                    fine_time INT,
                    total_time INT,
                    switching_state INT,
                    overflow_amount INT DEFAULT 0,
                    underflow_amount INT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create parsed data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parsed_filling_data_table (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    original_id INT,
                    raw_data TEXT,
                    coarse_time INT,
                    fine_time INT,
                    total_time INT,
                    switching_state INT,
                    overflow_amount INT DEFAULT 0,
                    underflow_amount INT DEFAULT 0,
                    q_value JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (original_id) REFERENCES filling_data_table(id)
                )
            """)
            
            # Create statistics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS filling_statistics_table (
                    id INT PRIMARY KEY,
                    valid_fillings INT DEFAULT 0,
                    overflow_fillings INT DEFAULT 0,
                    underflow_fillings INT DEFAULT 0,
                    safe_fillings INT DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
            """)
            
            self.connection.commit()
            cursor.close()
            logging.info("Database tables verified/created")
            
        except Error as e:
            logging.error(f"Error creating tables: {e}")
    
    def save_original_episode(self, 
                            raw_data: str,
                            coarse_time: int,
                            fine_time: int,
                            total_time: int,
                            switching_state: int,
                            overflow_amount: int = 0,
                            underflow_amount: int = 0) -> Optional[int]:
        """
        Save original episode data.
        
        Returns:
            Database ID of saved record, None if failed
        """
        if not self.connection:
            if not self.connect():
                return None
                
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO filling_data_table 
                (raw_data, coarse_time, fine_time, total_time, switching_state, overflow_amount, underflow_amount)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (raw_data, coarse_time, fine_time, total_time, 
                                 switching_state, overflow_amount, underflow_amount))
            row_id = cursor.lastrowid
            self.connection.commit()
            cursor.close()
            
            logging.info(f"Original episode data saved with ID: {row_id}")
            return row_id
            
        except Error as e:
            logging.error(f"Error saving original episode: {e}")
            return None
    
    def save_parsed_episode(self,
                          original_id: int,
                          weight_sequence: List[int],
                          coarse_time: int,
                          fine_time: int,
                          total_time: int,
                          switching_state: int,
                          overflow_amount: int = 0,
                          underflow_amount: int = 0) -> Optional[int]:
        """
        Save parsed episode data.
        
        Returns:
            Database ID of saved record, None if failed
        """
        if not self.connection:
            if not self.connect():
                return None
                
        try:
            # Convert weight sequence to string
            if isinstance(weight_sequence, list):
                raw_data = ','.join(map(str, weight_sequence))
            else:
                raw_data = str(weight_sequence)
                
            cursor = self.connection.cursor()
            query = """
                INSERT INTO parsed_filling_data_table 
                (original_id, raw_data, coarse_time, fine_time, total_time, 
                 switching_state, overflow_amount, underflow_amount)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (original_id, raw_data, coarse_time, fine_time, 
                                 total_time, switching_state, overflow_amount, underflow_amount))
            row_id = cursor.lastrowid
            self.connection.commit()
            cursor.close()
            
            logging.info(f"Parsed episode data saved with ID: {row_id}")
            return row_id
            
        except Error as e:
            logging.error(f"Error saving parsed episode: {e}")
            return None
    
    def update_q_values(self, episode_id: int, q_values: Dict[Any, Any]) -> bool:
        """
        Update Q-values for an episode.
        
        Args:
            episode_id: Database ID of the episode
            q_values: Q-value dictionary to save
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connection:
            if not self.connect():
                return False
                
        try:
            # Serialize Q-values to JSON
            serialized_q_values = json.dumps({
                str(key): value for key, value in q_values.items()
            })
            
            cursor = self.connection.cursor()
            query = """
                UPDATE parsed_filling_data_table
                SET q_value = %s
                WHERE id = %s
            """
            cursor.execute(query, (serialized_q_values, episode_id))
            self.connection.commit()
            cursor.close()
            
            logging.info(f"Q-values updated for episode {episode_id}")
            return True
            
        except Error as e:
            logging.error(f"Error updating Q-values: {e}")
            return False
    
    def save_statistics(self, **stats: int) -> bool:
        """
        Save or update filling statistics.
        
        Args:
            **stats: Statistics to save (valid_fillings, overflow_fillings, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connection:
            if not self.connect():
                return False
                
        try:
            cursor = self.connection.cursor()
            query = """
                INSERT INTO filling_statistics_table 
                (id, valid_fillings, overflow_fillings, underflow_fillings, safe_fillings)
                VALUES (1, %(valid_fillings)s, %(overflow_fillings)s, %(underflow_fillings)s, %(safe_fillings)s)
                ON DUPLICATE KEY UPDATE 
                valid_fillings = valid_fillings + VALUES(valid_fillings),
                overflow_fillings = overflow_fillings + VALUES(overflow_fillings),
                underflow_fillings = underflow_fillings + VALUES(underflow_fillings),
                safe_fillings = safe_fillings + VALUES(safe_fillings)
            """
            cursor.execute(query, stats)
            self.connection.commit()
            cursor.close()
            
            logging.info("Statistics updated successfully")
            return True
            
        except Error as e:
            logging.error(f"Error saving statistics: {e}")
            return False
    
    def get_episode_data(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve episode data by ID.
        
        Args:
            episode_id: Database ID of the episode
            
        Returns:
            Episode data dictionary or None if not found
        """
        if not self.connection:
            if not self.connect():
                return None
                
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT * FROM parsed_filling_data_table WHERE id = %s"
            cursor.execute(query, (episode_id,))
            result = cursor.fetchone()
            cursor.close()
            
            return result
            
        except Error as e:
            logging.error(f"Error retrieving episode data: {e}")
            return None
    
    def close(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logging.info("Database connection closed")