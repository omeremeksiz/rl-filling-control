"""
Real-world testing integration for the RL filling control system.
Handles communication with physical device and data collection.
"""

import logging
import time
from typing import Optional, Dict, Any, List
from tcp_client import TCPClient
from modbus_client import ModbusClient
from real_data_processor import RealDataProcessor
from database_handler import DatabaseHandler
from reward_calculator import RewardCalculator
from config import (
    DEFAULT_TCP_IP, DEFAULT_TCP_PORT, DEFAULT_MODBUS_IP, DEFAULT_MODBUS_PORT,
    DEFAULT_MODBUS_REGISTER, DEFAULT_DB_CONFIG, DEFAULT_SAFE_WEIGHT_MIN, 
    DEFAULT_SAFE_WEIGHT_MAX, DEFAULT_WEIGHT_QUANTIZATION_STEP
)


class RealWorldTester:
    """Handles real-world testing with physical filling device."""
    
    def __init__(self,
                 tcp_ip: str = DEFAULT_TCP_IP,
                 tcp_port: int = DEFAULT_TCP_PORT,
                 modbus_ip: str = DEFAULT_MODBUS_IP,
                 modbus_port: int = DEFAULT_MODBUS_PORT,
                 modbus_register: int = DEFAULT_MODBUS_REGISTER,

                 quantization_step: int = DEFAULT_WEIGHT_QUANTIZATION_STEP,
                 db_config: Dict[str, Any] = None,
                 reward_calculator: Optional[RewardCalculator] = None):
        """
        Initialize real-world tester.
        
        Args:
            tcp_ip: TCP server IP address
            tcp_port: TCP server port
            modbus_ip: Modbus device IP address
            modbus_port: Modbus device port
            modbus_register: Modbus register for switching points
            quantization_step: Weight quantization step
            db_config: Database configuration
            reward_calculator: Reward calculator instance
        """
        # Communication clients
        self.tcp_client = TCPClient(tcp_ip, tcp_port)
        self.modbus_client = ModbusClient(modbus_ip, modbus_port, modbus_register)
        
        # Data processing (use common weight limits scaled for real device)
        real_tolerance_limits = [
            DEFAULT_SAFE_WEIGHT_MIN * quantization_step,
            DEFAULT_SAFE_WEIGHT_MAX * quantization_step
        ]
        self.data_processor = RealDataProcessor(
            real_tolerance_limits,
            quantization_step
        )
        
        # Database handler
        self.db_handler = DatabaseHandler(db_config or DEFAULT_DB_CONFIG)
        
        # Reward calculator
        self.reward_calculator = reward_calculator or RewardCalculator()
        
        # Statistics
        self.session_stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'failed_episodes': 0,
            'safe_episodes': 0,
            'overflow_episodes': 0,
            'underflow_episodes': 0
        }
        
    def connect_devices(self) -> bool:
        """
        Connect to all devices.
        
        Returns:
            True if all connections successful, False otherwise
        """
        logging.info("Connecting to devices...")
        
        tcp_connected = self.tcp_client.connect()
        modbus_connected = self.modbus_client.connect()
        db_connected = self.db_handler.connect()
        
        if tcp_connected and modbus_connected:
            logging.info("All communication devices connected successfully")
            if not db_connected:
                logging.warning("Database connection failed - episodes will not be saved")
            return True
        else:
            logging.error("Failed to connect to devices")
            return False
    
    def run_episode(self, switching_point: float) -> Optional[Dict[str, Any]]:
        """
        Run a single filling episode with the given switching point.
        
        Args:
            switching_point: Switching point to send to device
            
        Returns:
            Episode data dictionary or None if failed
        """
        logging.info(f"Starting episode with switching point: {switching_point}")
        
        # Send switching point to device
        if not self.modbus_client.send_switching_point(switching_point):
            logging.error("Failed to send switching point")
            self.session_stats['failed_episodes'] += 1
            return None
        
        # Wait for episode to complete and receive data
        raw_data = self.tcp_client.receive_data()
        if not raw_data:
            logging.error("No data received from device")
            self.session_stats['failed_episodes'] += 1
            return None
        
        logging.info(f"Received raw data: {raw_data[:100]}...")  # Log first 100 chars
        
        # Parse the data
        parsed_data = self.data_processor.parse_raw_data(raw_data)
        if not parsed_data:
            logging.error("Failed to parse received data")
            self.session_stats['failed_episodes'] += 1
            return None
        
        # Create filling session for reward calculation
        filling_session = self.data_processor.create_filling_session(parsed_data)
        if not filling_session:
            logging.error("Failed to create filling session")
            self.session_stats['failed_episodes'] += 1
            return None
        
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            filling_session.episode_length,
            filling_session.final_weight,
            method="standard"  # Use standard reward calculation
        )
        
        # Add reward to parsed data
        parsed_data['reward'] = reward
        parsed_data['filling_session'] = filling_session
        
        # Update statistics
        self._update_session_stats(parsed_data)
        
        # Save to database if connected
        self._save_episode_to_database(parsed_data)
        
        logging.info(f"Episode completed - Final weight: {parsed_data['final_weight']}, "
                    f"Reward: {reward:.2f}")
        
        return parsed_data
    
    def _update_session_stats(self, episode_data: Dict[str, Any]) -> None:
        """Update session statistics."""
        self.session_stats['total_episodes'] += 1
        self.session_stats['successful_episodes'] += 1
        
        if episode_data['overflow_amount'] > 0:
            self.session_stats['overflow_episodes'] += 1
        elif episode_data['underflow_amount'] > 0:
            self.session_stats['underflow_episodes'] += 1
        else:
            self.session_stats['safe_episodes'] += 1
    
    def _save_episode_to_database(self, episode_data: Dict[str, Any]) -> None:
        """Save episode to database."""
        try:
            # Save original data
            original_id = self.db_handler.save_original_episode(
                episode_data['raw_data'],
                episode_data['coarse_time'],
                episode_data['fine_time'],
                episode_data['total_time'],
                episode_data['switching_point'],
                episode_data['overflow_amount'],
                episode_data['underflow_amount']
            )
            
            if original_id:
                # Save parsed data
                parsed_id = self.db_handler.save_parsed_episode(
                    original_id,
                    episode_data['weight_sequence'],
                    episode_data['coarse_time'],
                    episode_data['fine_time'],
                    episode_data['total_time'],
                    episode_data['switching_point'],
                    episode_data['overflow_amount'],
                    episode_data['underflow_amount']
                )
                
                if parsed_id:
                    # Update statistics
                    episode_stats = self.data_processor.get_episode_stats(episode_data)
                    self.db_handler.save_statistics(**episode_stats)
                    
                    logging.info(f"Episode saved to database (IDs: {original_id}, {parsed_id})")
                
        except Exception as e:
            logging.error(f"Error saving episode to database: {e}")
    
    def run_agent_testing(self, agent, num_episodes: int = 10, initial_switch_point: float = 45.0) -> List[Dict[str, Any]]:
        """
        Run multiple episodes using an RL agent to select switching points.
        
        Args:
            agent: Trained RL agent
            num_episodes: Number of episodes to run
            initial_switch_point: Starting switch point (from DEFAULT_STARTING_SWITCH_POINT)
            
        Returns:
            List of episode data dictionaries
        """
        if not self.connect_devices():
            logging.error("Failed to connect to devices")
            return []
        
        episodes = []
        current_switch_point = initial_switch_point
        
        try:
            for episode_num in range(num_episodes):
                logging.info(f"Running episode {episode_num + 1}/{num_episodes}")
                
                # Use current switching point for this episode (like training)
                experienced_switch_point = current_switch_point
                
                # Run episode (steps 2-5: device filling, data reception, parsing, reward calculation)
                episode_data = self.run_episode(experienced_switch_point)
                if episode_data:
                    episodes.append(episode_data)
                    
                    # Step 6: Update agent with real-world episode data (exactly like training)
                    filling_session = episode_data['filling_session']
                    self._update_agent_with_episode(agent, filling_session, experienced_switch_point)
                    
                    # Determine termination type
                    final_weight = episode_data['final_weight']
                    if final_weight < self.reward_calculator.safe_weight_min * self.quantization_step:
                        termination_type = "underweight"
                    elif final_weight > self.reward_calculator.safe_weight_max * self.quantization_step:
                        termination_type = "overweight"
                    else:
                        termination_type = "safe"
                    
                    # Get model-selected next switching point (what agent thinks is best now)
                    model_selected_next_switch_point = agent.get_optimal_switch_point()
                    
                    # Select next action for next episode (may include exploration)
                    next_switch_point, exploration_flag = agent.select_action(experienced_switch_point)
                    
                    # Determine explored switching point
                    explored_switch_point = next_switch_point if exploration_flag else None
                    
                    # Console output matching training format EXACTLY
                    print(f"--- Episode {episode_num + 1}/{num_episodes} ---")
                    print(f"Experienced Switching Point: {experienced_switch_point}")
                    print(f"Termination Type: {termination_type}")
                    print(f"Model-Selected Next Switching Point: {model_selected_next_switch_point}")
                    print(f"Explored Switching Point: {explored_switch_point}")
                    print()
                    
                    logging.info(f"Episode {episode_num + 1} completed - "
                               f"Final weight: {episode_data['final_weight']}, "
                               f"Reward: {episode_data['reward']:.2f}")
                    
                    # Update current switch point for next episode
                    current_switch_point = next_switch_point
                    
                else:
                    print(f"--- Episode {episode_num + 1}/{num_episodes} ---")
                    print(f"Episode FAILED")
                    print()
                    logging.warning(f"Episode {episode_num + 1} failed")
                
                # Brief pause between episodes
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            logging.info("Testing interrupted by user")
        except Exception as e:
            logging.error(f"Error during testing: {e}")
        finally:
            self.disconnect_devices()
        
        self._print_session_summary()
        return episodes
    
    def run_manual_testing(self, switching_points: List[float]) -> List[Dict[str, Any]]:
        """
        Run episodes with manually specified switching points.
        
        Args:
            switching_points: List of switching points to test
            
        Returns:
            List of episode data dictionaries
        """
        if not self.connect_devices():
            logging.error("Failed to connect to devices")
            return []
        
        episodes = []
        
        try:
            for i, switching_point in enumerate(switching_points):
                logging.info(f"Running episode {i + 1}/{len(switching_points)} "
                           f"with switching point: {switching_point}")
                
                episode_data = self.run_episode(switching_point)
                if episode_data:
                    episodes.append(episode_data)
                else:
                    logging.warning(f"Episode {i + 1} failed")
                
                # Brief pause between episodes
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            logging.info("Testing interrupted by user")
        except Exception as e:
            logging.error(f"Error during testing: {e}")
        finally:
            self.disconnect_devices()
        
        self._print_session_summary()
        return episodes
    
    def _print_session_summary(self) -> None:
        """Print session statistics summary."""
        stats = self.session_stats
        logging.info("\n" + "="*50)
        logging.info("REAL-WORLD TESTING SESSION SUMMARY")
        logging.info("="*50)
        logging.info(f"Total Episodes: {stats['total_episodes']}")
        logging.info(f"Successful Episodes: {stats['successful_episodes']}")
        logging.info(f"Failed Episodes: {stats['failed_episodes']}")
        logging.info(f"Safe Episodes: {stats['safe_episodes']}")
        logging.info(f"Overflow Episodes: {stats['overflow_episodes']}")
        logging.info(f"Underflow Episodes: {stats['underflow_episodes']}")
        
        if stats['successful_episodes'] > 0:
            safe_rate = (stats['safe_episodes'] / stats['successful_episodes']) * 100
            logging.info(f"Safe Fill Rate: {safe_rate:.1f}%")
        
        logging.info("="*50)
    
    def _update_agent_with_episode(self, agent, filling_session, switching_point):
        """
        Update the RL agent with real-world episode data (step 6 of the testing loop).
        
        Args:
            agent: RL agent to update
            filling_session: FillingSession object from real episode
            switching_point: Switching point that was used
        """
        try:
            # Different agents need different update methods
            agent_type = type(agent).__name__
            
            if agent_type == "QLearningAgent":  # MAB agent
                # MAB: Direct Q-value update with episode reward
                episode_length = filling_session.episode_length
                final_weight = filling_session.final_weight
                reward = self.reward_calculator.calculate_reward(episode_length, final_weight, method="mab")
                agent._update_q_value(switching_point, reward)
                logging.info(f"MAB agent updated: Q({switching_point}) with reward {reward:.2f}")
                
            elif agent_type in ["MonteCarloAgent", "TDAgent", "StandardQLearningAgent"]:
                # MC/TD/Q-Learning: Episode-based learning using FillingSession
                episode_length, final_weight = agent.train_episode(switching_point)
                logging.info(f"{agent_type} updated with episode: length={episode_length}, weight={final_weight}")
                
            else:
                logging.warning(f"Unknown agent type: {agent_type}, skipping update")
                
        except Exception as e:
            logging.error(f"Error updating agent with episode: {e}")
            
        # Update exploration rate if decay is enabled
        if hasattr(agent, '_update_exploration_rate'):
            agent._update_exploration_rate()
            logging.debug(f"Exploration rate: {agent.exploration_rate:.3f}")
    
    def disconnect_devices(self) -> None:
        """Disconnect from all devices."""
        logging.info("Disconnecting from devices...")
        self.tcp_client.close()
        self.modbus_client.close()
        self.db_handler.close()
        logging.info("All devices disconnected")