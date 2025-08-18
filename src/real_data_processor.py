"""
Real-world data processor for converting device data to model format.
Handles parsing and preprocessing of real filling episode data.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
from data_processor import FillingSession
from config import DEFAULT_WEIGHT_QUANTIZATION_STEP


class RealDataProcessor:
    """Processes real-world filling data from physical device."""
    
    def __init__(self, 
                 tolerance_limits: List[int] = None,
                 quantization_step: int = DEFAULT_WEIGHT_QUANTIZATION_STEP):
        """
        Initialize real data processor.
        
        Args:
            tolerance_limits: [min_weight, max_weight] tolerance limits in grams
            quantization_step: Step for quantizing weights to model format
        """
        self.tolerance_limits = tolerance_limits
        self.quantization_step = quantization_step
        
    def parse_raw_data(self, raw_data: str) -> Optional[Dict[str, Any]]:
        """
        Parse raw data from device into structured format.
        
        Args:
            raw_data: Raw data string from TCP connection
            
        Returns:
            Parsed data dictionary or None if invalid
        """
        if not raw_data:
            return None
            
        try:
            # Clean and process raw data
            cleaned_data = raw_data.replace(' ', '')
            cleaned_data = cleaned_data.replace('30,30', '-1,-1')  # Replace switch indicator
            
            data_pairs = cleaned_data.split(';')
            data_pairs = [pair for pair in data_pairs if pair]
            
            if not data_pairs:
                return None
                
            # Extract timing information and switching state
            timing_info = self._extract_timing_info(data_pairs)
            if not timing_info:
                return None
                
            # Parse weight sequence
            weight_sequence = self._parse_weight_sequence(data_pairs)
            if not weight_sequence:
                return None
                
            # Calculate overflow/underflow
            final_weight = self._get_final_weight(data_pairs)
            overflow_amount, underflow_amount = self._calculate_overflow_underflow(final_weight)
            
            return {
                'weight_sequence': weight_sequence,
                'final_weight': final_weight,
                'switching_point': timing_info['switching_state'],
                'episode_length': len(weight_sequence),
                'coarse_time': timing_info['coarse_time'],
                'fine_time': timing_info['fine_time'],
                'total_time': timing_info['total_time'],
                'overflow_amount': overflow_amount,
                'underflow_amount': underflow_amount,
                'raw_data': raw_data
            }
            
        except Exception as e:
            logging.error(f"Error parsing raw data: {e}")
            return None
    
    def _extract_timing_info(self, data_pairs: List[str]) -> Optional[Dict[str, Any]]:
        """Extract timing information from data pairs."""
        try:
            # Check for required markers
            if "300,300" not in data_pairs or "-1,-1" not in data_pairs:
                return None
                
            # Extract time from last pair
            last_pair = data_pairs[-1]
            fine_time, total_time = map(int, last_pair.split(','))
            
            fine_time *= 100  # Convert to milliseconds
            total_time *= 10  # Convert to milliseconds
            coarse_time = total_time - fine_time
            
            # Find switching state
            switching_state = None
            for i in range(len(data_pairs) - 1, -1, -1):
                if data_pairs[i] == "-1,-1":
                    if i > 0:
                        switching_state = int(data_pairs[i - 1].split(',')[0])
                    break
                    
            return {
                'coarse_time': coarse_time,
                'fine_time': fine_time,
                'total_time': total_time,
                'switching_state': switching_state
            }
            
        except Exception as e:
            logging.error(f"Error extracting timing info: {e}")
            return None
    
    def _parse_weight_sequence(self, data_pairs: List[str]) -> Optional[List[int]]:
        """Parse weight sequence from data pairs."""
        try:
            # Rearrange pairs to handle termination marker
            self._rearrange_pairs(data_pairs)
            
            # Extract weight values
            weights = []
            for pair in data_pairs:
                weight = self._extract_weight_from_pair(pair)
                if weight is not None:
                    weights.append(weight)
            
            # Quantize weights to model format
            quantized_weights = self._quantize_weights(weights)
            
            # Remove initial elements and final weight as per model format
            processed_weights = self._remove_elements(quantized_weights)
            
            return processed_weights
            
        except Exception as e:
            logging.error(f"Error parsing weight sequence: {e}")
            return None
    
    def _rearrange_pairs(self, data_pairs: List[str]) -> None:
        """Rearrange pairs to handle termination marker."""
        for i in range(len(data_pairs)):
            if '300,300' in data_pairs[i]:
                if i + 1 < len(data_pairs):
                    pair_after_300 = data_pairs.pop(i + 1)
                    data_pairs.insert(i, pair_after_300)
                break
    
    def _extract_weight_from_pair(self, pair: str) -> Optional[int]:
        """Extract weight value from data pair."""
        try:
            elements = pair.strip().split(',')
            if len(elements) > 0:
                return int(elements[0].strip())
        except (ValueError, IndexError):
            pass
        return None
    
    def _quantize_weights(self, weights: List[int]) -> List[int]:
        """Quantize weights to model format."""
        quantized = []
        for weight in weights:
            if weight == -1 or weight == 300:
                quantized.append(weight)  # Keep special markers
            else:
                quantized.append(max(0, round(weight / self.quantization_step)))
        return quantized
    
    def _remove_elements(self, weights: List[int]) -> List[int]:
        """Remove initial elements and final weight as per model requirements."""
        if len(weights) > 51:
            weights = weights[50:]  # Remove first 50 elements
        if len(weights) > 0:
            weights.pop()  # Remove final weight
        return weights
    
    def _get_final_weight(self, data_pairs: List[str]) -> Optional[int]:
        """Extract final weight from data pairs."""
        try:
            for i in range(len(data_pairs)):
                if data_pairs[i] == "300,300":
                    if i + 1 < len(data_pairs):
                        return int(data_pairs[i + 1].split(',')[0])
        except (ValueError, IndexError):
            pass
        return None
    
    def _calculate_overflow_underflow(self, final_weight: Optional[int]) -> Tuple[int, int]:
        """Calculate overflow and underflow amounts."""
        if final_weight is None:
            return 0, 0
            
        overflow_amount = 0
        underflow_amount = 0
        
        if final_weight > self.tolerance_limits[1]:
            overflow_amount = final_weight - self.tolerance_limits[1]
        elif final_weight < self.tolerance_limits[0]:
            underflow_amount = self.tolerance_limits[0] - final_weight
            
        return overflow_amount, underflow_amount
    
    def create_filling_session(self, parsed_data: Dict[str, Any]) -> Optional[FillingSession]:
        """
        Create a FillingSession object from parsed data.
        
        Args:
            parsed_data: Dictionary containing parsed episode data
            
        Returns:
            FillingSession object compatible with existing agents
        """
        try:
            # Add termination marker
            weight_sequence = parsed_data['weight_sequence'].copy()
            weight_sequence.append(300)  # Add termination marker
            
            # Add final weight
            if parsed_data['final_weight'] is not None:
                weight_sequence.append(parsed_data['final_weight'])
            
            session = FillingSession(
                weight_sequence=weight_sequence,
                switch_point=parsed_data['switching_point'],
                final_weight=parsed_data['final_weight'],
                episode_length=parsed_data['episode_length']
            )
            
            return session if session.is_valid() else None
            
        except Exception as e:
            logging.error(f"Error creating filling session: {e}")
            return None
    
    def get_episode_stats(self, parsed_data: Dict[str, Any]) -> Dict[str, int]:
        """Get episode statistics."""
        overflow_amount = parsed_data.get('overflow_amount', 0)
        underflow_amount = parsed_data.get('underflow_amount', 0)
        
        return {
            'valid_fillings': 1,
            'overflow_fillings': 1 if overflow_amount > 0 else 0,
            'underflow_fillings': 1 if underflow_amount > 0 else 0,
            'safe_fillings': 1 if overflow_amount == 0 and underflow_amount == 0 else 0
        }