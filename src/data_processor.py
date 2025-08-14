"""
Data processing module for container filling control system.
Handles loading and preprocessing of filling session data.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
# Hardcoded special tokens
SWITCH_TOKEN = -1
TERMINATION_TOKEN = 300


class FillingSession:
    """Represents a single container filling session with extracted metadata."""
    
    def __init__(self, session_id: str, weight_sequence: List[int]):
        self.session_id = session_id
        self.weight_sequence = weight_sequence
        self.switch_point = self._extract_switch_point()
        self.final_weight = self._extract_final_weight()
        self.episode_length = self._calculate_episode_length()
    
    def _extract_switch_point(self) -> Optional[int]:
        """Extract the switching point (last fast-mode weight)."""
        try:
            switch_index = self.weight_sequence.index(SWITCH_TOKEN)
            if switch_index > 0:
                return self.weight_sequence[switch_index - 1]
        except ValueError:
            pass
        return None
    
    def _extract_final_weight(self) -> Optional[int]:
        """Extract the final weight (value before termination)."""
        try:
            termination_index = self.weight_sequence.index(TERMINATION_TOKEN)
            if termination_index > 0:
                return self.weight_sequence[termination_index - 1]
        except ValueError:
            pass
        return None
    
    def _calculate_episode_length(self) -> int:
        """Calculate the total episode length (steps until termination)."""
        try:
            return self.weight_sequence.index(TERMINATION_TOKEN)
        except ValueError:
            return len(self.weight_sequence)
    
    def is_valid(self) -> bool:
        """Check if the session has valid switch point and final weight."""
        return self.switch_point is not None and self.final_weight is not None


class DataProcessor:
    """Handles loading and preprocessing of filling session data."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.sessions: List[FillingSession] = []
        self.switch_point_clusters: Dict[int, List[FillingSession]] = {}
        self.used_sessions: Dict[int, List[FillingSession]] = {}
    
    def load_data(self) -> None:
        """Load and process the Excel data file."""
        try:
            df = pd.read_excel(self.file_path)
            self._process_sessions(df)
            self._create_switch_point_clusters()
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {self.file_path}: {e}")
    
    def _process_sessions(self, df: pd.DataFrame) -> None:
        """Process each column as a separate filling session."""
        self.sessions = []
        
        for column in df.columns:
            # Convert to list and filter out NaN values
            weight_sequence = df[column].dropna().astype(int).tolist()
            
            if weight_sequence:
                session = FillingSession(str(column), weight_sequence)
                if session.is_valid():
                    self.sessions.append(session)
    
    def _create_switch_point_clusters(self) -> None:
        """Group sessions by their switch points for training."""
        self.switch_point_clusters = {}
        
        for session in self.sessions:
            switch_point = session.switch_point
            if switch_point not in self.switch_point_clusters:
                self.switch_point_clusters[switch_point] = []
            self.switch_point_clusters[switch_point].append(session)
    
    def get_available_switch_points(self) -> List[int]:
        """Get all available switching points from the data."""
        return sorted(list(self.switch_point_clusters.keys()))
    
    def get_all_available_weights(self) -> List[int]:
        """Get all available weights from the data."""
        all_weights = set()
        for session in self.sessions:
            # Add all weights from the sequence (excluding special tokens)
            for weight in session.weight_sequence:
                if weight != SWITCH_TOKEN and weight != TERMINATION_TOKEN:
                    all_weights.add(weight)
        return sorted(list(all_weights))
    
    def get_sessions_for_switch_point(self, switch_point: int) -> List[FillingSession]:
        """Get all sessions that used a specific switch point."""
        return self.switch_point_clusters.get(switch_point, [])
    
    def get_unused_sessions_for_switch_point(self, switch_point: int) -> List[FillingSession]:
        """Get unused sessions for a specific switch point."""
        all_sessions = self.switch_point_clusters.get(switch_point, [])
        used_sessions = self.used_sessions.get(switch_point, [])
        
        # Return sessions that haven't been used yet
        unused_sessions = [s for s in all_sessions if s not in used_sessions]
        
        # If no unused sessions, reset the cluster
        if not unused_sessions:
            self._reset_cluster(switch_point)
            unused_sessions = all_sessions.copy()
        
        return unused_sessions
    
    def mark_session_as_used(self, switch_point: int, session: FillingSession) -> None:
        """Mark a session as used for a specific switch point."""
        if switch_point not in self.used_sessions:
            self.used_sessions[switch_point] = []
        self.used_sessions[switch_point].append(session)
    
    def _reset_cluster(self, switch_point: int) -> None:
        """Reset the used sessions for a specific switch point."""
        if switch_point in self.used_sessions:
            self.used_sessions[switch_point] = []
    
    def get_session_statistics(self) -> Dict:
        """Get statistics about the loaded sessions."""
        if not self.sessions:
            return {}
        
        switch_points = [s.switch_point for s in self.sessions if s.switch_point is not None]
        final_weights = [s.final_weight for s in self.sessions if s.final_weight is not None]
        episode_lengths = [s.episode_length for s in self.sessions]
        
        return {
            'total_sessions': len(self.sessions),
            'unique_switch_points': len(set(switch_points)),
            'switch_point_range': (min(switch_points), max(switch_points)) if switch_points else None,
            'final_weight_range': (min(final_weights), max(final_weights)) if final_weights else None,
            'avg_episode_length': np.mean(episode_lengths) if episode_lengths else 0
        } 