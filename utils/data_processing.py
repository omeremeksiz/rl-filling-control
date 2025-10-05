"""
Port of src/data_processor.py to utils namespace (no external src imports).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any
import pandas as pd
import numpy as np
import logging

SWITCH_TOKEN = -1
TERMINATION_TOKEN = 300

class FillingSession:
    def __init__(self, session_id: str, weight_sequence: List[int]):
        self.session_id = session_id
        self.weight_sequence = weight_sequence
        self.switch_point = self._extract_switch_point()
        self.final_weight = self._extract_final_weight()
        self.episode_length = self._calculate_episode_length()

    def _extract_switch_point(self) -> Optional[int]:
        try:
            i = self.weight_sequence.index(SWITCH_TOKEN)
            return self.weight_sequence[i - 1] if i > 0 else None
        except ValueError:
            return None

    def _extract_final_weight(self) -> Optional[int]:
        try:
            j = self.weight_sequence.index(TERMINATION_TOKEN)
            return self.weight_sequence[j - 1] if j > 0 else None
        except ValueError:
            return None

    def _calculate_episode_length(self) -> int:
        try:
            return self.weight_sequence.index(TERMINATION_TOKEN)
        except ValueError:
            return len(self.weight_sequence)

    def is_valid(self) -> bool:
        return self.switch_point is not None and self.final_weight is not None


@dataclass
class EpisodeMeta:
    raw_data: Optional[str] = None
    coarse_time: Optional[int] = None
    fine_time: Optional[int] = None
    total_time: Optional[int] = None
    overflow_amount: int = 0
    underflow_amount: int = 0
    switch_point: Optional[int] = None
    final_weight: Optional[int] = None
    episode_length: Optional[int] = None


class DataProcessor:
    def __init__(self, quantization_step: int = 10000, tolerance_limits: Optional[List[int]] = None):
        self.quantization_step = quantization_step
        self.tolerance_limits = tolerance_limits or [0, 0]
        self.sessions: List[FillingSession] = []
        self.switch_point_clusters: Dict[int, List[FillingSession]] = {}
        self.used_sessions: Dict[int, List[FillingSession]] = {}

    def load_excel(self, file_path: str) -> None:
        try:
            df = pd.read_excel(file_path)
            self._process_sessions_from_df(df)
            self._create_switch_point_clusters()
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {file_path}: {e}")

    def _process_sessions_from_df(self, df: pd.DataFrame) -> None:
        self.sessions = []
        for column in df.columns:
            seq = df[column].dropna().astype(int).tolist()
            if not seq:
                continue
            session = FillingSession(str(column), seq)
            if session.is_valid():
                self.sessions.append(session)

    def _create_switch_point_clusters(self) -> None:
        self.switch_point_clusters = {}
        for s in self.sessions:
            sp = s.switch_point
            self.switch_point_clusters.setdefault(sp, []).append(s)

    def get_available_switch_points(self) -> List[int]:
        return sorted(self.switch_point_clusters.keys())

    def get_all_available_weights(self) -> List[int]:
        all_w = set()
        for s in self.sessions:
            for w in s.weight_sequence:
                if w not in (SWITCH_TOKEN, TERMINATION_TOKEN):
                    all_w.add(w)
        return sorted(all_w)

    def get_unused_sessions_for_switch_point(self, switch_point: int) -> List[FillingSession]:
        all_s = self.switch_point_clusters.get(switch_point, [])
        used_s = self.used_sessions.get(switch_point, [])
        unused = [s for s in all_s if s not in used_s]
        if not unused:
            self._reset_cluster(switch_point)
            unused = all_s.copy()
        return unused

    def mark_session_as_used(self, switch_point: int, session: FillingSession) -> None:
        self.used_sessions.setdefault(switch_point, []).append(session)

    def _reset_cluster(self, switch_point: int) -> None:
        if switch_point in self.used_sessions:
            self.used_sessions[switch_point] = []

    # Real/TCP parsing helpers
    def parse_real_episode(self, raw_data: str, session_id: str = "real"
                           ) -> Tuple[Optional[FillingSession], Optional[EpisodeMeta], Optional[List[int]]]:
        if not raw_data:
            return None, None, None
        try:
            cleaned = raw_data.replace(' ', '').replace('30,30', '-1,-1')
            pairs = [p for p in cleaned.split(';') if p]
            if not pairs:
                return None, None, None

            timing = self._extract_timing_info(pairs)
            if not timing:
                return None, None, None

            core_sequence = self._parse_weight_sequence(pairs)
            if not core_sequence:
                return None, None, None

            final_weight = self._get_final_weight(pairs)
            overflow, underflow = self._calculate_overflow_underflow(final_weight)

            full_seq = core_sequence.copy()
            if final_weight is not None:
                full_seq.append(final_weight)
            full_seq.append(TERMINATION_TOKEN)

            session = FillingSession(session_id, full_seq)
            if not session.is_valid():
                return None, None, None

            meta = EpisodeMeta(
                raw_data=raw_data,
                coarse_time=timing['coarse_time'],
                fine_time=timing['fine_time'],
                total_time=timing['total_time'],
                overflow_amount=overflow,
                underflow_amount=underflow,
                switch_point=session.switch_point,
                final_weight=session.final_weight,
                episode_length=session.episode_length,
            )
            return session, meta, core_sequence
        except Exception as e:
            logging.error(f"Error parsing real episode: {e}")
            return None, None, None

    def _extract_timing_info(self, data_pairs: List[str]) -> Optional[Dict[str, int]]:
        try:
            if "300,300" not in data_pairs or "-1,-1" not in data_pairs:
                return None
            last_pair = data_pairs[-1]
            fine_time, total_time = map(int, last_pair.split(','))
            fine_time *= 100
            total_time *= 10
            coarse_time = total_time - fine_time
            return {
                'coarse_time': coarse_time,
                'fine_time': fine_time,
                'total_time': total_time,
            }
        except Exception:
            return None

    def _parse_weight_sequence(self, data_pairs: List[str]) -> Optional[List[int]]:
        try:
            self._rearrange_pairs(data_pairs)
            weights = []
            for pair in data_pairs:
                w = self._extract_weight_from_pair(pair)
                if w is not None:
                    weights.append(w)
            q = self._quantize_weights(weights)
            core = self._remove_elements(q)
            return core
        except Exception:
            return None

    def _rearrange_pairs(self, data_pairs: List[str]) -> None:
        for i in range(len(data_pairs)):
            if '300,300' in data_pairs[i]:
                if i + 1 < len(data_pairs):
                    pair_after_300 = data_pairs.pop(i + 1)
                    data_pairs.insert(i, pair_after_300)
                break

    def _extract_weight_from_pair(self, pair: str) -> Optional[int]:
        try:
            left = pair.strip().split(',')[0]
            return int(left)
        except Exception:
            return None

    def _quantize_weights(self, weights: List[int]) -> List[int]:
        out = []
        for w in weights:
            if w in (SWITCH_TOKEN, TERMINATION_TOKEN):
                out.append(w)
            else:
                out.append(max(0, round(w / self.quantization_step)))
        return out

    def _remove_elements(self, weights: List[int]) -> List[int]:
        if len(weights) > 51:
            weights = weights[50:]
        if weights:
            weights = weights[:-1]
        return weights

    def _get_final_weight(self, data_pairs: List[str]) -> Optional[int]:
        try:
            for i in range(len(data_pairs)):
                if data_pairs[i] == "300,300" and i + 1 < len(data_pairs):
                    return int(data_pairs[i + 1].split(',')[0])
        except Exception:
            pass
        return None

    def _calculate_overflow_underflow(self, final_weight: Optional[int]) -> Tuple[int, int]:
        if final_weight is None:
            return 0, 0
        lo, hi = self.tolerance_limits
        overflow = max(0, final_weight - hi)
        underflow = max(0, lo - final_weight)
        return overflow, underflow


