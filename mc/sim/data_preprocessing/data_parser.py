import pandas as pd

class DataParser:
    def __init__(self, raw_data, tolerance_limits, quantization_step=10000):
        self.raw_data = raw_data
        self.tolerance_limits = tolerance_limits
        self.quantization_step = quantization_step
        self.parsed_data = []
        self.coarse_time = None
        self.fine_time = None
        self.total_time = None
        self.switching_state = None
        self.overflow_amount = 0
        self.underflow_amount = 0

    def parse_data(self, is_original=True):
        self.raw_data = self.raw_data.replace(' ', '')
        self.raw_data = self.raw_data.replace('30,30', '-1,-1')
        data_pairs = self.raw_data.split(';')
        data_pairs = [pair for pair in data_pairs if pair]

        if not self._extract_times(data_pairs):
            return None
        
        if is_original:
            self._calculate_overflow_underflow(data_pairs)
            return

        self._rearrange_pairs(data_pairs)

        for pair in data_pairs:
            decimal_value = self._process_pair(pair)
            if decimal_value is not None:
                self.parsed_data.append(decimal_value)

        self.parsed_data = self.quantize_data(self.parsed_data)
        self.parsed_data = self._remove_elements(self.parsed_data)

        self._calculate_overflow_underflow(self.parsed_data)
        
        self.parsed_data = ','.join(map(str, self.parsed_data))
        
    def is_data_valid(self):
        return self.switching_state is not None and (
            self.overflow_amount > 0 or
            self.underflow_amount > 0 or
            self.total_time > 0
        )

    def _extract_times(self, data_pairs):
        if "300,300" not in data_pairs or "-1,-1" not in data_pairs:
            return False

        last_pair = data_pairs[-1]
        try:
            fine_time, total_time = map(int, last_pair.split(','))
        except ValueError:
            return False

        self.fine_time = fine_time * 100
        self.total_time = total_time * 10
        self.coarse_time = self.total_time - self.fine_time

        for i in range(len(data_pairs) - 1, -1, -1):
            if data_pairs[i] == "-1,-1":
                if i > 0:
                    self.switching_state = int(data_pairs[i - 1].split(',')[0])
                break

        return True

    def _rearrange_pairs(self, data_pairs):
        for i in range(len(data_pairs)):
            if '300,300' in data_pairs[i]:
                if i + 1 < len(data_pairs):
                    pair_after_300 = data_pairs.pop(i + 1)
                    data_pairs.insert(i, pair_after_300)
                break

    def _process_pair(self, pair):
        elements = pair.strip().split(',')
        if len(elements) > 0:
            try:
                return int(elements[0].strip())
            except ValueError:
                return None
        return None
    
    def quantize_data(self, data):
        quantized_values = []
        for value in data:
            if pd.isna(value) or value == -1 or value == 300:
                quantized_values.append(value)
            else:
                quantized_values.append(max(0, round(value / self.quantization_step)))
        return quantized_values

    def _remove_elements(self, data):
        if len(data) > 51:
            data = data[50:]  
        if len(data) > 0:
            data.pop()
        return data

    def _calculate_overflow_underflow(self, data_pairs):
        for i in range(len(data_pairs)):
            if data_pairs[i] == "300,300":
                if i + 1 < len(data_pairs):
                    total_weight = int(data_pairs[i + 1].split(',')[0])
                    if total_weight > self.tolerance_limits[1]:
                        self.overflow_amount = total_weight - self.tolerance_limits[1]
                    elif total_weight < self.tolerance_limits[0]:
                        self.underflow_amount = self.tolerance_limits[0] - total_weight
                break

    def get_parsed_data(self):
        return {
            "raw_data": self.raw_data,
            "coarse_time": self.coarse_time,
            "fine_time": self.fine_time,
            "total_time": self.total_time,
            "switching_state": self.switching_state,
            "overflow_amount": self.overflow_amount,
            "underflow_amount": self.underflow_amount
        }


    def get_stats(self):
        stats = {
            'valid_fillings': 1 if self.is_data_valid() else 0,
            'overflow_fillings': 1 if self.overflow_amount > 0 else 0,
            'underflow_fillings': 1 if self.underflow_amount > 0 else 0,
            'safe_fillings': 1 if self.overflow_amount == 0 and self.underflow_amount == 0 else 0,
        }
        return stats
