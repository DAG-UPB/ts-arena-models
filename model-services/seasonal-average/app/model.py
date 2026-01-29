from typing import List, Union, Optional

class SeasonalAverageModel:
    def __init__(self, num_seasons: Optional[int] = None):
        self.num_seasons = num_seasons

    def predict(self, history: Union[List[float], List[List[float]]], horizon=1, seasonality=24, offset: Union[int, List[int]] = 0) -> Union[List[float], List[List[float]]]:
        if not history:
            raise ValueError("History must not be empty.")
        if seasonality < 1:
            raise ValueError("Seasonality must be at least 1.")

        is_single_series = not history or not isinstance(history[0], list)

        if is_single_series:
            series = history
            current_offset = offset if isinstance(offset, int) else 0
            if not series:
                return [0] * horizon
            seasonal_averages = []
            for p in range(seasonality):
                target_rem = (p - current_offset) % seasonality
                values = series[target_rem::seasonality]
                if self.num_seasons is not None and self.num_seasons > 0:
                    values = values[-self.num_seasons:]
                if values:
                    avg = sum(values) / len(values)
                else:
                    avg = 0
                seasonal_averages.append(avg)
            
            start_pred_phase = (current_offset + len(series)) % seasonality
            return [seasonal_averages[(start_pred_phase + i) % seasonality] for i in range(horizon)]

        history_batch = history
        offsets = offset if isinstance(offset, list) else [offset] * len(history_batch)
        
        all_predictions = []
        for idx, series in enumerate(history_batch):
            current_offset = offsets[idx] if idx < len(offsets) else 0
            if not series:
                 all_predictions.append([0] * horizon)
                 continue
            seasonal_averages = []
            for p in range(seasonality):
                target_rem = (p - current_offset) % seasonality
                values = series[target_rem::seasonality]
                if self.num_seasons is not None and self.num_seasons > 0:
                    values = values[-self.num_seasons:]
                if values:
                    avg = sum(values) / len(values)
                else:
                    avg = 0
                seasonal_averages.append(avg)
            
            start_pred_phase = (current_offset + len(series)) % seasonality
            prediction = [seasonal_averages[(start_pred_phase + i) % seasonality] for i in range(horizon)]
            all_predictions.append(prediction)
        
        return all_predictions