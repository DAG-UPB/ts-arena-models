from typing import List, Union

class SeasonalAverageModel:
    def predict(self, history: Union[List[float], List[List[float]]], horizon=1, seasonality=24) -> Union[List[float], List[List[float]]]:
        if not history:
            raise ValueError("History must not be empty.")
        if seasonality < 1:
            raise ValueError("Seasonality must be at least 1.")

        is_single_series = not history or not isinstance(history[0], list)

        if is_single_series:
            series = history
            if not series:
                return [0] * horizon
            seasonal_averages = []
            for i in range(seasonality):
                values = series[i::seasonality]
                if values:
                    avg = sum(values) / len(values)
                else:
                    avg = 0
                seasonal_averages.append(avg)
            return [seasonal_averages[i % seasonality] for i in range(horizon)]

        history_batch = history
        all_predictions = []
        for series in history_batch:
            if not series:
                 all_predictions.append([0] * horizon)
                 continue
            seasonal_averages = []
            for i in range(seasonality):
                values = series[i::seasonality]
                if values:
                    avg = sum(values) / len(values)
                else:
                    avg = 0
                seasonal_averages.append(avg)
            prediction = [seasonal_averages[i % seasonality] for i in range(horizon)]
            all_predictions.append(prediction)
        
        return all_predictions