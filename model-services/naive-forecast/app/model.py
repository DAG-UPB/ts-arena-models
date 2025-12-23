from typing import List, Union

class NaiveForecastModel:
    def predict(self, history: Union[List[float], List[List[float]]], horizon=1) -> Union[List[float], List[List[float]]]:
        if not history:
            raise ValueError("Historie darf nicht leer sein.")
        
        is_single_series = not history or not isinstance(history[0], list)
        
        if is_single_series:
            # Type assertion to satisfy the type checker
            history_single = history
            if not history_single:
                 return [0] * horizon
            last_value = history_single[-1]
            return [last_value] * horizon
        
        # It's a batch
        history_batch = history
        predictions = []
        for series in history_batch:
            if not series:
                predictions.append([0] * horizon)
            else:
                last_value = series[-1]
                predictions.append([last_value] * horizon)
        return predictions
