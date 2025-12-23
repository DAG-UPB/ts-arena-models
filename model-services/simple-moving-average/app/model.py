from typing import List, Union

class SimpleMovingAverageModel:
    def predict(self, history: Union[List[float], List[List[float]]], horizon=1) -> Union[List[float], List[List[float]]]:
        if not history:
            raise ValueError("Historie darf nicht leer sein.")

        is_single_series = not history or not isinstance(history[0], list)
        
        if is_single_series:
            series = history
            if not series:
                return [0] * horizon
            avg = sum(series) / len(series)
            return [avg] * horizon

        history_batch = history
        predictions = []
        for series in history_batch:
            if not series:
                predictions.append([0] * horizon)
            else:
                avg = sum(series) / len(series)
                predictions.append([avg] * horizon)
        
        return predictions
