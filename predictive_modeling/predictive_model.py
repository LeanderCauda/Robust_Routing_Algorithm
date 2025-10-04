import operator
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from bisect import bisect_right
from functools import reduce
from datetime import datetime, timedelta
from scipy.stats import expon
from graph.path import Path

from config import (
    BPUIC_STANDARD_LENGTH,
    PM_EXPONENTIAL_LAMBDA,
    PM_RAIN_THRESHOLD,
    PM_TEMPERATURE_LOW_THRESHOLD,
    PM_TEMPERATURE_MEDIUM_THRESHOLD
)


class PredictiveModel:
    def __init__(self, spark: SparkSession, parquet_path: str = None, dataframe: DataFrame = None):
        self.distributions = {}
        
        if parquet_path:
            dataframe = spark.read.parquet(parquet_path)
            
        for row in dataframe.collect():
            key = (row.stop_id, row.hour, row.rain, row.temperature)
            value = (row.percentiles, row.count)
            self.distributions[key] = value

    @staticmethod
    def get_exponential_percentiles(lambda_: float = PM_EXPONENTIAL_LAMBDA) -> list[float]:
        """
        Return percentiles corresponding to an exponential distribution with parameter `lambda_`.
        """
        if lambda_ <= 0: raise ValueError("Lambda must be positive.")
        
        percentiles = np.linspace(0, 1, 101)
        values = expon.ppf(percentiles, scale=1/lambda_)
        return values.tolist()

    @staticmethod
    def get_probability(value: float, percentile_values: list[float]) -> float:
        if value <= percentile_values[0]: return 0
        if value >= percentile_values[-1]: return 1
    
        idx = bisect_right(percentile_values, value)
        percentiles = list(range(101))
        x0, x1 = percentile_values[idx - 1], percentile_values[idx]
        p0, p1 = percentiles[idx - 1], percentiles[idx]
    
        if x0 == x1:
            cdf = (p0 + p1) / 2.0
        else:
            cdf = p0 + (p1 - p0) * (value - x0) / (x1 - x0)
        return cdf / 100.0

    def get_percentiles(self, stop_id: str, hour: int, rain: bool, temperature: str) -> tuple[list[float], int]:
        key = (stop_id, hour, rain, temperature)
        if key not in self.distributions:
            print(f"Key {key} don't match any distributions, fallback to exponential distribution.")
            return (PredictiveModel.get_exponential_percentiles(), -1)
        return self.distributions[key]

    def get_path_confidence(self, path: Path, rain: bool, temperature: str) -> float:
        i = 0
        confidence = 1.0
        print(f"Computing path confidence | {path.trips[0].source.stop_name} -> {path.trips[-1].target.stop_name}")
    
        while i < len(path.trips) and path.trips[i].is_walking(): i += 1
    
        while i < len(path.trips):        
            j = i + 1
            d = 0.0
            while j < len(path.trips) and path.trips[j].is_walking():
                d += (path.trips[j].target_arrival_time - path.trips[j].source_departure_time).total_seconds()
                j += 1
            
            target_arrival_time = path.trips[i].target_arrival_time + timedelta(seconds=d)
            
            percentile_values, count = self.get_percentiles(
                stop_id=path.trips[i].target.stop_id[:BPUIC_STANDARD_LENGTH],
                hour=target_arrival_time.time().hour,
                rain=rain,
                temperature=temperature
            )
    
            if j >= len(path.trips):
                source_departure_time = path.expected_arrival_datetime
            else:
                source_departure_time = path.trips[j].source_departure_time
    
            max_delay = (source_departure_time - target_arrival_time).total_seconds()
            probability = PredictiveModel.get_probability(max_delay, percentile_values)
            confidence = confidence * probability
    
            if j >= len(path.trips):
                print(f"[{path.trips[i].trip_id}] -> ({path.trips[i].target.stop_name}) = Arrival | P(d<{max_delay})={probability} ({count} samples)")
            else:
                print(f"[{path.trips[i].trip_id}] -> ({path.trips[i].target.stop_name}) -> [{path.trips[j].trip_id}] | P(d<{max_delay})={probability} ({count} samples)")
            i = j
        
        return confidence
