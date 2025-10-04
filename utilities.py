from datetime import datetime, timedelta
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from zoneinfo import ZoneInfo

from config import (
    PM_TEMPERATURE_LOW_THRESHOLD,
    PM_TEMPERATURE_MEDIUM_THRESHOLD,
    WEEKDAYS
)


def string_to_switzerland_time(datetime_string: str) -> datetime:
    return datetime.strptime(datetime_string, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("Europe/Zurich"))


def get_weekday(date: datetime) -> str:
    return WEEKDAYS[date.weekday()]


def substract_time_seconds(time: datetime.time, seconds: int) -> datetime.time:
    dummy_datetime = datetime.combine(datetime(1970, 1, 1), time)
    result_datetime = dummy_datetime - timedelta(seconds=seconds)

    if result_datetime.date() != dummy_datetime.date():
        raise ValueError("Computation wrapped around midnight (result is the previous day)")
    return result_datetime.time()


def add_time_seconds(time: datetime.time, seconds: int) -> datetime.time:
    dummy_datetime = datetime.combine(datetime(1970, 1, 1), time)
    result_datetime = dummy_datetime + timedelta(seconds=seconds)

    if result_datetime.date() != dummy_datetime.date():
        raise ValueError("Computation wrapped around midnight (result is the next day)")
    return result_datetime.time()


def substract_time_time(time_1: datetime.time, time_2: datetime.time) -> int:
    dummy_datetime_1 = datetime.combine(datetime(1970, 1, 1), time_1)
    dummy_datetime_2 = datetime.combine(datetime(1970, 1, 1), time_2)
    return (dummy_datetime_1 - dummy_datetime_2).total_seconds()


def time_to_seconds(time: datetime.time) -> int:
    return (time.hour * 60 + time.minute) * 60 + time.second


def time_to_string(time: datetime.time) -> str:
    dummy_datetime = datetime.combine(datetime(1970, 1, 1), time)
    return dummy_datetime.strftime("%H:%M:%S")


def discretize_temperature_scalar(
    value: int,
    low_threshold: int = PM_TEMPERATURE_LOW_THRESHOLD,
    medium_threshold: int = PM_TEMPERATURE_MEDIUM_THRESHOLD
) -> str:
    if value < low_threshold:
        return "low"
    if value < medium_threshold:
        return "medium"
    return "high"
