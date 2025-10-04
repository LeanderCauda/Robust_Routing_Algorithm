import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame, Window
from datetime import datetime

from config import (
    REGION_UUIDS,
    VALID_YEARS,
    BPUIC_STANDARD_LENGTH,
    SATURDAY_INDEX,
    SUNDAY_INDEX,
    MIN_DEPARTURE_HOUR,
    MAX_ARRIVAL_HOUR,
    PM_INFERENCE_START_DATE,
    PM_INFERENCE_END_DATE,
    PM_RAIN_THRESHOLD,
    PM_TEMPERATURE_LOW_THRESHOLD,
    PM_TEMPERATURE_MEDIUM_THRESHOLD,
    PM_SAMPLE_SIZE_THRESHOLD,
    PM_PERCENTILES_ACCURACY,
    PM_AGGREGATION_COLUMNS,
    PM_VALUE_COLUMN,
    PM_EXPONENTIAL_LAMBDA,
    PM_TABLE_NAME,
    PM_DELAY_MAX_CAPPING
)


def compute_centroid(spark: SparkSession, region_uuids: list[str] = REGION_UUIDS) -> DataFrame:
    """
    Compute centroid of multiple geometries.
    """
    return spark.table("iceberg.geo.shapes")\
        .filter((F.col("level") == "district") & (F.trim(F.col("uuid")).isin(region_uuids)))\
        .select(F.expr("ST_Centroid(ST_Union_Aggr(ST_GeomFromWKB(wkb_geometry)))").alias("centroid"))\
        .select(F.expr("ST_Point(ST_X(centroid), ST_Y(centroid))").alias("centroid"))


def get_closest_site(spark: SparkSession, region_uuids: list[str] = REGION_UUIDS) -> str:
    """
    Compute the centroid of regions provided and find the closest weather station.
    """
    return spark.read.options(header=True)\
        .csv(f'/data/com-490/csv/weather_stations')\
        .withColumns({'lat': F.col('lat').cast('double'), 'lon': F.col('lon').cast('double')})\
        .filter(F.col("Active") == True)\
        .crossJoin(compute_centroid(spark, region_uuids))\
        .withColumn("distance", F.expr("ST_DistanceSphere(centroid, ST_Point(lon, lat))"))\
        .agg(F.min_by("Name", "distance").alias("closest_site"))\
        .first()["closest_site"]


def filter_sbb_stops(spark: SparkSession, region_uuids: list[str] = REGION_UUIDS) -> DataFrame:
    """
    Filter table sbb.stops to retain stops within the provided regions
    """
    stops_df = spark.table("iceberg.sbb.stops")\
        .filter(F.col("location_type").isNull())\
        .select("stop_id", "stop_lon", "stop_lat")\
        .distinct()\
        .withColumn("point", F.expr("ST_Point(stop_lon, stop_lat)"))

    shapes_df = spark.table("iceberg.geo.shapes")\
        .filter((F.col("level") == "district") & (F.trim(F.col("uuid")).isin(region_uuids)))\
        .withColumn("geometry", F.expr("ST_GeomFromWKB(wkb_geometry)"))

    return stops_df.join(shapes_df, F.expr("ST_Within(point, geometry)"))\
        .select("stop_id")\
        .distinct()


def filter_weather(
    spark: SparkSession, 
    start_date: datetime = PM_INFERENCE_START_DATE,
    end_date: datetime = PM_INFERENCE_END_DATE, 
    region_uuids: list[str] = REGION_UUIDS, 
    valid_years: list[int] = VALID_YEARS
) -> DataFrame:
    """
    Filter weather data to return the precipitation and temperature information for each (year, month, day of month, hour).
    It consider all data available before `cutoff_date` and within `valid_years` by considering the closest station to the
    provided `region_uuids`.
    """
    return spark.read.option("multiLine", True)\
        .json("/data/com-490/json/weather_history/")\
        .withColumn("observation", F.explode(F.col("observations"))).drop("observations")\
        .withColumn("observation_time_tz", F.from_utc_timestamp(F.from_unixtime(F.col("observation.valid_time_gmt")), "Europe/Zurich"))\
        .withColumns({
            "year": F.year("observation_time_tz"),
            "month": F.month("observation_time_tz"),
            "dayofmonth": F.dayofmonth("observation_time_tz"),
            "hour": F.hour("observation_time_tz"),
            "minute": F.minute("observation_time_tz")
        })\
        .filter(
            (F.col("year").isin(valid_years)) & 
            (F.col("site") == get_closest_site(spark, region_uuids)) &
            (F.col("observation_time_tz") >= start_date.strftime("%Y-%m-%d")) &
            (F.col("observation_time_tz") <= end_date.strftime("%Y-%m-%d"))
        )\
        .select("year", "month", "dayofmonth", "hour", "minute",
            F.col("observation.precip_hrly").alias("rain"),
            F.col("observation.temp").alias("temperature"),
        )\
        .groupBy(["year", "month", "dayofmonth", "hour"]).agg(
            F.max_by("rain", "minute").alias("rain"),
            F.max_by("temperature", "minute").alias("temperature")
        )


def filter_historical_data(
    spark: SparkSession,
    stops_df: DataFrame,
    start_date: datetime = PM_INFERENCE_START_DATE,
    end_date: datetime = PM_INFERENCE_END_DATE, 
    min_departure_hour: int = MIN_DEPARTURE_HOUR,
    max_arrival_hour: int = MAX_ARRIVAL_HOUR,
    valid_years: list[int] = VALID_YEARS
) -> DataFrame:
    """
    Filter historical data until `cutoff_date`.
    """
    return spark.table("iceberg.sbb.istdaten")\
        .filter(
            (F.col("unplanned") == False) &
            (F.col("failed") == False) &
            (F.col("transit") == False) &
            (F.col("arr_status").isin(["REAL", "GESCHAETZT", "PROGNOSE"])) &
            ~(F.col("product_id").isin(["Taxi", "Schiff", ""])) &
            (F.col("operating_day") >= start_date.strftime("%Y-%m-%d")) &
            (F.col("operating_day") <= end_date.strftime("%Y-%m-%d")) &
            (F.col("arr_actual").isNotNull()) &
            (F.dayofweek("operating_day") != SUNDAY_INDEX) &
            (F.dayofweek("operating_day") != SATURDAY_INDEX) &
            (F.hour("arr_time") <= max_arrival_hour) &
            (F.hour("dep_time") >= min_departure_hour) &
            (F.year("operating_day").isin(valid_years))
        )\
        .withColumns({
            "type": F.when(F.col("product_id").isin(["WM-BUS", "BUS", "Bus"]),"bus")\
                    .when(F.col("product_id").isin(["Zug", "Zahnradbahn", "Standseilbahn", "CS"]), "train")\
                    .when(F.col("product_id").isin(["Tram", "Metro", "Stadtbahn"]), "metro"),
            "delay": (F.col("arr_actual") - F.col("arr_time")).cast("long")
        })\
        .withColumn("delay", F.when(F.col("delay") < 0, 0).otherwise(F.col("delay")))\
        .withColumn("delay", F.when(F.col("delay") >= PM_DELAY_MAX_CAPPING, PM_DELAY_MAX_CAPPING).otherwise(F.col("delay")))\
        .join(stops_df.select("stop_id"), F.col("bpuic").cast("string") == F.substring("stop_id", 1, BPUIC_STANDARD_LENGTH))\
        .withColumn("stop_id", F.col("bpuic").cast("string"))\
        .select("stop_id", "operating_day", "arr_time", "delay", "type")\
        .distinct()


def merge_historical_data_weather(historical_df: DataFrame, weather_df: DataFrame) -> DataFrame:
    return historical_df.join(weather_df,
        (
            (F.year("operating_day") == F.col("year")) &
            (F.month("operating_day") == F.col("month")) &
            (F.dayofmonth("operating_day") == F.col("dayofmonth")) &
            (F.hour("arr_time") == F.col("hour"))
        ),
        how="left")\
        .drop("year", "month", "dayofmonth", "hour")\
        .withColumns({
            "year": F.year("operating_day"),
            "month": F.month("operating_day"),
            "weekofyear": F.weekofyear("operating_day"),
            "dayofmonth": F.dayofmonth("operating_day"),
            "hour": F.hour("arr_time")
        })


def handle_null_rain(historical_df: DataFrame) -> DataFrame:
    """
    The null precipitation values will be replaced by the average value for the
    same day, week or month in this order of priority.
    """
    daily_window   = Window.partitionBy("year", "month", "dayofmonth")
    weekly_window  = Window.partitionBy("year", "weekofyear")
    monthly_window = Window.partitionBy("year", "month")
    return historical_df.withColumn("rain", F.coalesce(
        F.col("rain"), 
        F.avg("rain").over(daily_window),
        F.avg("rain").over(weekly_window),
        F.avg("rain").over(monthly_window)
    ))


def handle_null_temperature(historical_df: DataFrame) -> DataFrame:
    """
    The null temperature values will be replaced by the average value at the same hour
    in the same week or month in this order of priority.
    """
    weekly_window  = Window.partitionBy("year", "weekofyear", "hour")
    monthly_window = Window.partitionBy("year", "month", "hour")
    return historical_df.withColumn("temperature", F.coalesce(
        F.col("temperature"), 
        F.avg("temperature").over(weekly_window),
        F.avg("temperature").over(monthly_window)
    ))


def discretize_rain(historical_df: DataFrame, threshold: int = PM_RAIN_THRESHOLD) -> DataFrame:
    return historical_df.withColumn("rain", F.when(F.col("rain") > threshold, True).otherwise(False))


def discretize_temperature(
    historical_df: DataFrame, 
    low_threshold: int = PM_TEMPERATURE_LOW_THRESHOLD, 
    medium_threshold: int = PM_TEMPERATURE_MEDIUM_THRESHOLD
) -> DataFrame:
    return historical_df.withColumn(
        "temperature",
        F.when(F.col("temperature") < low_threshold, "low")\
        .when(F.col("temperature") < medium_threshold, "medium")\
        .otherwise("high")
    )


def compute_non_parametric_distributions(
    historical_df: DataFrame, 
    aggregation_columns: list[str] = PM_AGGREGATION_COLUMNS,
    value_column: str = PM_VALUE_COLUMN,
    sample_size_threshold: int = PM_SAMPLE_SIZE_THRESHOLD,
    accuracy: int = PM_PERCENTILES_ACCURACY
):
    percentiles = [i / 100 for i in range(101)]
    level = 0
    distribution_df = historical_df.groupBy(aggregation_columns).agg(
        F.percentile_approx(value_column, percentiles, accuracy).alias("percentiles"),
        F.count("*").alias("sample_size"),
        F.lit(level).alias("level")
    )

    level += 1
    while distribution_df.filter(f"sample_size < {sample_size_threshold}").count() != 0:
        if level == len(aggregation_columns):
            print(f"[compute_gaussian_distributions] Distributions with less than {sample_size_threshold} samples and no aggregation columns remaining.")
            break
            
        distribution_alt_df = historical_df.groupBy(aggregation_columns[:-level]).agg(
            F.percentile_approx(value_column, percentiles, accuracy).alias("percentiles_alt"),
            F.count("*").alias("sample_size_alt"),
            F.lit(level).alias("level_alt")
        )

        distribution_df = distribution_df.join(distribution_alt_df, on=aggregation_columns[:-level], how="left")
        distribution_df = distribution_df\
            .withColumn("percentiles", F.when(F.col("sample_size") >= sample_size_threshold, F.col("percentiles")).otherwise(F.col("percentiles_alt")))\
            .withColumn("sample_size", F.when(F.col("sample_size") >= sample_size_threshold, F.col("sample_size")).otherwise(F.col("sample_size_alt")))\
            .withColumn("level", F.when(F.col("sample_size") >= sample_size_threshold, F.col("level")).otherwise(F.col("level_alt")))\
            .drop("percentiles_alt", "sample_size_alt", "level_alt")
        level += 1

    distribution_df = distribution_df.filter(F.col("sample_size") >= sample_size_threshold)
    return distribution_df
