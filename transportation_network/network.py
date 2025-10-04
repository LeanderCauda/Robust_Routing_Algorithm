import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from datetime import datetime

from config import (
    WEEKDAYS
)


def get_most_recent_publication_date(spark: SparkSession, current_date: datetime) -> datetime:
    """
    Compute the most recent publication date of GTFS static asset that occured before
    or on the current date.
    """
    result = spark.table("iceberg.sbb.stops")\
        .filter(F.col("pub_date") <= current_date.strftime("%Y-%m-%d"))\
        .select(F.max("pub_date").alias("pub_date"))\
        .first()
    
    if not result:
        error_message = f"No publication of static assets made on or before {current_date.strftime('%Y-%m-%d')}"
        raise ValueError(error_message)
    return result.pub_date


def filter_sbb_stops(spark: SparkSession, publication_date: datetime, region_uuids: list[str]) -> DataFrame:
    """
    Filter table sbb.stops to retain stops within the provided regions and information published
    on the provided publication date
    """
    stops_df = spark.table("iceberg.sbb.stops")\
        .filter(
            (F.col("pub_date") == publication_date.strftime("%Y-%m-%d")) &
            (F.col("location_type").isNull())
        )\
        .withColumn("point", F.expr("ST_Point(stop_lon, stop_lat)"))

    shapes_df = spark.table("iceberg.geo.shapes")\
        .filter((F.col("level") == "district") & (F.trim(F.col("uuid")).isin(region_uuids)))\
        .withColumn("geometry", F.expr("ST_GeomFromWKB(wkb_geometry)"))

    return stops_df.join(shapes_df, F.expr("ST_Within(point, geometry)"))\
        .select("stop_id", "stop_name", "stop_lat", "stop_lon", 
            F.regexp_replace(F.col("parent_station"), "Parent", "").alias("parent_stop_id")
        )\
        .distinct()


def filter_sbb_calendar(spark: SparkSession, publication_date: datetime, arrival_datetime: datetime) -> DataFrame:
    """
    Filter table sbb.calendar to return service information published on the provided publication
    date and that match the travel date.

    Note : while the problem only concerns typical business day, it is not possible to apply the
    filter on weekdays here since the actual values could be shifted later in the processing of
    sbb.stop_times
    """
    return spark.table("iceberg.sbb.calendar")\
        .filter(
            (F.col("pub_date")   == publication_date.strftime("%Y-%m-%d")) &
            (F.col("start_date") <= arrival_datetime.strftime("%Y-%m-%d")) & 
            (F.col("end_date")   >= arrival_datetime.strftime("%Y-%m-%d"))
        )\
        .select("service_id", *WEEKDAYS)\
        .distinct()


def filter_sbb_trips(spark: SparkSession, publication_date: datetime) -> DataFrame:
    """
    Filter table sbb.trips to return trips published on the provided publication date
    """
    return spark.table("iceberg.sbb.trips")\
        .filter(F.col("pub_date") == publication_date.strftime("%Y-%m-%d"))\
        .distinct()


def filter_sbb_routes(spark: SparkSession, publication_date: datetime) -> DataFrame:
    """
    Filter table sbb.routes to return routes published on the provided publication date
    """
    return spark.table("iceberg.sbb.routes")\
        .filter(F.col("pub_date")   == publication_date.strftime("%Y-%m-%d"))\
        .distinct()


def shift_invalid_times(stop_times_df: DataFrame) -> DataFrame:
    """
    Table sbb.stop_times may contain arrival and departure times in an incorrect format with
    time value exceeding 24:00:00. This is allowed and used by the GTFS standard to indicate
    trips occuring after midnight but still related to a given service day (e.g. trip starting
    at 23:45:00 on a given day and finishing at 01:15:00 the next day, for the last stop, the
    arrival_time value could be 25:15:00).

    The route planning algorithm expect the real time of the trip, this function will shift the
    time to valid value (between 00:00:00 and 24:00:00) while also updating the values of the
    weekday fields
    """
    stop_times_df = stop_times_df\
        .withColumn("shift", F.floor(F.split("departure_time", ":").getItem(0).cast("int") / 24))\
        .withColumn("departure_time", F.format_string(
                "%02d:%02d:%02d",
                F.pmod(F.split("departure_time", ":").getItem(0).cast("int"), 24),
                F.split("departure_time", ":").getItem(1).cast("int"),
                F.split("departure_time", ":").getItem(2).cast("int")
            )
        )\
        .withColumn("arrival_time", F.format_string(
                "%02d:%02d:%02d",
                F.pmod(F.split("arrival_time", ":").getItem(0).cast("int"), 24),
                F.split("arrival_time", ":").getItem(1).cast("int"),
                F.split("arrival_time", ":").getItem(2).cast("int")
            )
        )

    for i, weekday in enumerate(WEEKDAYS):
        shifted_field = F.when(F.col("shift") == 0, F.col(WEEKDAYS[i]))
        for j in range(1, len(WEEKDAYS)):
            shifted_field = shifted_field.when(F.col("shift") == j, F.col(WEEKDAYS[(i - j) % len(WEEKDAYS)]))
        stop_times_df = stop_times_df.withColumn(f"{weekday}_shifted", shifted_field)

    stop_times_df = stop_times_df.drop(*WEEKDAYS)
    for weekday in WEEKDAYS:
        stop_times_df = stop_times_df.withColumnRenamed(f"{weekday}_shifted", weekday)
    return stop_times_df


def filter_sbb_stop_times(
    spark: SparkSession, 
    stops_df: DataFrame, 
    calendar_df: DataFrame,
    trips_df: DataFrame,
    routes_df: DataFrame,
    publication_date: datetime, 
    arrival_datetime: datetime,
    min_departure_hour: int,
    max_arrival_hour: int
) -> DataFrame:
    """
    Filter table sbb.stop_times to retain only stop times published on the provided publication date. The table
    is joined with relevant information from sbb.stops, sbb.trips, sbb.calendar and sbb.routes to enrich the
    information.
    """
    stop_times_df = spark.table("iceberg.sbb.stop_times")\
        .filter((F.col("pub_date") == publication_date.strftime("%Y-%m-%d")))\
        .join(stops_df, "stop_id", "inner")\
        .join(trips_df, "trip_id", "inner")\
        .join(calendar_df, "service_id", "inner")\
        .join(routes_df, "route_id", "inner")

    stop_times_df = shift_invalid_times(stop_times_df)
    
    return stop_times_df\
        .filter(
            # Remove trips arriving after our desired arrival time
            (F.col("arrival_time") <= arrival_datetime.strftime("%H:%M:%S")) &
            # Filter for journeys occuring on a typical business day at reasonable hours
            (F.col("monday") == True) &  (F.col("tuesday") == True) & 
            (F.col("wednesday") == True) & (F.col("thursday") == True) & 
            (F.col("friday") == True) & 
            (F.hour("departure_time") >= min_departure_hour) & 
            (F.hour("arrival_time") <= max_arrival_hour)
        )\
        .drop(*WEEKDAYS)


def filter_sbb_transfers(spark: SparkSession, publication_date: datetime) -> DataFrame:
    """
    Filter table sbb.transfers to return information published on the provided publication date
    """
    return spark.table("iceberg.sbb.transfers")\
        .filter(F.col("pub_date") == publication_date.strftime("%Y-%m-%d"))\
        .select("from_stop_id", "to_stop_id", F.col("min_transfer_time").alias("transfer_time"))


def build_timetable(
    spark: SparkSession,
    arrival_datetime: datetime,
    region_uuids: list[str],
    min_departure_hour: int,
    max_arrival_hour: int
) -> tuple[DataFrame, DataFrame]:
    """
    Merge information from different tables to construct a timetable of the transportation
    network within the provided regions and with trips arriving before the arrival time
    """
    publication_date = get_most_recent_publication_date(spark, arrival_datetime)
    
    stops_df      = filter_sbb_stops(spark, publication_date, region_uuids)
    calendar_df   = filter_sbb_calendar(spark, publication_date, arrival_datetime)
    trips_df      = filter_sbb_trips(spark, publication_date)
    routes_df     = filter_sbb_routes(spark, publication_date)
    transfer_df   = filter_sbb_transfers(spark, publication_date)
    stop_times_df = filter_sbb_stop_times(
        spark,
        stops_df,
        calendar_df,
        trips_df,
        routes_df,
        publication_date,
        arrival_datetime,
        min_departure_hour,
        max_arrival_hour
    )
    return stop_times_df, stops_df, transfer_df


def build_walking_network(
    stops_df: DataFrame, 
    transfer_df: DataFrame, 
    maximum_walking_distance: int, 
    walking_speed: float
) -> DataFrame:
    """
    Build a walking network by joining stops within reach of each other based on the maximum walking distance.

    Note: The table sbb.transfers provide information to override the minimum transfer time needed between some stops.
    Indeed the default transfer time is based on the distance between the two stops and the provided walking speed.
    However, in some situation (e.g. train station), the Euclidean distance is not a good measure of time needed to
    catch another trip. Thus the time it takes to walk between two stops is max(distance / speed, transfer time).

    Note: Walking edges are relative in time, thus we only store the time it takes relative to the walking speed in
    the field "source_arrival_time" and nothing in the field "target_departure_time".
    """
    return stops_df.alias("A")\
        .join(stops_df.alias("B"),
            ((F.col("A.stop_id") != F.col("B.stop_id")) &
            (F.expr("ST_DistanceSphere(ST_Point(A.stop_lon, A.stop_lat), ST_Point(B.stop_lon, B.stop_lat))") <= maximum_walking_distance))
        )\
        .join(transfer_df.alias("C"), (
                (F.col("A.stop_id") == F.col("C.from_stop_id")) &
                (F.col("B.stop_id") == F.col("C.to_stop_id"))
        ), how="left")\
        .select(
            F.col("A.stop_id").alias("source_stop_id"),
            F.col("A.stop_name").alias("source_stop_name"),
            F.col("A.stop_lon").alias("source_stop_lon"),
            F.col("A.stop_lat").alias("source_stop_lat"),
            F.col("A.parent_stop_id").alias("source_parent_stop_id"),
            F.col("B.stop_id").alias("target_stop_id"),
            F.col("B.stop_name").alias("target_stop_name"),
            F.col("B.stop_lon").alias("target_stop_lon"),
            F.col("B.stop_lat").alias("target_stop_lat"),
            F.col("B.parent_stop_id").alias("target_parent_stop_id"),
            F.lit("walking").alias("trip_id"),
            F.lit("Walking").alias("trip_description"),
            F.lit(None).alias("source_stop_sequence"),
            F.col("C.transfer_time").alias("transfer_time"),
            F.lit(None).alias("target_departure_time"),
            (F.expr("""
                ST_DistanceSphere(ST_Point(A.stop_lon, A.stop_lat), ST_Point(B.stop_lon, B.stop_lat))
            """).cast("double") / F.lit(walking_speed)).cast("long").alias("source_arrival_time")
        )\
        .fillna({"transfer_time": 0})


def build_network(
    stops_df: DataFrame,
    stop_times_df: DataFrame,
    transfer_df: DataFrame,
    maximum_walking_distance: int, 
    walking_speed: float
) -> DataFrame:
    """
    Build a transportation network by joining stops connected by a trip and then adding the walking network.

    Note: Edge are created in a reverse way, meaning that if a trip goes from stop B to stop A, the created edge will be
    A --> B and not B --> A. Indeed, the network will be used to answer shortest path problem with a given arrival time. 
    """
    network_df = stop_times_df.alias("A")\
        .join(stop_times_df.alias("B"),
            ((F.col("A.stop_id") != F.col("B.stop_id")) &
            (F.col("A.trip_id") == F.col("B.trip_id")) &
            (F.col("A.stop_sequence") == F.col("B.stop_sequence") + 1) & 
            (F.col("A.direction_id") == F.col("B.direction_id")))
        )\
        .select(
            F.col("A.stop_id").alias("source_stop_id"),
            F.col("A.stop_name").alias("source_stop_name"),
            F.col("A.stop_lon").alias("source_stop_lon"),
            F.col("A.stop_lat").alias("source_stop_lat"),
            F.col("A.parent_stop_id").alias("source_parent_stop_id"),
            F.col("B.stop_id").alias("target_stop_id"),
            F.col("B.stop_name").alias("target_stop_name"),
            F.col("B.stop_lon").alias("target_stop_lon"),
            F.col("B.stop_lat").alias("target_stop_lat"),
            F.col("B.parent_stop_id").alias("target_parent_stop_id"),
            F.col("A.trip_id").alias("trip_id"),
            F.format_string("%s - %s (%s)", "A.route_short_name", "A.route_desc", "A.trip_headsign").alias("trip_description"),
            F.col("A.stop_sequence").alias("source_stop_sequence"), # target_stop_sequence = source_stop_sequence - 1
            F.lit(0).alias("transfer_time"),
            F.col("B.departure_time").alias("target_departure_time"),
            F.col("A.arrival_time").alias("source_arrival_time")
        )
    
    return network_df.union(build_walking_network(stops_df, transfer_df, maximum_walking_distance, walking_speed))
