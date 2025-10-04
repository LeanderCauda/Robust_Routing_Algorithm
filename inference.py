# # Setup

# %load_ext autoreload
# %autoreload 2

# This notebook requires spatial data processing library [Apache Sedona](https://sedona.apache.org/latest/) which can be installed using the following command.

# +
# # !pip install apache-sedona geopandas

# +
import os
import pwd
import sys
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame, Window
from sedona.spark import SedonaContext
from random import randrange

username = pwd.getpwuid(os.getuid()).pw_name
hadoopFS=os.getenv('HADOOP_FS', None)
groupName = 'A1'
namespace = 'iceberg.' + username
sharedNS = 'iceberg.com490_iceberg'

print(os.getenv('SPARK_HOME'))
print(f"hadoopFSs={hadoopFS}")
print(f"username={username}")
print(f"group={groupName}")

# +
spark = SparkSession\
    .builder\
    .appName(pwd.getpwuid(os.getuid()).pw_name)\
    .config('spark.ui.port', randrange(4040, 4440, 5))\
    .config("spark.executorEnv.PYTHONPATH", ":".join(sys.path)) \
    .config('spark.jars', f"{hadoopFS}/data/com-490/jars/iceberg-spark-runtime-3.5_2.13-1.6.1.jar")\
    .config(
        "spark.jars.packages",
        "org.apache.sedona:sedona-spark-3.5_2.13:1.7.1,"
        "org.datasyslab:geotools-wrapper:1.7.1-28.5",
    )\
    .config(
        "spark.jars.repositories",
        "https://artifacts.unidata.ucar.edu/repository/unidata-all",
    )\
    .config('spark.sql.extensions', 'org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions')\
    .config('spark.sql.catalog.iceberg', 'org.apache.iceberg.spark.SparkCatalog')\
    .config('spark.sql.catalog.iceberg.type', 'hadoop')\
    .config('spark.sql.catalog.iceberg.warehouse', f'{hadoopFS}/data/com-490/iceberg/')\
    .config('spark.sql.catalog.spark_catalog', 'org.apache.iceberg.spark.SparkSessionCatalog')\
    .config('spark.sql.catalog.spark_catalog.type', 'hadoop')\
    .config('spark.sql.catalog.spark_catalog.warehouse', f'{hadoopFS}/user/{username}/assignment-3/warehouse')\
    .config("spark.sql.warehouse.dir", f'{hadoopFS}/user/{username}/assignment-3/spark/warehouse')\
    .config('spark.eventLog.gcMetrics.youngGenerationGarbageCollectors', 'G1 Young Generation')\
    .config("spark.executor.memory", "6g")\
    .config("spark.executor.cores", "4")\
    .config("spark.executor.instances", "4")\
    .master('yarn')\
    .getOrCreate()

sedona = SedonaContext.create(spark)
# -

spark.sql(f'CREATE SCHEMA IF NOT EXISTS spark_catalog.{username}')
spark.sql(f'USE spark_catalog.{username}')

# # Delays modeling
# This notebook generates a Parquet file containing delay models stratified by group. The data is aggregated based on four features: stop identifier, hour of the day, a binary indicator for rainfall, and a categorical temperature class (low, medium, or high). For each group, the 101 percentiles of the delay distribution are computed. These percentiles will subsequently be used to estimate the cumulative distribution functions (CDFs) of delays under specific contextual conditions.

from predictive_modeling.functions import (
    filter_sbb_stops,
    filter_weather,
    filter_historical_data,
    merge_historical_data_weather,
    handle_null_rain,
    handle_null_temperature,
    discretize_rain,
    discretize_temperature,
    compute_non_parametric_distributions
)

from config import PM_TABLE_NAME

stops_df = filter_sbb_stops(spark)
weather_df = filter_weather(spark)
historical_df = filter_historical_data(spark, stops_df)
historical_df = merge_historical_data_weather(historical_df, weather_df)

historical_df = handle_null_rain(historical_df)
historical_df = handle_null_temperature(historical_df)

historical_df = discretize_rain(historical_df)
historical_df = discretize_temperature(historical_df)

historical_df = historical_df.cache()
distributions_df = compute_non_parametric_distributions(historical_df)

distributions_df.write.parquet(f'/user/{username}/assignment-3/{PM_TABLE_NAME}.parquet', mode='overwrite')

# +
# spark.stop()
