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

# # Validation
#
# To validate our approach to delay modeling, we compute the percentiles of the delay distribution separately for each group over two non-overlapping time periods: a training set spanning from January 1, 2022 to March 31, 2024, and a validation set covering the period from April 1, 2024 to December 31, 2024.

# +
SEED = 490

import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from tqdm import tqdm
from scipy.stats import ks_2samp
# -

random.seed(490)

# +
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

from predictive_modeling.predictive_model import PredictiveModel
# -

from config import (
    PM_TRAINING_START_DATE,
    PM_TRAINING_END_DATE,
    PM_VALIDATION_START_DATE,
    PM_VALIDATION_END_DATE,
    PM_SAMPLE_SIZE_THRESHOLD,
    PM_VALUE_COLUMN,
    PM_AGGREGATION_COLUMNS
)

stops_df = filter_sbb_stops(spark)

# +
weather_df_train = filter_weather(
    spark,
    start_date=PM_TRAINING_START_DATE,
    end_date=PM_TRAINING_END_DATE
)
historical_df_train = filter_historical_data(
    spark, 
    stops_df,
    start_date=PM_TRAINING_START_DATE,
    end_date=PM_TRAINING_END_DATE
)

historical_df_train = merge_historical_data_weather(historical_df_train, weather_df_train)
historical_df_train = handle_null_rain(historical_df_train)
historical_df_train = handle_null_temperature(historical_df_train)

historical_df_train = discretize_rain(historical_df_train)
historical_df_train = discretize_temperature(historical_df_train)

historical_df_train = historical_df_train.cache()
distributions_df_train = compute_non_parametric_distributions(historical_df_train)
distributions_df_train = distributions_df_train.cache()

# +
weather_df_val = filter_weather(
    spark,
    start_date=PM_VALIDATION_START_DATE,
    end_date=PM_VALIDATION_END_DATE
)
historical_df_val = filter_historical_data(
    spark, 
    stops_df,
    start_date=PM_VALIDATION_START_DATE,
    end_date=PM_VALIDATION_END_DATE
)

historical_df_val = merge_historical_data_weather(historical_df_val, weather_df_val)
historical_df_val = handle_null_rain(historical_df_val)
historical_df_val = handle_null_temperature(historical_df_val)

historical_df_val = discretize_rain(historical_df_val)
historical_df_val = discretize_temperature(historical_df_val)

historical_df_val = historical_df_val.cache()
distributions_df_val = compute_non_parametric_distributions(historical_df_val)
distributions_df_val = distributions_df_val.cache()


# -
def sample_group(
    distribution_df: DataFrame, 
    aggregation_columns: list[str] = PM_AGGREGATION_COLUMNS,
    sample_size_threshold: int = PM_SAMPLE_SIZE_THRESHOLD
) -> tuple:
    if not all([c in distribution_df.columns for c in aggregation_columns]):
        raise ValueError(f"Some column(s) amomg {aggregation_columns} are not in the dataframe.")

    rows = distribution_df.groupBy(PM_AGGREGATION_COLUMNS)\
        .agg(F.count("*").alias("sample_size"))\
        .filter(F.col("sample_size") >= sample_size_threshold)\
        .select(PM_AGGREGATION_COLUMNS)\
        .distinct()\
        .rdd.takeSample(withReplacement=False, num=1, seed=SEED)

    if not rows:
        raise ValueError(f"No group has at least {PM_SAMPLE_SIZE_THRESHOLD} samples")
    return {c: rows[0][c] for c in aggregation_columns}


def sample_delay_distributions(
    train_df: DataFrame,
    val_df: DataFrame,
    sample_size_threshold: int = PM_SAMPLE_SIZE_THRESHOLD,
    value_column: str = PM_VALUE_COLUMN,
):
    group = sample_group(val_df, sample_size_threshold=sample_size_threshold)
    filters = reduce(lambda x, y: x & y, [F.col(k) == v for (k, v) in group.items()])
    
    count_train = train_df.filter(filters).count()
    if count_train < sample_size_threshold:
        raise ValueError(f"Not enough samples in train distributions for {group}")

    delays_train = train_df.filter(filters).select(value_column).collect()
    delays_val = val_df.filter(filters).select(value_column).collect()
    statistic, p_value = ks_2samp(delays_train, delays_val)
    return group, statistic, p_value


# ## Similarity of distributions

# +
N_ITERATIONS = 1000
SAMPLE_SIZE_THRESHOLD = 300

metrics = []
for n in tqdm(range(N_ITERATIONS)):
    try:
        group, statistic, p_value = sample_delay_distributions(
            historical_df_train,
            historical_df_val,
            sample_size_threshold=SAMPLE_SIZE_THRESHOLD
        )
        metrics.append((group, statistic[0], p_value[0]))
    except ValueError as error:
        pass

metrics_df = pd.DataFrame(metrics, columns=["group", "statistic", "p_value"])

# +
ALPHA = 0.05
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

fig.suptitle("Two-sample Kolmogorov-Smirnov test")

sns.histplot(metrics_df["p_value"], bins=int(math.sqrt(len(metrics_df))), kde=True, ax=axes[0])
axes[0].axvline(ALPHA, color='red', linestyle='--', label=f"Alpha = {ALPHA}")
axes[0].set_title(f"p-values distribution ({(metrics_df['p_value'] < ALPHA).mean()*100:.2f}% < Alpha)")
axes[0].set_xlabel("p-value")
axes[0].set_ylabel("Frequency")
axes[0].legend()

sns.histplot(metrics_df["statistic"], bins=int(math.sqrt(len(metrics_df))), kde=True, ax=axes[1])
axes[1].set_title("Statistics distribution")
axes[1].set_xlabel("Statistic")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.show()
# -

# The two-sample Kolmogorov–Smirnov (KS) test [1] is a non-parametric statistical test used to assess whether two samples are drawn from the same continuous distribution. It compares the empirical cumulative distribution functions (ECDFs) of the two samples and quantifies the maximum absolute difference between them. We can observe on the left plot that ~85% of the performed tests yielded p-values below the threshold $\alpha=0.05$ suggesting changes in the distributions of delays between the training and validation period.
#
# The values of the KS statistics can be interpreted as the maximum error in estimation of probability $P(\text{delay} \leq X)$ between the train and the validation distributions. The values are concentrated towards low values around $0.1$. This remains non-negligeable but acceptable for a practical use. Nevertheless, we have to keep in mind that the robustness and generalizability of models trained on historical data may be limited when applied to the future.
#
# [1] Kolmogorov-Smirnov Test: A Practical Guide, Arize AI. Accessed: May 14, 2025. [Online]. Available: https://arize.com/blog-course/kolmogorov-smirnov-test/

# +
path_lengths = np.arange(1, 6)
true_prob = 0.9
abs_error = 0.1

true_path_probs = true_prob ** path_lengths
upper_estimates = (true_prob + abs_error) ** path_lengths
lower_estimates = (true_prob - abs_error) ** path_lengths

plt.figure(figsize=(7, 4))
plt.plot(path_lengths, true_path_probs, label='Confidence', linestyle='--')
plt.fill_between(path_lengths, lower_estimates, upper_estimates, alpha=0.25, label='Error bounds')
plt.title(f'Path confidence error by number of transfers (true probability = {true_prob}, absolute error = {abs_error})')
plt.xlabel('Number of transfers')
plt.ylabel('Confidence')
plt.legend()
plt.grid(True)
plt.show()
# -

# We observe that even a modest estimation error of 0.1 at each transfer can significantly affect the overall confidence in a given path. However, it is important to note that this sensitivity is an inherent characteristic of probabilistic path modeling: regardless of model accuracy, the precision of confidence estimates naturally decreases as the number of transfers increases.

# ## Calibration of Cumulative Distribution Functions (CDFs)

predictive_model = PredictiveModel(spark, None, distributions_df_train)


def sample_both_cdf(
    predictive_model: PredictiveModel, 
    historical_df_val: DataFrame, 
    aggregation_columns: list[str] = PM_AGGREGATION_COLUMNS,
    value_column: str = PM_VALUE_COLUMN
):
    key = random.choice(list(predictive_model.distributions.keys()))
    percentiles = predictive_model.distributions[key][0]

    idx = random.choice(range(len(percentiles)))
    value = percentiles[idx]

    filters = reduce(lambda x, y: x & y, [F.col(aggregation_columns[i]) == key[i] for i in range(len(key))])
    
    total_count = historical_df_val.filter(filters).count()
    below_count = historical_df_val.filter(filters).filter(F.col(value_column) <= value).count()

    train_probability = idx / 100.0
    val_probability = below_count / total_count if total_count != 0.0 else 0.0
    return train_probability, val_probability


# +
N_ITERATIONS = 100

metrics = []
for n in tqdm(range(N_ITERATIONS)):
    try:
        train_probability, val_probability = sample_both_cdf(predictive_model, historical_df_val)
        metrics.append((train_probability, "train"))
        metrics.append((val_probability, "val"))
    except ValueError as error:
        pass

metrics_df = pd.DataFrame(metrics, columns=["probability", "group"])
# -

plt.figure(figsize=(7, 4))
plt.title("Distribution of estimated probabilities")
sns.kdeplot(data=metrics_df, x="probability", hue="group", fill=True, alpha=0.4, legend=True)
plt.show()

# We conducted multiple iterations of the following procedure: a group was randomly sampled from the training distribution, and within this group, a specific percentile value was selected. The empirical cumulative distribution function (ECDF) was then computed for the corresponding group in the validation data. This process yielded paired probability estimates derived respectively from the training and validation datasets.
#
# We observe that the distributions of probability estimates from both the training and validation datasets are broadly similar in terms of central tendency and overall shape. Despite prior evidence of theoretical invalidation for certain groups, this empirical similarity suggests that the probability estimates may still be practically reliable and useful for our application.

# +
# spark.stop()
