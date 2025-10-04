from datetime import datetime

# +
WEEKDAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

# Default region UUIDs which corresond to the Lausanne area
REGION_UUIDS = [
    "a7a21b73-6ffe-4fbf-a635-6e2b961f3072", # Lausanne
    "e168fd57-f57a-4075-a350-0dcfbb55147f"  # Ouest Lausannois
]

# Index of saturday in the week (PySpark functions)
SATURDAY_INDEX = 7

# Index of sunday in the week (PySpark functions)
SUNDAY_INDEX = 1

# Minimum departure hour of a trip
MIN_DEPARTURE_HOUR = 7

# Maximum arrival hour of a trip
MAX_ARRIVAL_HOUR = 20

# Years with full weather data that can be used for predictive modeling
VALID_YEARS = [2022, 2023, 2024]

# Default transfer time in seconds required between two trips (e.g. from Metro M1 to Bus 17) at a given stop
DEFAULT_TRANSFER_TIME = 120

# Default walking speed in m/s
WALKING_SPEED = 50 / 60 

# Standard length of the BPUIC (stop identifiers)
BPUIC_STANDARD_LENGTH = 7

# Dates for computation of historical delays for inference
PM_INFERENCE_START_DATE = datetime.strptime("2022-01-01", "%Y-%m-%d")
PM_INFERENCE_END_DATE = datetime.strptime("2024-12-31", "%Y-%m-%d")

# Dates for computation of historical delays for training
PM_TRAINING_START_DATE = datetime.strptime("2022-01-01", "%Y-%m-%d")
PM_TRAINING_END_DATE = datetime.strptime("2024-03-31", "%Y-%m-%d")

# Dates for computation of historical delays for validation
PM_VALIDATION_START_DATE = datetime.strptime("2024-04-01", "%Y-%m-%d")
PM_VALIDATION_END_DATE = datetime.strptime("2024-12-31", "%Y-%m-%d")

# Threshold for discretization of the rain features. Values below this
# threshold will be mapped to False and values higher to True
PM_RAIN_THRESHOLD = 2.5

# Thresholds for discretization of the temperature features. Values will
# be mapped to "low", "medium" or "high"
PM_TEMPERATURE_LOW_THRESHOLD = 6
PM_TEMPERATURE_MEDIUM_THRESHOLD = 14

# Minimum sample size required to estimate percentiles among a group of
# features. Please read explanation in `compute_non_parametric_distributions`
PM_SAMPLE_SIZE_THRESHOLD = 30

# The percentiles accuracy defines the inverse of percentiles estimation
# relative error. Example : 1000 => 1/1000 = 0.001 => relative error of 0.001
PM_PERCENTILES_ACCURACY = 1000

# Lambda parameter of the exponential distribution used as fallback when the
# percentiles are not available for a group of features.
PM_EXPONENTIAL_LAMBDA = 1.6

# Aggregation columns used to compute the delay distributions. The order of
# columns reflects the order of priority.
PM_AGGREGATION_COLUMNS = ["stop_id", "hour", "rain", "temperature"]

# Name of the value column to compute the distributions on
PM_VALUE_COLUMN = "delay"

# Name of the Parquet file that will be created with the distributions
PM_TABLE_NAME = "distributions"

# Value above which delays will be capped to reduce effect of outliers (we consider
# than delays higher than 30 minutes are not relevant, since it signifies major problem
# on the transportation network)
PM_DELAY_MAX_CAPPING = 30 * 60 # 30 minutes
