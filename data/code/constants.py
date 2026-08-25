from typing import Mapping

NBR_INTERVALS_PER_DAY = 96 # 15 min intervals per day

LAG_FEATURES: Mapping[str, int] = {
    "lag_1d": NBR_INTERVALS_PER_DAY,
    "lag_7d": NBR_INTERVALS_PER_DAY * 7,
    "lag_30d": NBR_INTERVALS_PER_DAY * 30,
    "lag_365d": NBR_INTERVALS_PER_DAY * 365
}

ROLLING_MEAN_FEATURES: Mapping[str, int] = {
    "rolling_mean_7d": NBR_INTERVALS_PER_DAY * 7,
    "rolling_mean_30d": NBR_INTERVALS_PER_DAY * 30,
}