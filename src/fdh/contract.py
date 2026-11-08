"""Expected structure of the published source files.

MHCLG daily incident datasets and the Office for National Statistics fire and
rescue authority layers. A file that departs from these values stops the run.

This module imports only `re`, so configuration loading and the tests stay free
of the HTTP and geospatial dependencies the other modules carry.
"""

from __future__ import annotations

import re

# Column order as published, nine fields.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "FRS_TERRITORY",
    "E_CODE_TERRITORY",
    "FINANCIAL_YEAR",
    "CALENDAR_YEAR",
    "MONTH",
    "MONTH_CODE",
    "DAY",
    "DAY_OF_WEEK",
    "FRIS_INCIDENT_TYPE",
)

# Sheet holding one financial year of incidents, named like 202324.
DATA_SHEET_PATTERN = re.compile(r"^\d{6}$")

# Sheet carrying the publication date and data cut date.
CONFIG_SHEET_NAMES: tuple[str, ...] = ("config", "Config", "CONFIG")

# Labels as they appear in the file. The published guidance uses different
# category names for the same eleven types.
KNOWN_INCIDENT_LABELS: frozenset[str] = frozenset(
    {
        "Dwellings",
        "Other Buildings",
        "Road Vehicles",
        "Other Outdoors",
        "Chimney Fires",
        "Secondary Fires",
        "Due to apparatus",
        "Good intent",
        "Malicious",
        "Non-fire incidents",
        "Non-fire false alarms",
    }
)

# Abbreviated to three letters in the published file.
KNOWN_DAY_OF_WEEK: frozenset[str] = frozenset(
    {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
)

# ONS fire and rescue authority layers. Property names carry the edition year,
# so the year is matched rather than fixed: FRA22CD and FRA22NM in the
# December 2022 boundaries, FRA25CD and FRA25NM in the 2025 classification.
FRA_CODE_FIELD = re.compile(r"^FRA\d{2}CD$")
FRA_NAME_FIELD = re.compile(r"^FRA\d{2}NM$")

# Rural Urban Classification 2021 of fire and rescue areas. Field names are
# carried verbatim from the layer metadata, trailing underscores included.
CLASSIFICATION_KEY = "FRA25CD"
CLASSIFICATION_NAME = "FRA25NM"
SETTLEMENT_CLASS = "RUC21_settlement_class"
RURAL_SHARE = "Proportion_of_population_in_rural_OAs____"
CLASSIFICATION_FIELDS: tuple[str, ...] = (
    "Rural_Urban_flag",
    SETTLEMENT_CLASS,
    "RUC21_relative_access",
    RURAL_SHARE,
    "Proportion_of_population_in_OAs_further_from_a_major_town_or_city____",
)

# The four published settlement classes.
SETTLEMENT_CLASSES: frozenset[str] = frozenset(
    {"Majority rural", "Intermediate rural", "Intermediate urban", "Urban"}
)