"""Expected structure of the published MHCLG daily incident files.

Source: Ministry of Housing, Communities and Local Government, Fire statistics
incident level datasets, Open Government Licence v3.0.

Observed in the 2023-24 to present file, sha256
a034f01def8aeb6b1cca4ed34633861c298443486fdb576b438712121c1992ec, on
3 October 2026. The 2010-11 to 2016-17 and 2017-18 to 2022-23 files have not
been examined, so these values may be specific to the recent era.
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

MONTH_CODE_RANGE = range(1, 13)