# File paths
RAW_DATA_PATH = "/home/nawaf/code/TwiQcV/Digital_Shield/Digital_Shield_data/raw/Digital_Shield_dataset.csv"
CLEANED_DATA_PATH = "/home/nawaf/code/TwiQcV/Digital_Shield/Digital_Shield_data/cleaned/Digital_Shield_cleaned.csv"

# Text columns to normalize
TEXT_COLUMNS = [
    "country",
    "attack type",
    "target industry",
    "attack source",
    "security vulnerability type",
    "defense mechanism used"
]

# Spelling mappings
SPELLING_MAPS = {
    "attack type": {
        "phishing": "phishing",
        "phishng": "phishing",
        "ddos": "ddos",
        "d dos": "ddos",
        "sql injection": "sql injection",
        "sqlinjection": "sql injection",
        "ransomware": "ransomware",
        "ransomwre": "ransomware",
        "man in the middle": "man-in-the-middle",
        "maninthemiddle": "man-in-the-middle",
        "malware": "malware",
        "m alware": "malware"
    },
    "country": {
        "china": "china",
        "chian": "china",
        "united kingdom": "uk",
        "uk": "uk",
        "england": "uk"
    }
}

# Data type conversions
DTYPE_MAP = {
    "year": "Int64",
    "financial loss (in million $)": "float",
    "number of affected users": "float",
    "incident resolution time (in hours)": "float",
    "data breach in gb": "float"
}

# Numeric and categorical columns
NUMERIC_COLUMNS = list(DTYPE_MAP.keys())
CATEGORICAL_COLUMNS = TEXT_COLUMNS.copy()

# Columns to drop
DROP_COLUMNS = ["attack source"]

# Imputation strategies
NUMERIC_IMPUTATION_STRATEGY = "median"
CATEGORICAL_IMPUTATION_STRATEGY = "most_frequent"
