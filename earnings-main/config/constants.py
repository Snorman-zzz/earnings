"""Constants for financial analysis"""

# Financial Metrics to Extract
FINANCIAL_METRICS = [
    "EPS (Earnings Per Share)",
    "Revenue",
    "Operating Income",
    "Net Income",
    "Gross Margin",
    "Operating Margin",
    "Free Cash Flow"
]

# Unit Mappings
UNIT_MULTIPLIERS = {
    "M": 1,           # Millions (base unit)
    "B": 1000,        # Billions
    "K": 0.001,       # Thousands
    "": 1             # No unit (assume millions)
}

# GAAP Classifications
GAAP_TYPES = ["GAAP", "Non-GAAP", "Adjusted"]

# Document Types
DOC_TYPES = {
    "press_release": "Press Release",
    "presentation": "Earnings Presentation"
}

# Allowed file types
ALLOWED_FILE_TYPES = [".pdf"]
