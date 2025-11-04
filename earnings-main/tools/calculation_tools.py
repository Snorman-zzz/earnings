"""
Calculation Tools for Financial Analysis.
Critical functions for unit normalization and surprise calculations.
"""

from langchain_core.tools import tool
from typing import Dict, Union
import logging

from config.constants import UNIT_MULTIPLIERS

logger = logging.getLogger(__name__)


@tool
def normalize_to_millions(value: float, unit: str) -> float:
    """
    Normalize a financial value to millions for consistent comparison.

    Args:
        value: The financial value
        unit: Unit indicator ("M" for millions, "B" for billions, "K" for thousands)

    Returns:
        Value normalized to millions
    """
    multiplier = UNIT_MULTIPLIERS.get(unit.upper(), 1)
    normalized = value * multiplier

    logger.debug(f"Normalized {value}{unit} to {normalized}M")
    return normalized


@tool
def calculate_surprise_percentage(
    reported: float,
    expected: float,
    reported_unit: str = "M",
    expected_unit: str = "M",
) -> float:
    """
    Calculate earnings surprise percentage with unit normalization.

    CRITICAL: This ensures accurate surprise calculations when reported
    and expected values have different units (e.g., B vs M).

    Args:
        reported: Reported financial value
        expected: Expected/estimated value
        reported_unit: Unit of reported value ("M" or "B")
        expected_unit: Unit of expected value ("M" or "B")

    Returns:
        Surprise percentage
    """
    # Normalize both to millions
    reported_m = normalize_to_millions.invoke(
        {"value": reported, "unit": reported_unit}
    )
    expected_m = normalize_to_millions.invoke(
        {"value": expected, "unit": expected_unit}
    )

    if expected_m == 0:
        logger.warning(f"Expected value is 0, cannot calculate surprise percentage")
        return 0.0

    surprise = ((reported_m - expected_m) / abs(expected_m)) * 100

    logger.info(
        f"Surprise: Reported={reported}{reported_unit}, "
        f"Expected={expected}{expected_unit}, Surprise={surprise:.2f}%"
    )

    return round(surprise, 2)


@tool
def calculate_yoy_growth(current: float, prior: float, unit: str = "M") -> float:
    """
    Calculate year-over-year growth percentage.

    Args:
        current: Current period value
        prior: Prior year same period value
        unit: Unit for both values (should be same)

    Returns:
        YoY growth percentage
    """
    if prior == 0:
        logger.warning("Prior value is 0, cannot calculate YoY growth")
        return 0.0

    growth = ((current - prior) / abs(prior)) * 100

    logger.info(
        f"YoY Growth: Current={current}{unit}, Prior={prior}{unit}, Growth={growth:.2f}%"
    )

    return round(growth, 2)


@tool
def format_financial_value(
    value: float, unit: str = "M", include_symbol: bool = False
) -> str:
    """
    Format a financial value for display.

    Args:
        value: The financial value
        unit: Unit indicator ("M" or "B")
        include_symbol: Whether to include $ symbol

    Returns:
        Formatted string (e.g., "$1,234.5M" or "1,234.5M")
    """
    symbol = "$" if include_symbol else ""

    if abs(value) >= 1000 and unit == "M":
        # Convert to billions if M value is >= 1000
        value_b = value / 1000
        formatted = f"{symbol}{value_b:,.1f}B"
    else:
        formatted = f"{symbol}{value:,.1f}{unit}"

    return formatted


def calculate_metrics_dict(
    reported: Dict[str, Union[float, str]],
    expected: Dict[str, Union[float, str]],
    prior_year: Dict[str, Union[float, str]] = None,
) -> Dict[str, any]:
    """
    Calculate comprehensive financial metrics comparing reported vs expected values.

    Args:
        reported: Dict with 'value' and 'unit' keys for reported figures
        expected: Dict with 'value' and 'unit' keys for expected figures
        prior_year: Optional dict for YoY calculations

    Returns:
        Dictionary with calculated metrics including surprise % and YoY growth
    """
    metrics = {
        "reported_value": reported["value"],
        "reported_unit": reported["unit"],
        "expected_value": expected["value"],
        "expected_unit": expected["unit"],
    }

    # Calculate surprise
    metrics["surprise_pct"] = calculate_surprise_percentage.invoke(
        {
            "reported": reported["value"],
            "expected": expected["value"],
            "reported_unit": reported["unit"],
            "expected_unit": expected["unit"],
        }
    )

    # Calculate YoY if prior year data available
    if prior_year:
        metrics["prior_year_value"] = prior_year["value"]
        metrics["yoy_growth_pct"] = calculate_yoy_growth.invoke(
            {
                "current": reported["value"],
                "prior": prior_year["value"],
                "unit": reported["unit"],
            }
        )

    # Format for display
    metrics["reported_formatted"] = format_financial_value.invoke(
        {
            "value": reported["value"],
            "unit": reported["unit"],
            "include_symbol": True,
        }
    )

    metrics["expected_formatted"] = format_financial_value.invoke(
        {
            "value": expected["value"],
            "unit": expected["unit"],
            "include_symbol": True,
        }
    )

    return metrics
