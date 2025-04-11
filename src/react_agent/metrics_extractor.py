import re
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Set up basic logging configuration if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class MetricsExtractor:
    """Financial metrics extraction with focus on number accuracy and consistency."""

    def __init__(self):
        """Initialize the metrics extractor."""
        self.extraction_methods = {
            "eps": self._extract_eps,
            "revenue": self._extract_revenue,
            "net_income": self._extract_net_income,
            "gross_margin": self._extract_margin,
            "operating_income": self._extract_operating_income,
            "revenue_growth": self._extract_growth,
            "guidance": self._extract_guidance
        }
        # Store extracted PDF data for reference by extraction methods
        self.pdf_extracted_data = {}

    def extract_all_metrics(self, text: str, tables: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Extract all financial metrics using multiple methods.

        Args:
            text: Full text content from PDF
            tables: List of extracted tables

        Returns:
            Dictionary of metrics with their values
        """
        # Store text and tables for use in extraction methods
        self.pdf_extracted_data = {
            "text": text,
            "tables": tables,
            "metrics": {}
        }

        # Process text to fix common PDF extraction issues
        processed_text = self._preprocess_text(text)

        # Step 1: First pass with regex patterns (primary method)
        metrics = self._extract_all_metrics_regex(processed_text)
        self.pdf_extracted_data["metrics"] = metrics

        # Step 2: Extract metrics from tables (parallel method)
        table_metrics = self._extract_metrics_from_tables(tables)

        # Step 3: Merge results, prioritizing text-based extraction
        for metric, value in table_metrics.items():
            if metric not in metrics or not metrics[metric]:
                metrics[metric] = value

        # Step 4: For missing metrics, try specialized extraction methods
        for metric_name, extraction_method in self.extraction_methods.items():
            if metric_name not in metrics or not metrics[metric_name]:
                value = extraction_method(processed_text, tables)
                if value:
                    metrics[metric_name] = value

        # Step 5: Standardize values for consistency
        normalized_metrics = self._normalize_metrics(metrics)

        # Step 6: Perform validation checks
        validated_metrics = self._validate_metrics(normalized_metrics)

        logger.info(f"Extracted {len(validated_metrics)} metrics")
        return validated_metrics

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text to improve pattern matching success."""
        processed = text

        # Fix broken numbers (e.g., "$1 , 234" → "$1,234")
        processed = re.sub(r'(\d+)\s*,\s*(\d+)', r'\1,\2', processed)

        # Standardize whitespace around punctuation
        processed = re.sub(r'\s+([,.:])', r'\1', processed)

        # Fix spacing in units
        processed = re.sub(r'(\d+)\s+(million|billion|trillion|m|b|t)', r'\1 \2', processed)

        # Standardize EPS formatting
        processed = re.sub(r'earnings\s+per\s+share', 'EPS', processed, flags=re.IGNORECASE)

        return processed

    def _extract_all_metrics_regex(self, text: str) -> Dict[str, str]:
        """Extract metrics using comprehensive regex patterns with focus on numbers."""
        metrics = {}

        # Define regex patterns for each metric type
        patterns = {
            # EPS patterns
            "eps": [
                r"(?:Earnings Per Share|EPS)[:\s]*([\$]?[\d\.,]+)",
                r"(?:diluted earnings per share|diluted EPS)[:\s]*([\$]?[\d\.,]+)",
                r"(?:reported EPS)[:\s]*([\$]?[\d\.,]+)",
                r"(?:GAAP EPS)[:\s]*([\$]?[\d\.,]+)",
                r"EPS (?:was|of|at)[:\s]*([\$]?[\d\.,]+)",
                r"([\$]?[\d\.,]+)(?:\s*per diluted share|\s*per share)",
                r"earnings of \$([\d\.,]+) per (?:diluted )?share",
                r"EPS:?\s*([\$]?[\d\.,]+)",
                r"earnings per share of \$([\d\.,]+)",
                r"(?:earnings|income) of \$[\d\.,]+ per share, or \$([\d\.,]+) per diluted share",
                r"diluted earnings[^\.]*?\$([\d\.,]+) per share",
                r"diluted EPS of \$([\d\.,]+)",
                r"GAAP diluted (?:earnings|EPS)[^\.]*?\$([\d\.,]+) per share",
                r"(?:earnings|income) of \$([\d\.,]+) per share"
            ],

            # Adjusted EPS patterns
            "adj_eps": [
                r"(?:Adjusted Earnings Per Share|Adjusted EPS|Non-GAAP EPS)[:\s]*([\$]?[\d\.,]+)",
                r"(?:non-GAAP earnings per share|non-GAAP EPS)[:\s]*([\$]?[\d\.,]+)",
                r"(?:non GAAP EPS)[:\s]*([\$]?[\d\.,]+)",
                r"Adjusted EPS (?:was|of|at)[:\s]*([\$]?[\d\.,]+)",
                r"adjusted earnings per share of \$([\d\.,]+)",
                r"non-GAAP diluted EPS[^\.]*?\$([\d\.,]+)",
                r"Non-GAAP diluted earnings[^\.]*?\$([\d\.,]+) per share",
                r"Adjusted diluted EPS[^\.]*?\$([\d\.,]+)",
                r"non-GAAP earnings[^\.]*?\$([\d\.,]+) per diluted share"
            ],

            # Revenue patterns
            "revenue": [
                r"(?:Revenue|Net Revenue|Total Revenue)[:\s]*([\$]?[\d\.,]+\s*(?:million|billion|trillion|m|b|t|M|B|MM))",
                r"(?:Revenue|Net Revenue|Total Revenue)[:\s]*\$([\d\.,]+)",
                r"(?:Revenue|Net Revenue|Total Revenue)[:\s]*([\d\.,]+)",
                r"(?:Revenue|Net Revenue|Total Revenue) (?:was|of|at)[:\s]*([\$]?[\d\.,]+\s*(?:million|billion|trillion|m|b|t|M|B|MM)?)",
                r"(?:Revenue|Net Revenue|Total Revenue) reached[:\s]*([\$]?[\d\.,]+\s*(?:million|billion|trillion|m|b|t|M|B|MM)?)",
                r"(?:Revenue|Net Revenue|Total Revenue)[:\s]*\$([\d\.,]+\s*(?:million|billion|trillion|m|b|t|M|B|MM)?)",
                r"Revenue:?\s*([\$]?[\d\.,]+\s*(?:million|billion|trillion|m|b|t|M|B|MM)?)",
                r"revenue of \$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"quarterly revenue[^\.]*?\$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"sales of \$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"revenue was \$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"reported.{1,30}revenue of \$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"revenue.{1,15}increased.{1,15}to \$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))"
            ],

            # Net income patterns
            "net_income": [
                r"(?:Net Income|Net Earnings)[:\s]*([\$]?[\d\.,]+\s*(?:million|billion|trillion|m|b|t|M|B|MM))",
                r"(?:Net Income|Net Earnings)[:\s]*\$([\d\.,]+)",
                r"(?:Net Income|Net Earnings) (?:was|of|at)[:\s]*([\$]?[\d\.,]+\s*(?:million|billion|trillion|m|b|t|M|B|MM)?)",
                r"(?:Net Income|Net Earnings) reached[:\s]*([\$]?[\d\.,]+\s*(?:million|billion|trillion|m|b|t|M|B|MM)?)",
                r"net income of \$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"Net Income:?\s*([\$]?[\d\.,]+\s*(?:million|billion|trillion|m|b|t|M|B|MM)?)",
                r"GAAP net income[^\.]*?\$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"reported.{1,30}net income of \$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"income of \$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"net earnings[^\.]*?\$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))"
            ],

            # Gross margin patterns
            "gross_margin": [
                r"(?:Gross Margin)[:\s]*([\d\.,]+\s*\%)",
                r"(?:Gross Margin)[:\s]*([\d\.,]+)",
                r"(?:Gross Margin) (?:was|of|at)[:\s]*([\d\.,]+\s*\%?)",
                r"(?:Gross Margin) increased to[:\s]*([\d\.,]+\s*\%?)",
                r"gross margin of ([\d\.,]+\%?)",
                r"gross margin was ([\d\.,]+\%?)",
                r"gross margin:?\s*([\d\.,]+\%?)",
                r"GAAP gross margin[^\.]*?([\d\.,]+\%)",
                r"gross margin.{1,15}increased.{1,15}to ([\d\.,]+\%)",
                r"gross profit margin of ([\d\.,]+\%)",
                r"gross margin percentage of ([\d\.,]+\%?)"
            ],

            # Operating income patterns
            "operating_income": [
                r"(?:Operating Income|Income from Operations|Operating Profit)[:\s]*([\$]?[\d\.,]+\s*(?:million|billion|trillion|m|b|t|M|B|MM))",
                r"(?:Operating Income|Income from Operations|Operating Profit)[:\s]*\$([\d\.,]+)",
                r"(?:Operating Income|Income from Operations|Operating Profit) (?:was|of|at)[:\s]*([\$]?[\d\.,]+\s*(?:million|billion|trillion|m|b|t|M|B|MM)?)",
                r"operating income of \$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"Operating Income:?\s*([\$]?[\d\.,]+\s*(?:million|billion|trillion|m|b|t|M|B|MM)?)",
                r"GAAP operating income[^\.]*?\$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"income from operations[^\.]*?\$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"operating profit[^\.]*?\$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"operating income was \$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"reported.{1,30}operating income of \$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))"
            ],

            # Year-over-year growth
            "revenue_growth": [
                r"(?:revenue growth|growth in revenue)[:\s]*([\d\.,]+\s*\%)",
                r"(?:increased|grew)(?:[^\n.]+)by[:\s]*([\d\.,]+\s*\%)",
                r"year[- ]over[- ]year growth of ([\d\.,]+\%)",
                r"revenue increased(?:[^\n.]+)([\d\.,]+\%)",
                r"revenue grew(?:[^\n.]+)([\d\.,]+\%)",
                r"sales grew(?:[^\n.]+)([\d\.,]+\%)",
                r"growth rate of ([\d\.,]+\%)",
                r"grew ([\d\.,]+\%) compared",
                r"up ([\d\.,]+\%) year[- ]over[- ]year",
                r"increased ([\d\.,]+\%) year[- ]over[- ]year",
                r"revenue.{1,30}increased.{1,30}([\d\.,]+\%) (?:from|compared)",
                r"revenue.{1,30}grew.{1,30}([\d\.,]+\%) (?:from|compared)"
            ],

            # Guidance patterns
            "guidance": [
                r"(?:Q\d guidance|next quarter guidance|outlook)(?:[^\n.]+)([\$]?[\d\.,]+\s*(?:million|billion|trillion|m|b|t|M|B|MM)?)",
                r"(?:expects|anticipates|forecasts|projecting)(?:[^\n.]+)revenue of \$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"(?:guidance|outlook)[:\s]*([\$]?[\d\.,]+\s*(?:million|billion|trillion|m|b|t|M|B|MM)?)",
                r"revenue guidance of \$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"guidance for (?:Q\d|next quarter|full year)(?:[^\n.]+)revenue of \$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"expects (?:Q\d|next quarter|full year) revenue of \$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"(?:forecasting|guiding|projecting)[^\.]*?revenue of \$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))",
                r"expects EPS of \$([\d\.,]+)",
                r"EPS guidance of \$([\d\.,]+)",
                r"anticipates revenue.{1,30}\$([\d\.,]+)(?:\s*(?:million|billion|trillion|m|b|t))"
            ]
        }

        # Process each metric and its patterns
        for metric, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Use the first match found
                    metrics[metric] = matches[0]
                    break  # Stop after first successful pattern

        return metrics

    def _extract_metrics_from_tables(self, tables: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract metrics from table data focused on numerical values."""
        metrics = {}

        # Sort tables by relevance to ensure we process most relevant tables first
        sorted_tables = sorted(tables, key=lambda t: t.get("financial_relevance", 0), reverse=True)

        # First pass: Look for tables with explicit metric labels
        for table in sorted_tables:
            data = table.get("data", [])
            columns = table.get("columns", [])

            if not data or not columns:
                continue

            # Check for tables with metric labels in rows
            for row in data:
                # Skip if first column missing (would contain labels)
                if columns[0] not in row:
                    continue

                label = str(row[columns[0]]).lower()

                # Look for rows with specific metric labels
                metric_patterns = {
                    "eps": ["eps", "earnings per share", "diluted earnings"],
                    "revenue": ["revenue", "net revenue", "total revenue", "sales"],
                    "net_income": ["net income", "net earnings", "net profit"],
                    "operating_income": ["operating income", "income from operations", "operating profit"],
                    "gross_margin": ["gross margin", "gross profit margin", "margin percentage"]
                }

                for metric, patterns in metric_patterns.items():
                    if any(p in label for p in patterns):
                        # Look for a value in another column (typically the first non-header column)
                        for col in columns[1:]:
                            if col in row and row[col]:
                                value = row[col]
                                if str(value).strip() != 'N/A':
                                    metrics[metric] = self._clean_metric_value(value)
                                    break

        # Second pass: Look for tables with metrics in column headers
        if len(metrics) < 4:  # If we haven't found most metrics yet
            for table in sorted_tables:
                data = table.get("data", [])
                columns = table.get("columns", [])

                if not data or not columns:
                    continue

                # Check for columns with metric names
                for col_idx, col in enumerate(columns):
                    col_lower = str(col).lower()

                    # Look for specific metrics in column names
                    metric_patterns = {
                        "eps": ["eps", "earnings per share", "per share"],
                        "revenue": ["revenue", "sales", "net revenue"],
                        "net_income": ["net income", "earnings", "net profit"],
                        "operating_income": ["operating income", "operating profit"],
                        "gross_margin": ["gross margin", "margin", "gross profit"]
                    }

                    for metric, patterns in metric_patterns.items():
                        if any(p in col_lower for p in patterns) and (metric not in metrics or not metrics[metric]):
                            # Get the value from the first row
                            if data and col in data[0]:
                                value = data[0][col]
                                if str(value).strip() != 'N/A':
                                    metrics[metric] = self._clean_metric_value(value)

        return metrics

    def _clean_metric_value(self, value) -> Optional[str]:
        """Clean and format a metric value with focus on number accuracy."""
        if value is None:
            return None

        # Convert to string
        value_str = str(value).strip()
        if value_str == "" or value_str.lower() == "n/a":
            return None

        # Clean up whitespace
        value_str = re.sub(r'\s+', ' ', value_str)

        # Add dollar sign for financial values if not present (but not for percentages)
        if re.search(r'^\d+(\.\d+)?$', value_str) and "margin" not in str(value).lower() and "%" not in value_str:
            # Make sure to handle only dollar amounts, not percentages
            if re.search(r'^\d+(\.\d+)?$', value_str) and float(value_str) < 100 and "margin" not in str(value).lower():
                # Likely EPS
                value_str = f"${value_str}"

        # Ensure percentages have % symbol
        if "margin" in str(value).lower() or "growth" in str(value).lower():
            if "%" not in value_str:
                try:
                    if float(value_str) <= 100:  # Sanity check for percentage
                        value_str = f"{value_str}%"
                except ValueError:
                    pass

        # Ensure consistent representation for billions/millions
        if re.search(r'(billion|b|bn)', value_str, re.IGNORECASE):
            # Extract the numeric part
            num_match = re.search(r'[\$]?([\d\.,]+)', value_str)
            if num_match:
                number = num_match.group(1).replace(',', '')
                try:
                    value_str = f"${float(number)}B"
                except ValueError:
                    pass
        elif re.search(r'(million|m|mn)', value_str, re.IGNORECASE):
            # Extract the numeric part
            num_match = re.search(r'[\$]?([\d\.,]+)', value_str)
            if num_match:
                number = num_match.group(1).replace(',', '')
                try:
                    value_str = f"${float(number)}M"
                except ValueError:
                    pass

        return value_str

    def _normalize_metrics(self, metrics: Dict[str, str]) -> Dict[str, str]:
        """Standardize metrics for consistent representation."""
        normalized = {}

        for metric, value in metrics.items():
            if value is None:
                normalized[metric] = None
                continue

            # Convert to string if not already
            value_str = str(value)

            # Handle different metric types
            if metric in ["eps", "adj_eps"]:
                # Normalize EPS values (should be in $ format)
                if not value_str.startswith("$"):
                    try:
                        number = float(value_str.replace("%", "").replace(",", ""))
                        value_str = f"${number:.2f}"
                    except ValueError:
                        pass
            elif metric in ["revenue", "net_income", "operating_income"]:
                # Normalize monetary values (should have $ and proper unit)
                if not value_str.startswith("$"):
                    value_str = f"${value_str}"

                # Ensure unit is present (M or B)
                if not any(unit in value_str for unit in ["M", "B", "million", "billion"]):
                    try:
                        number = float(value_str.replace("$", "").replace(",", ""))
                        if number >= 1000000000:
                            value_str = f"${number / 1000000000:.2f}B"
                        elif number >= 1000000:
                            value_str = f"${number / 1000000:.2f}M"
                    except ValueError:
                        pass
            elif metric in ["gross_margin", "revenue_growth"]:
                # Normalize percentage values
                if not "%" in value_str:
                    try:
                        number = float(value_str.replace(",", ""))
                        # If decimal (like 0.42), convert to percentage
                        if number < 1:
                            number *= 100
                        value_str = f"{number:.2f}%"
                    except ValueError:
                        pass

            normalized[metric] = value_str

        return normalized

    def _validate_metrics(self, metrics: Dict[str, str]) -> Dict[str, str]:
        """Validate metrics for reasonable ranges and consistency."""
        validated = {}

        for metric, value in metrics.items():
            if value is None:
                validated[metric] = "N/A"
                continue

            # Basic validation checks
            if metric in ["eps", "adj_eps"]:
                # EPS is typically between -10 and 10 for most companies
                try:
                    number = float(value.replace("$", "").replace(",", ""))
                    if abs(number) > 100:  # Extremely high, likely an error
                        logger.warning(f"Unusual EPS value: {value}, setting to N/A")
                        validated[metric] = "N/A"
                        continue
                except ValueError:
                    pass
            elif metric in ["revenue", "net_income", "operating_income"]:
                # Check for reasonable monetary values
                try:
                    # Extract just the numeric part
                    num_match = re.search(r'[\$]?([\d\.,]+)', value)
                    if num_match:
                        number = float(num_match.group(1).replace(',', ''))

                        # Check for unreasonable values based on unit
                        if "B" in value and number > 1000:  # Over a trillion dollars
                            logger.warning(f"Unusually high {metric}: {value}, setting to N/A")
                            validated[metric] = "N/A"
                            continue
                except ValueError:
                    pass
            elif metric in ["gross_margin", "revenue_growth"]:
                # Percentages should typically be between -100% and 500%
                try:
                    number = float(value.replace("%", "").replace(",", ""))
                    if number < -100 or number > 500:
                        logger.warning(f"Unusual percentage for {metric}: {value}, setting to N/A")
                        validated[metric] = "N/A"
                        continue
                except ValueError:
                    pass

            validated[metric] = value

        return validated

    def _extract_eps(self, text: str, tables: List[Dict[str, Any]]) -> Optional[str]:
        """Specialized function to extract EPS focusing on accurate numbers."""
        # First attempt: Find sentences with both "EPS" and a dollar sign followed by a number
        eps_sentences = []
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            sentence_lower = sentence.lower()
            if ("eps" in sentence_lower or "earnings per share" in sentence_lower) and "$" in sentence:
                eps_sentences.append(sentence)

        # Try to find dollar values in these sentences
        for sentence in eps_sentences:
            # Look specifically for dollar values that are likely EPS (typically under $100)
            eps_matches = re.findall(r'\$([\d\.]+)', sentence)
            if eps_matches:
                eps_values = [float(m) for m in eps_matches if float(m) < 100]
                if eps_values:
                    # Take the smallest reasonable value (most likely EPS)
                    # EPS is typically smaller than other metrics
                    eps_values = [v for v in eps_values if v > 0.01]  # Avoid tiny values
                    if eps_values:
                        return f"${min(eps_values)}"

        # Second attempt: Check tables for EPS
        for table in tables:
            data = table.get("data", [])
            columns = table.get("columns", [])

            if not data or not columns:
                continue

            # Look for EPS in column headers
            eps_col_idx = -1
            for i, col in enumerate(columns):
                col_lower = str(col).lower()
                if "eps" in col_lower or "earnings per share" in col_lower:
                    eps_col_idx = i
                    break

            if eps_col_idx != -1 and data:
                # Get value from the first row
                try:
                    row = data[0]
                    col_name = columns[eps_col_idx]
                    if col_name in row:
                        value = row[col_name]
                        if value and str(value).strip() != 'N/A':
                            return self._clean_metric_value(value)
                except (IndexError, KeyError):
                    pass

            # Look for EPS in row labels
            if columns and len(columns) > 0:
                first_col = columns[0]
                for row in data:
                    if first_col not in row:
                        continue

                    label = str(row[first_col]).lower()
                    if "eps" in label or "earnings per share" in label:
                        # Get the value from another column
                        for col in columns[1:]:
                            if col in row:
                                value = row[col]
                                if value and str(value).strip() != 'N/A':
                                    return self._clean_metric_value(value)

        return None

    def _extract_revenue(self, text: str, tables: List[Dict[str, Any]]) -> Optional[str]:
        """Specialized function to extract revenue with focus on numbers."""
        # First approach: Look for explicit revenue mentions with units
        revenue_patterns = [
            r'revenue[^\.]*?\$([\d\.,]+)\s*(?:billion|million|b|m)',
            r'total revenue[^\.]*?\$([\d\.,]+)\s*(?:billion|million|b|m)',
            r'revenue was \$([\d\.,]+)\s*(?:billion|million|b|m)',
            r'revenue of \$([\d\.,]+)\s*(?:billion|million|b|m)',
            r'reported revenue[^\.]*?\$([\d\.,]+)\s*(?:billion|million|b|m)'
        ]

        for pattern in revenue_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Extract the value and determine the unit
                value = matches[0]
                unit = "B" if "billion" in pattern or "b" in pattern else "M"

                # Format consistently
                try:
                    value_num = float(value.replace(',', ''))
                    return f"${value_num}{unit}"
                except ValueError:
                    return f"${value}{unit}"

        # Second approach: Look in tables for revenue
        for table in tables:
            data = table.get("data", [])
            columns = table.get("columns", [])

            if not data or not columns:
                continue

            # Look for revenue in column headers
            revenue_col_idx = -1
            for i, col in enumerate(columns):
                col_lower = str(col).lower()
                if "revenue" in col_lower or "sales" in col_lower:
                    revenue_col_idx = i
                    break

            if revenue_col_idx != -1 and data:
                # Get value from the first row
                try:
                    row = data[0]
                    col_name = columns[revenue_col_idx]
                    if col_name in row:
                        value = row[col_name]
                        if value and str(value).strip() != 'N/A':
                            # Look for unit indicator in the column name or nearby
                            col_str = str(col_name).lower()
                            unit = "M"  # Default to millions
                            if "billion" in col_str or "b" in col_str:
                                unit = "B"
                            elif "million" in col_str or "m" in col_str:
                                unit = "M"

                            # Apply unit to the value if it doesn't already have one
                            value_str = str(value)
                            if "billion" not in value_str and "million" not in value_str and "b" not in value_str and "m" not in value_str:
                                try:
                                    value_num = float(value_str.replace('$', '').replace(',', ''))
                                    return f"${value_num}{unit}"
                                except ValueError:
                                    pass

                            return self._clean_metric_value(value)
                except (IndexError, KeyError):
                    pass

            # Look for revenue in row labels
            if columns and len(columns) > 0:
                first_col = columns[0]
                for row in data:
                    if first_col not in row:
                        continue

                    label = str(row[first_col]).lower()
                    if "revenue" in label or "sales" in label:
                        # Get the value from another column
                        for col in columns[1:]:
                            if col in row:
                                value = row[col]
                                if value and str(value).strip() != 'N/A':
                                    return self._clean_metric_value(value)

        # Third approach: Look for largest dollar amount in revenue paragraphs
        revenue_paragraphs = []
        paragraphs = text.split('\n\n')
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) < 20:
                continue

            paragraph_lower = paragraph.lower()
            if "revenue" in paragraph_lower or "sales" in paragraph_lower:
                revenue_paragraphs.append(paragraph)

        if revenue_paragraphs:
            # Look for dollar amounts with units
            all_matches = []
            for paragraph in revenue_paragraphs:
                # Find all dollar amounts with units
                matches = re.findall(r'\$([\d\.,]+)\s*(?:billion|million|b|m)', paragraph, re.IGNORECASE)
                if matches:
                    for match in matches:
                        value = match.replace(',', '')
                        try:
                            value_num = float(value)
                            # Store the value and whether it's billions or millions
                            unit = "B" if "billion" in paragraph.lower() or " b" in paragraph.lower() else "M"
                            all_matches.append((value_num, unit))
                        except ValueError:
                            pass

            if all_matches:
                # Revenue is typically one of the largest values
                # Sort by value in billions
                sorted_matches = sorted(all_matches, key=lambda x: x[0] * (1000 if x[1] == "B" else 1), reverse=True)
                highest_value, unit = sorted_matches[0]
                return f"${highest_value}{unit}"

        return None

    def _extract_net_income(self, text: str, tables: List[Dict[str, Any]]) -> Optional[str]:
        """Specialized function to extract net income with focus on numbers."""
        # First approach: Look for explicit net income mentions with units
        income_patterns = [
            r'net income[^\.]*?\$([\d\.,]+)\s*(?:billion|million|b|m)',
            r'net earnings[^\.]*?\$([\d\.,]+)\s*(?:billion|million|b|m)',
            r'net profit[^\.]*?\$([\d\.,]+)\s*(?:billion|million|b|m)'
        ]

        for pattern in income_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Extract the value and determine the unit
                value = matches[0]
                unit = "B" if "billion" in pattern or "b" in pattern else "M"

                # Format consistently
                try:
                    value_num = float(value.replace(',', ''))
                    return f"${value_num}{unit}"
                except ValueError:
                    return f"${value}{unit}"

        # Second approach: Check tables
        for table in tables:
            data = table.get("data", [])
            columns = table.get("columns", [])

            if not data or not columns:
                continue

            # Similar approach as revenue extraction but for net income
            income_terms = ["net income", "net earnings", "net profit"]

            # Look in column headers
            income_col_idx = -1
            for i, col in enumerate(columns):
                col_lower = str(col).lower()
                if any(term in col_lower for term in income_terms):
                    income_col_idx = i
                    break

            if income_col_idx != -1 and data:
                # Get value from the first row
                try:
                    row = data[0]
                    col_name = columns[income_col_idx]
                    if col_name in row:
                        value = row[col_name]
                        if value and str(value).strip() != 'N/A':
                            return self._clean_metric_value(value)
                except (IndexError, KeyError):
                    pass

            # Look in row labels
            if columns and len(columns) > 0:
                first_col = columns[0]
                for row in data:
                    if first_col not in row:
                        continue

                    label = str(row[first_col]).lower()
                    if any(term in label for term in income_terms):
                        # Get value from another column
                        for col in columns[1:]:
                            if col in row:
                                value = row[col]
                                if value and str(value).strip() != 'N/A':
                                    return self._clean_metric_value(value)

        # Third approach: Look for dollar amounts in net income paragraphs
        income_paragraphs = []
        paragraphs = text.split('\n\n')
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) < 20:
                continue

            paragraph_lower = paragraph.lower()
            if any(term in paragraph_lower for term in income_terms):
                income_paragraphs.append(paragraph)

        if income_paragraphs:
            # Look for dollar amounts with units
            all_matches = []
            for paragraph in income_paragraphs:
                # Find all dollar amounts with units
                matches = re.findall(r'\$([\d\.,]+)\s*(?:billion|million|b|m)', paragraph, re.IGNORECASE)
                if matches:
                    for match in matches:
                        value = match.replace(',', '')
                        try:
                            value_num = float(value)
                            # Store the value and whether it's billions or millions
                            unit = "B" if "billion" in paragraph.lower() or " b" in paragraph.lower() else "M"
                            all_matches.append((value_num, unit))
                        except ValueError:
                            pass

            if all_matches:
                # Net income is typically smaller than revenue but still a significant value
                # Sort by value in billions
                sorted_matches = sorted(all_matches, key=lambda x: x[0] * (1000 if x[1] == "B" else 1), reverse=True)
                # Take the first match if we have only one, otherwise take second largest (assuming revenue is largest)
                idx = 0 if len(sorted_matches) == 1 else 1
                if idx < len(sorted_matches):
                    value, unit = sorted_matches[idx]
                    return f"${value}{unit}"

        return None

    def _extract_margin(self, text: str, tables: List[Dict[str, Any]]) -> Optional[str]:
        """Specialized function to extract gross margin with focus on percentages."""
        # First approach: Look for percentage values in margin-related text
        margin_patterns = [
            r'gross margin[^\.]*?([\d\.]+)\s*\%',
            r'gross margin[^\.]*?([\d\.]+)\s*percent',
            r'gross margin of ([\d\.]+)\s*\%',
            r'gross margin was ([\d\.]+)\s*\%',
            r'margin of ([\d\.]+)\s*\%'
        ]

        for pattern in margin_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Extract the percentage value
                value = matches[0]

                # Format consistently
                try:
                    value_num = float(value.replace(',', ''))
                    return f"{value_num}%"
                except ValueError:
                    return f"{value}%"

        # Second approach: Check tables
        for table in tables:
            data = table.get("data", [])
            columns = table.get("columns", [])

            if not data or not columns:
                continue

            # Look for margin in column headers
            margin_terms = ["gross margin", "margin percentage", "gross profit margin"]

            margin_col_idx = -1
            for i, col in enumerate(columns):
                col_lower = str(col).lower()
                if any(term in col_lower for term in margin_terms):
                    margin_col_idx = i
                    break

            if margin_col_idx != -1 and data:
                # Get value from the first row
                try:
                    row = data[0]
                    col_name = columns[margin_col_idx]
                    if col_name in row:
                        value = row[col_name]
                        if value and str(value).strip() != 'N/A':
                            # Ensure it has a % symbol
                            value_str = str(value)
                            if "%" not in value_str:
                                try:
                                    value_num = float(value_str.replace(',', ''))
                                    # If it's a decimal like 0.42, convert to percentage
                                    if value_num < 1:
                                        value_num *= 100
                                    return f"{value_num}%"
                                except ValueError:
                                    pass
                            return value_str
                except (IndexError, KeyError):
                    pass

            # Look in row labels
            if columns and len(columns) > 0:
                first_col = columns[0]
                for row in data:
                    if first_col not in row:
                        continue

                    label = str(row[first_col]).lower()
                    if any(term in label for term in margin_terms):
                        # Get value from another column
                        for col in columns[1:]:
                            if col in row:
                                value = row[col]
                                if value and str(value).strip() != 'N/A':
                                    # Ensure it has a % symbol
                                    value_str = str(value)
                                    if "%" not in value_str:
                                        try:
                                            value_num = float(value_str.replace(',', ''))
                                            # If it's a decimal like 0.42, convert to percentage
                                            if value_num < 1:
                                                value_num *= 100
                                            return f"{value_num}%"
                                        except ValueError:
                                            pass
                                    return value_str

        # Third approach: Look for percentages in margin paragraphs
        margin_paragraphs = []
        paragraphs = text.split('\n\n')
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) < 20:
                continue

            paragraph_lower = paragraph.lower()
            if "gross margin" in paragraph_lower or "profit margin" in paragraph_lower:
                margin_paragraphs.append(paragraph)

        if margin_paragraphs:
            # Look for percentage values
            for paragraph in margin_paragraphs:
                # Find all percentages
                matches = re.findall(r'([\d\.]+)\s*\%', paragraph)
                if matches:
                    # Take the most likely value
                    for match in matches:
                        try:
                            value = float(match)
                            # Typical gross margin range check
                            if 1 <= value <= 100:
                                return f"{value}%"
                        except ValueError:
                            pass

        return None

    def _extract_operating_income(self, text: str, tables: List[Dict[str, Any]]) -> Optional[str]:
        """Specialized function to extract operating income with focus on numbers."""
        # First approach: Look for explicit operating income mentions with units
        income_patterns = [
            r'operating income[^\.]*?\$([\d\.,]+)\s*(?:billion|million|b|m)',
            r'income from operations[^\.]*?\$([\d\.,]+)\s*(?:billion|million|b|m)',
            r'operating profit[^\.]*?\$([\d\.,]+)\s*(?:billion|million|b|m)'
        ]

        for pattern in income_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Extract the value and determine the unit
                value = matches[0]
                unit = "B" if "billion" in pattern or "b" in pattern else "M"

                # Format consistently
                try:
                    value_num = float(value.replace(',', ''))
                    return f"${value_num}{unit}"
                except ValueError:
                    return f"${value}{unit}"

        # Second approach: Similar to net income but for operating income
        # Use table extraction similar to previous metrics
        income_terms = ["operating income", "income from operations", "operating profit"]

        for table in tables:
            data = table.get("data", [])
            columns = table.get("columns", [])

            if not data or not columns:
                continue

            # Look in column headers
            income_col_idx = -1
            for i, col in enumerate(columns):
                col_lower = str(col).lower()
                if any(term in col_lower for term in income_terms):
                    income_col_idx = i
                    break

            if income_col_idx != -1 and data:
                # Get value from the first row
                try:
                    row = data[0]
                    col_name = columns[income_col_idx]
                    if col_name in row:
                        value = row[col_name]
                        if value and str(value).strip() != 'N/A':
                            return self._clean_metric_value(value)
                except (IndexError, KeyError):
                    pass

            # Look in row labels
            if columns and len(columns) > 0:
                first_col = columns[0]
                for row in data:
                    if first_col not in row:
                        continue

                    label = str(row[first_col]).lower()
                    if any(term in label for term in income_terms):
                        # Get value from another column
                        for col in columns[1:]:
                            if col in row:
                                value = row[col]
                                if value and str(value).strip() != 'N/A':
                                    return self._clean_metric_value(value)

        # Third approach: Look in operating income paragraphs
        income_paragraphs = []
        paragraphs = text.split('\n\n')
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) < 20:
                continue

            paragraph_lower = paragraph.lower()
            if any(term in paragraph_lower for term in income_terms):
                income_paragraphs.append(paragraph)

        if income_paragraphs:
            # Similar to net income extraction
            all_matches = []
            for paragraph in income_paragraphs:
                matches = re.findall(r'\$([\d\.,]+)\s*(?:billion|million|b|m)', paragraph, re.IGNORECASE)
                if matches:
                    for match in matches:
                        value = match.replace(',', '')
                        try:
                            value_num = float(value)
                            unit = "B" if "billion" in paragraph.lower() or " b" in paragraph.lower() else "M"
                            all_matches.append((value_num, unit))
                        except ValueError:
                            pass

            if all_matches:
                # Sort and select most likely value
                sorted_matches = sorted(all_matches, key=lambda x: x[0] * (1000 if x[1] == "B" else 1), reverse=True)
                if sorted_matches:
                    value, unit = sorted_matches[0]
                    return f"${value}{unit}"

        return None

    def _extract_growth(self, text: str, tables: List[Dict[str, Any]]) -> Optional[str]:
        """Specialized function to extract revenue growth rates with focus on percentages."""
        # First approach: Look for explicit growth mentions with percentages
        growth_patterns = [
            r'revenue growth[^\.]*?([\d\.]+)\s*\%',
            r'revenue increased[^\.]*?([\d\.]+)\s*\%',
            r'grew by[^\.]*?([\d\.]+)\s*\%',
            r'growth of[^\.]*?([\d\.]+)\s*\%',
            r'year-over-year growth[^\.]*?([\d\.]+)\s*\%',
            r'year over year growth[^\.]*?([\d\.]+)\s*\%',
            r'YoY growth[^\.]*?([\d\.]+)\s*\%'
        ]

        for pattern in growth_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Extract the percentage value
                value = matches[0]

                # Format consistently
                try:
                    value_num = float(value.replace(',', ''))
                    return f"{value_num}%"
                except ValueError:
                    return f"{value}%"

        # Second approach: Check tables for growth rates
        for table in tables:
            data = table.get("data", [])
            columns = table.get("columns", [])

            if not data or not columns:
                continue

            # Look for growth in column headers
            growth_terms = ["growth", "increase", "yoy", "y/y", "year over year"]

            growth_col_idx = -1
            for i, col in enumerate(columns):
                col_lower = str(col).lower()
                if any(term in col_lower for term in growth_terms):
                    growth_col_idx = i
                    break

            if growth_col_idx != -1 and data:
                # Get value from the first row
                try:
                    row = data[0]
                    col_name = columns[growth_col_idx]
                    if col_name in row:
                        value = row[col_name]
                        if value and str(value).strip() != 'N/A':
                            # Ensure it has a % symbol
                            value_str = str(value)
                            if "%" not in value_str:
                                try:
                                    value_num = float(value_str.replace(',', ''))
                                    # If it's a decimal like 0.42, convert to percentage
                                    if value_num < 1:
                                        value_num *= 100
                                    return f"{value_num}%"
                                except ValueError:
                                    pass
                            return value_str
                except (IndexError, KeyError):
                    pass

        # Third approach: Calculate growth from financial statements if possible
        # Look for current and previous values to calculate growth
        current_revenue = None
        previous_revenue = None

        # Try to find these in metrics or tables
        if "revenue" in self.pdf_extracted_data.get("metrics", {}):
            current_revenue_str = self.pdf_extracted_data["metrics"]["revenue"]
            # Extract numeric value
            match = re.search(r'[\$]?([\d\.,]+)', current_revenue_str)
            if match:
                try:
                    current_revenue = float(match.group(1).replace(',', ''))
                    # Adjust for units
                    if "B" in current_revenue_str:
                        current_revenue *= 1000
                    # Now look for previous year revenue in text
                    prev_patterns = [
                        r'compared to [^\.]*?\$([\d\.,]+)\s*(?:billion|million|B|M)',
                        r'versus [^\.]*?\$([\d\.,]+)\s*(?:billion|million|B|M) in the prior year',
                        r'up from [^\.]*?\$([\d\.,]+)\s*(?:billion|million|B|M)'
                    ]

                    for pattern in prev_patterns:
                        matches = re.findall(pattern, text, re.IGNORECASE)
                        if matches:
                            try:
                                prev_val = float(matches[0].replace(',', ''))
                                if "billion" in pattern.lower() or "B" in pattern:
                                    prev_val *= 1000  # Convert to same unit as current

                                # Calculate growth
                                growth_pct = ((current_revenue - prev_val) / prev_val) * 100
                                return f"{growth_pct:.2f}%"
                            except ValueError:
                                pass
                except ValueError:
                    pass

        return None

    def _extract_guidance(self, text: str, tables: List[Dict[str, Any]]) -> Optional[str]:
        """Specialized function to extract forward guidance with focus on accurate numbers."""
        # First approach: Look for explicit guidance mentions in text
        guidance_patterns = [
            r'guidance for [^\.]*?revenue of \$([\d\.,]+)\s*(?:billion|million|B|M)',
            r'expect[s]? [^\.]*?revenue [^\.]*?\$([\d\.,]+)\s*(?:billion|million|B|M)',
            r'forecast[s]? [^\.]*?revenue [^\.]*?\$([\d\.,]+)\s*(?:billion|million|B|M)',
            r'guidance [^\.]*?revenue [^\.]*?\$([\d\.,]+)\s*(?:billion|million|B|M)',
            r'outlook [^\.]*?revenue [^\.]*?\$([\d\.,]+)\s*(?:billion|million|B|M)'
        ]

        for pattern in guidance_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Extract the value and determine the unit based on pattern
                value = matches[0].replace(',', '')
                unit = "B" if "billion" in pattern.lower() or "b" in pattern else "M"

                # Format consistently
                try:
                    value_num = float(value)
                    return f"Revenue of ${value_num}{unit}"
                except ValueError:
                    return f"Revenue of ${value}{unit}"

        # Second approach: Look for guidance paragraphs
        guidance_paragraphs = []
        paragraphs = text.split('\n\n')
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) < 20:
                continue

            paragraph_lower = paragraph.lower()
            if any(term in paragraph_lower for term in
                   ['guidance', 'outlook', 'expect', 'forecast', 'next quarter', 'next fiscal']):
                guidance_paragraphs.append(paragraph)

        if guidance_paragraphs:
            # Extract the most relevant guidance paragraph
            best_paragraph = None
            best_score = 0

            for paragraph in guidance_paragraphs:
                score = 0
                p_lower = paragraph.lower()

                # Score based on revenue mentions
                if "revenue" in p_lower:
                    score += 3

                # Score based on EPS mentions
                if "eps" in p_lower or "earnings per share" in p_lower:
                    score += 3

                # Score based on dollar amounts
                currency_matches = len(re.findall(r'\$([\d\.,]+)', paragraph))
                score += currency_matches * 2

                # Score based on percentage values
                pct_matches = len(re.findall(r'([\d\.]+)\s*\%', paragraph))
                score += pct_matches

                if score > best_score:
                    best_score = score
                    best_paragraph = paragraph

            if best_paragraph:
                # Extract key financial values
                guidance_info = []

                # Extract revenue guidance
                revenue_matches = re.findall(r'revenue [^\.]*?\$([\d\.,]+)\s*(?:billion|million|B|M)', best_paragraph,
                                             re.IGNORECASE)
                if revenue_matches:
                    val = revenue_matches[0].replace(',', '')
                    unit = "B" if "billion" in best_paragraph.lower() or " b" in best_paragraph.lower() else "M"
                    guidance_info.append(f"Revenue: ${val}{unit}")

                # Extract EPS guidance
                eps_matches = re.findall(r'EPS [^\.]*?\$([\d\.,]+)', best_paragraph, re.IGNORECASE)
                if eps_matches:
                    guidance_info.append(f"EPS: ${eps_matches[0]}")

                # Extract growth guidance
                growth_matches = re.findall(r'growth [^\.]*?([\d\.]+)\s*\%', best_paragraph, re.IGNORECASE)
                if growth_matches:
                    guidance_info.append(f"Growth: {growth_matches[0]}%")

                if guidance_info:
                    return ", ".join(guidance_info)
                else:
                    # Return a cleaned up version of the paragraph
                    cleaned = re.sub(r'\s+', ' ', best_paragraph)
                    return cleaned[:200] + "..." if len(cleaned) > 200 else cleaned

        # Third approach: Try tables with guidance information
        for table in tables:
            data = table.get("data", [])
            columns = table.get("columns", [])

            if not data or not columns:
                continue

            # Check if this appears to be a guidance table
            col_str = ' '.join(str(col).lower() for col in columns)
            if any(term in col_str for term in ['guidance', 'outlook', 'forecast', 'next quarter', 'fy24', 'fy25']):
                guidance_rows = []

                for row in data:
                    # Check if this row contains guidance information
                    row_str = str(row).lower()
                    if any(term in row_str for term in ['revenue', 'sales', 'eps', 'earnings', 'income', 'margin']):
                        # Extract metric and value
                        metric = None
                        value = None

                        # Try to identify metric
                        for col in columns:
                            if col in row and any(
                                    term in str(col).lower() for term in ['metric', 'item', 'description']):
                                metric = row[col]
                                break

                        # If we couldn't find a metric column, use first column
                        if not metric and len(columns) > 0:
                            metric = row.get(columns[0])

                        # Look for value in guidance columns
                        for col in columns:
                            col_lower = str(col).lower()
                            if any(term in col_lower for term in ['guidance', 'outlook', 'forecast', 'fy24', 'fy25']):
                                if col in row:
                                    value = row[col]
                                    break

                        if metric and value:
                            guidance_rows.append(f"{metric}: {value}")

                if guidance_rows:
                    return ", ".join(guidance_rows)

        return None