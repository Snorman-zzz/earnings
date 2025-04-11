import io
import os
import re
import logging
import uuid
import PyPDF2
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Tuple
import tabula
import json
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logger = logging.getLogger(__name__)


class PDFDataExtractor:
    """PDF data extraction with focus on numerical accuracy and consistency."""

    def __init__(self):
        """Initialize the extractor."""
        self.extracted_data = {
            "tables": [],
            "metrics": {},
            "text": "",
            "structured_data": {}
        }

    def process_pdf_files(self, pdf_contents: List[bytes]) -> Dict[str, Any]:
        """Process multiple PDF files and extract all data."""
        all_text = ""
        all_tables = []
        all_metrics = {}

        for i, content in enumerate(pdf_contents):
            try:
                logger.info(f"Processing PDF #{i + 1} ({len(content)} bytes)")

                # 1. Extract text with preprocessing for better number detection
                text = self._extract_text_with_preprocessing(content)
                all_text += text + "\n\n"

                # 2. Extract tables focusing on number accuracy
                tables = self._extract_tables_with_number_focus(content, text)
                all_tables.extend(tables)

                # 3. Extract key metrics directly
                metrics = self._extract_key_financial_metrics(text)

                # Don't overwrite existing metrics if new ones are empty
                for k, v in metrics.items():
                    if k not in all_metrics or (k in all_metrics and all_metrics[k] == "N/A"):
                        all_metrics[k] = v

                logger.info(f"PDF #{i + 1} processed: {len(tables)} tables and {len(metrics)} metrics found")

            except Exception as e:
                logger.error(f"Error processing PDF #{i + 1}: {str(e)}")

        # Structure financial data for better organization
        structured_data = self._structure_financial_data(all_text, all_metrics, all_tables)

        # Store the extracted data
        self.extracted_data = {
            "tables": all_tables,
            "metrics": all_metrics,
            "text": all_text,
            "structured_data": structured_data
        }

        logger.info(f"Completed PDF processing. Found {len(all_tables)} tables and {len(all_metrics)} metrics")
        return self.extracted_data

    def _extract_text_with_preprocessing(self, pdf_file: bytes) -> str:
        """Extract text with preprocessing for better number detection."""
        text = ""

        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_file))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {str(e)}")

        # Preprocess the text for better number detection
        processed_text = self._preprocess_text(text)

        logger.info(f"Extracted {len(processed_text)} characters of text from PDF")
        return processed_text

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text to improve number extraction accuracy."""
        processed = text

        # Fix broken numbers (e.g., "$1 , 234" → "$1,234")
        processed = re.sub(r'(\d+)\s*,\s*(\d+)', r'\1,\2', processed)

        # Standardize whitespace around punctuation
        processed = re.sub(r'\s+([,.:])', r'\1', processed)

        # Fix spacing in financial units
        processed = re.sub(r'(\d+)\s+(million|billion|trillion|m|b|t)', r'\1 \2', processed)

        # Standardize financial statement terms
        processed = re.sub(r'earnings\s+per\s+share', 'EPS', processed, flags=re.IGNORECASE)
        processed = re.sub(r'net\s+sales', 'Revenue', processed, flags=re.IGNORECASE)

        return processed

    def _extract_tables_with_number_focus(self, pdf_file: bytes, text: str) -> List[Dict[str, Any]]:
        """Extract tables with focus on preserving numerical data accuracy."""
        processed_tables = []
        temp_filename = None

        try:
            # Generate a unique temporary filename
            temp_filename = f"temp_pdf_{uuid.uuid4()}.pdf"

            # Save PDF to a temporary file for tabula
            with open(temp_filename, "wb") as f:
                f.write(pdf_file)

            # Use tabula with settings optimized for financial tables
            extraction_methods = [
                # Method 1: Lattice mode - best for tables with ruled lines
                lambda: tabula.read_pdf(
                    temp_filename,
                    pages='all',
                    multiple_tables=True,
                    lattice=True,
                    pandas_options={'na_values': ['', ' ', 'N/A', 'n/a', '-']}
                ),
                # Method 2: Stream mode - better for tables without clear borders
                lambda: tabula.read_pdf(
                    temp_filename,
                    pages='all',
                    multiple_tables=True,
                    stream=True,
                    pandas_options={'na_values': ['', ' ', 'N/A', 'n/a', '-']}
                ),
                # Method 3: Guess mode with header detection
                lambda: tabula.read_pdf(
                    temp_filename,
                    pages='all',
                    multiple_tables=True,
                    guess=True,
                    pandas_options={'header': None, 'na_values': ['', ' ', 'N/A', 'n/a', '-']}
                )
            ]

            tables = []
            for i, method in enumerate(extraction_methods):
                try:
                    logger.info(f"Trying table extraction method {i + 1}")
                    method_tables = method()
                    non_empty_tables = [t for t in method_tables if not t.empty]
                    logger.info(f"Method {i + 1} found {len(non_empty_tables)} non-empty tables")

                    if non_empty_tables:
                        tables = method_tables
                        break
                except Exception as method_error:
                    logger.warning(f"Table extraction method {i + 1} failed: {str(method_error)}")

            # Process tables into a more usable format
            for i, table in enumerate(tables):
                if not table.empty:
                    try:
                        # Clean and process table data
                        processed_table = self._process_table_data(table, i)
                        if processed_table:
                            # Score financial relevance
                            financial_score = self._score_financial_relevance(processed_table)
                            processed_table["financial_relevance"] = financial_score

                            processed_tables.append(processed_table)
                            logger.info(
                                f"Table {i}: {table.shape[0]} rows x {table.shape[1]} columns, relevance: {financial_score}")
                    except Exception as table_error:
                        logger.error(f"Error processing table {i}: {str(table_error)}")

            # If no tables found via Tabula, try text-based table detection
            if not processed_tables:
                text_tables = self._extract_tables_text_based(text)
                if text_tables:
                    processed_tables.extend(text_tables)
                    logger.info(f"Found {len(text_tables)} tables via text-based extraction")

            # Clean up temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

            return processed_tables

        except Exception as e:
            logger.error(f"Error extracting tables from PDF: {str(e)}")

            # Clean up temp file if it exists
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except:
                    pass

            return []

    def _process_table_data(self, table: pd.DataFrame, index: int) -> Optional[Dict[str, Any]]:
        """Process and clean table data with focus on numerical values."""
        if table.empty:
            return None

        # Try to identify if this is a header row
        has_header = self._check_if_has_header(table)

        # If no header detected, use column positions
        if not has_header:
            # Create generic column names
            table.columns = [f"Column{j + 1}" for j in range(len(table.columns))]

        # Clean column names (handle potential multi-index)
        if isinstance(table.columns, pd.MultiIndex):
            # Flatten multi-index columns
            table.columns = [' '.join(col).strip() if isinstance(col, tuple) else str(col)
                             for col in table.columns.values]

        # Clean column names - remove whitespace and special characters
        table.columns = [re.sub(r'\s+', ' ', str(col)).strip() for col in table.columns]

        # Clean numeric data - ensure proper typing
        for col in table.columns:
            # Check if this column might contain numeric data
            if table[col].dtype == 'object':
                # Try to convert numbers with currency symbols and commas
                try:
                    # First clean the values
                    cleaned_values = table[col].astype(str).apply(self._clean_numeric_value)
                    # Then try to convert to numeric
                    table[col] = pd.to_numeric(cleaned_values, errors='ignore')
                except:
                    pass

        # Convert to records with cleaned data
        table_dict = {
            "index": index,
            "data": table.replace({pd.NA: None}).to_dict(orient='records'),
            "columns": [str(col).strip() for col in table.columns.tolist()],
            "shape": table.shape,
            "extraction_method": "tabula"
        }

        return table_dict

    def _clean_numeric_value(self, value_str: str) -> str:
        """Clean a potentially numeric value for better conversion."""
        if not value_str or value_str.lower() in ['n/a', 'na', '-', '']:
            return ''

        # Convert to string if not already
        value_str = str(value_str).strip()

        # Remove currency symbols
        value_str = re.sub(r'[$€£¥]', '', value_str)

        # Remove parentheses and replace with negative
        if '(' in value_str and ')' in value_str:
            value_str = value_str.replace('(', '-').replace(')', '')

        # Remove commas in numbers
        value_str = value_str.replace(',', '')

        # Remove % symbol
        value_str = value_str.replace('%', '')

        return value_str

    def _extract_tables_text_based(self, text: str) -> List[Dict[str, Any]]:
        """Extract tables from text using pattern recognition with number focus."""
        tables = []

        # Split text into lines
        lines = text.split('\n')

        # Identify potential table regions
        table_regions = []
        current_region = []
        in_potential_table = False

        for line in lines:
            # Count potential delimiters
            tab_count = line.count('\t')
            pipe_count = line.count('|')
            space_groups = len(re.findall(r'\s{2,}', line))
            number_count = len(re.findall(r'\d+(\.\d+)?', line))

            # Detect possible table rows - prioritize lines with numbers
            is_potential_row = (
                    (tab_count >= 2) or
                    (pipe_count >= 2) or
                    (space_groups >= 2 and number_count >= 2) or
                    (re.search(r'\d+\s+\d+\s+\d+', line) is not None)  # Numbers separated by whitespace
            )

            if is_potential_row:
                if not in_potential_table:
                    in_potential_table = True
                    current_region = [line]
                else:
                    current_region.append(line)
            else:
                if in_potential_table and len(current_region) >= 3:
                    # Found a table with at least 3 rows
                    table_regions.append(current_region)
                in_potential_table = False
                current_region = []

        # Process identified regions
        for idx, region in enumerate(table_regions):
            try:
                # Convert text region to structured table
                table_dict = self._convert_text_region_to_table(region, idx)
                if table_dict:
                    # Score financial relevance
                    financial_score = self._score_financial_relevance(table_dict)
                    table_dict["financial_relevance"] = financial_score
                    tables.append(table_dict)
            except Exception as e:
                logger.error(f"Error processing text table region {idx}: {str(e)}")

        return tables

    def _convert_text_region_to_table(self, region: List[str], idx: int) -> Optional[Dict[str, Any]]:
        """Convert a text region to a structured table format with focus on numbers."""
        if not region:
            return None

        # Determine delimiter type
        first_line = region[0]
        delimiter = None

        if '|' in first_line and first_line.count('|') >= 2:
            delimiter = '|'
        elif '\t' in first_line and first_line.count('\t') >= 2:
            delimiter = '\t'
        else:
            # Convert multiple spaces to a delimiter
            processed_region = []
            for line in region:
                processed_line = re.sub(r'\s{2,}', '|', line)
                processed_region.append(processed_line)
            region = processed_region
            delimiter = '|'

        # Process header and rows
        if delimiter:
            # Get potential header
            header = region[0].split(delimiter)
            header = [h.strip() for h in header if h.strip()]

            # Process data rows
            data_rows = []
            for row in region[1:]:
                cells = row.split(delimiter)
                cells = [c.strip() for c in cells if c.strip()]

                # Ensure consistent cell count
                while len(cells) < len(header):
                    cells.append("")

                if len(cells) > len(header) and len(header) > 0:
                    # Combine extra cells
                    last_column = ' '.join(cells[len(header) - 1:])
                    cells = cells[:len(header) - 1] + [last_column]

                if cells:
                    row_dict = {}
                    for i, h in enumerate(header):
                        if i < len(cells):
                            col_name = h if h else f"Column{i + 1}"
                            # Clean cell value with focus on numbers
                            cell_value = cells[i]

                            # Detect and clean numeric values
                            if re.search(r'[\$€£]?[\d\.,]+', cell_value):
                                # Try to preserve the original number format while ensuring it's clean
                                pure_number = self._clean_numeric_value(cell_value)
                                try:
                                    # If it's a valid number, convert it
                                    num_value = float(pure_number)
                                    # Keep original formatting for display
                                    row_dict[col_name] = cell_value
                                except ValueError:
                                    row_dict[col_name] = cell_value
                            else:
                                row_dict[col_name] = cell_value
                    data_rows.append(row_dict)

            # Return structured table if data exists
            if data_rows:
                return {
                    "index": idx,
                    "data": data_rows,
                    "columns": header if header else [f"Column{i + 1}" for i in range(len(data_rows[0]))],
                    "extraction_method": "text_based"
                }

        return None

    def _check_if_has_header(self, table: pd.DataFrame) -> bool:
        """Check if a table has a proper header row."""
        # If columns are not default numerical indexes
        if not all(isinstance(col, int) for col in table.columns):
            # Check for meaningful column names
            col_str = ''.join(str(col) for col in table.columns)
            if len(col_str.strip()) > 5:
                return True

        # Check first row for header-like content
        if not table.empty:
            first_row = table.iloc[0]
            # If first row contains mostly strings (potential headers)
            if sum(isinstance(val, str) for val in first_row) / len(first_row) > 0.7:
                return True

        return False

    def _score_financial_relevance(self, table_dict: Dict[str, Any]) -> int:
        """Score how likely a table is to contain financial data with focus on numbers."""
        score = 0

        # Financial terms to look for
        financial_terms = [
            'revenue', 'eps', 'earnings', 'income', 'profit', 'margin',
            'quarter', 'fiscal', 'guidance', 'growth', 'financial',
            'gaap', 'non-gaap', 'diluted', 'million', 'billion'
        ]

        # Check column names
        col_str = ' '.join(str(col).lower() for col in table_dict.get('columns', []))
        for term in financial_terms:
            if term in col_str:
                score += 3

        # Check data content
        data_str = str(table_dict.get('data', [])).lower()

        # Check for currency symbols (high importance for financial tables)
        for symbol in ['$', '€', '£', 'usd', 'eur']:
            if symbol in data_str:
                score += 3  # Increased weight

        # Check for percentage symbols
        if '%' in data_str:
            score += 2

        # Check for numeric content with currency patterns
        currency_pattern = r'[\$€£][\d\.,]+'
        currency_matches = re.findall(currency_pattern, data_str)
        score += min(len(currency_matches) * 2, 10)  # Prioritize currency values, cap at 10 points

        # Check for decimal numbers (likely financial data)
        numeric_pattern = r'\d+\.\d+'
        numeric_matches = re.findall(numeric_pattern, data_str)
        score += min(len(numeric_matches), 5)  # Cap at 5 points

        # Check for date/quarter patterns
        date_patterns = ['q1', 'q2', 'q3', 'q4', 'quarter', 'fy20', 'fy21', 'fy22', 'fy23', 'fy24', 'fy25']
        for pattern in date_patterns:
            if pattern in data_str:
                score += 1

        # Bonus for "million" or "billion" in data
        for term in ['million', 'billion', 'm', 'b']:
            if term in data_str:
                score += 2  # Increased weight

        return score

    def _extract_key_financial_metrics(self, text: str) -> Dict[str, str]:
        """Extract key financial metrics directly from text with focus on numbers."""
        # This is a basic direct extraction - a more comprehensive approach
        # is in the MetricsExtractor class
        metrics = {}

        # Define regex patterns for most important financial metrics
        key_patterns = {
            # EPS with strict number formatting
            "eps": [
                r"(?:EPS|Earnings Per Share)[:\s]*\$([\d\.,]+)",
                r"diluted EPS[:\s]*\$([\d\.,]+)",
                r"reported EPS[:\s]*\$([\d\.,]+)"
            ],

            # Revenue with billions/millions indicators
            "revenue": [
                r"(?:Revenue|Total Revenue)[:\s]*\$([\d\.,]+)\s*(?:billion|million|B|M)",
                r"revenue of \$([\d\.,]+)\s*(?:billion|million|B|M)",
                r"reported revenue[^\.]*?\$([\d\.,]+)\s*(?:billion|million|B|M)"
            ],

            # Net Income with numbers
            "net_income": [
                r"(?:Net Income|Net Earnings)[:\s]*\$([\d\.,]+)\s*(?:billion|million|B|M)",
                r"net income of \$([\d\.,]+)\s*(?:billion|million|B|M)"
            ]
        }

        # Apply the patterns
        for metric, pattern_list in key_patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Use the first match
                    metrics[metric] = self._format_metric_value(matches[0], pattern)
                    break

        return metrics

    def _format_metric_value(self, value: str, pattern: str) -> str:
        """Format metric value consistently based on pattern context."""
        # Clean the value
        value = value.replace(',', '')

        # Determine unit based on pattern
        if "billion" in pattern.lower() or "b" in pattern.lower():
            return f"${float(value)}B"
        elif "million" in pattern.lower() or "m" in pattern.lower():
            return f"${float(value)}M"
        else:
            # For EPS or unitless values
            return f"${float(value)}"

    def _structure_financial_data(self, text: str, metrics: Dict[str, str], tables: List[Dict[str, Any]]) -> Dict[
        str, Any]:
        """Organize extracted data into a structured format."""
        structured_data = {
            "reported_values": {},
            "expected_values": {},
            "guidance": {},
            "quarterly_financials": {},
            "dividend_info": {}
        }

        # Add reported values from metrics
        if "eps" in metrics:
            structured_data["reported_values"]["eps"] = metrics["eps"]
        if "revenue" in metrics:
            structured_data["reported_values"]["revenue"] = metrics["revenue"]
        if "net_income" in metrics:
            structured_data["reported_values"]["net_income"] = metrics["net_income"]
        if "gross_margin" in metrics:
            structured_data["reported_values"]["gross_margin"] = metrics["gross_margin"]
        if "operating_income" in metrics:
            structured_data["reported_values"]["operating_income"] = metrics["operating_income"]

        # Add guidance if available
        if "guidance" in metrics:
            structured_data["guidance"]["guidance"] = metrics["guidance"]
        if "revenue_growth" in metrics:
            structured_data["guidance"]["revenue_growth"] = metrics["revenue_growth"]

        # Extract quarterly data from tables with focus on numbers
        quarterly_data = self._extract_quarterly_data_from_tables(tables)
        if quarterly_data:
            structured_data["quarterly_financials"] = quarterly_data

        # Extract dividend information
        dividend_info = self._extract_dividend_info(text)
        if dividend_info:
            structured_data["dividend_info"] = dividend_info

        return structured_data

    def _extract_quarterly_data_from_tables(self, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract quarterly financial data from tables with focus on numbers."""
        quarterly_data = {}

        # Sort tables by financial relevance
        relevant_tables = sorted(tables, key=lambda t: t.get("financial_relevance", 0), reverse=True)

        if not relevant_tables:
            return quarterly_data

        # Try to extract from the most relevant tables
        for table in relevant_tables[:3]:  # Check top 3 most relevant tables
            data = table.get("data", [])
            columns = table.get("columns", [])

            # Skip empty tables
            if not data or not columns:
                continue

            # Check if this looks like a quarterly financial table
            col_str = ' '.join(str(col).lower() for col in columns)
            if any(term in col_str for term in ['quarter', 'q1', 'q2', 'q3', 'q4', 'fiscal']):
                # This might be a quarterly table
                for row in data:
                    # Look for key financial metrics in this row
                    row_str = str(row).lower()
                    if any(term in row_str for term in ['revenue', 'sales', 'income', 'eps', 'earnings']):
                        # Extract the metric name and values
                        metric_name = None
                        current_value = None
                        previous_value = None

                        # Try to identify the metric name column
                        for col in columns:
                            col_lower = str(col).lower()
                            if 'metric' in col_lower or 'item' in col_lower or 'description' in col_lower:
                                if col in row:
                                    metric_name = row[col]
                                    break

                        # If we couldn't find a designated metric column, use the first column
                        if not metric_name and len(columns) > 0:
                            metric_name = row.get(columns[0], None)

                        # Look for current and previous period values
                        for col in columns:
                            col_lower = str(col).lower()
                            value = row.get(col, None)

                            # Skip empty values
                            if not value:
                                continue

                            # Current period indicators
                            if any(term in col_lower for term in
                                   ['current', 'this', 'latest', 'q4', 'q1 2024', 'q1 24']):
                                current_value = value

                            # Previous period indicators
                            if any(term in col_lower for term in
                                   ['previous', 'prior', 'last', 'q4 2022', 'q1 2023', 'q1 23']):
                                previous_value = value

                        # If we found a metric and at least one value, add it to our data
                        if metric_name and (current_value or previous_value):
                            # Clean up the metric name
                            clean_metric = metric_name.lower().strip()
                            if clean_metric:
                                quarterly_data[clean_metric] = {
                                    "current": self._clean_financial_value(current_value),
                                    "previous": self._clean_financial_value(previous_value)
                                }

        return quarterly_data

    def _clean_financial_value(self, value) -> Optional[str]:
        """Clean financial value ensuring consistent formatting."""
        if value is None:
            return None

        value_str = str(value).strip()
        if value_str == "" or value_str.lower() == "n/a":
            return None

        # Handle currency values
        if '$' in value_str or re.match(r'^[\d\.,]+$', value_str):
            # Extract just the number
            num_match = re.search(r'([\d\.,]+)', value_str)
            if num_match:
                number = num_match.group(1).replace(',', '')
                try:
                    float_val = float(number)

                    # Check for unit indicators
                    if any(unit in value_str.lower() for unit in ['billion', 'b']):
                        return f"${float_val}B"
                    elif any(unit in value_str.lower() for unit in ['million', 'm']):
                        return f"${float_val}M"
                    else:
                        # If no unit, determine based on size
                        if float_val >= 1000000000:
                            return f"${float_val / 1000000000:.2f}B"
                        elif float_val >= 1000000:
                            return f"${float_val / 1000000:.2f}M"
                        else:
                            return f"${float_val:.2f}"
                except ValueError:
                    return value_str

        # Handle percentage values
        if '%' in value_str:
            # Extract just the percentage
            pct_match = re.search(r'([\d\.,]+)', value_str)
            if pct_match:
                try:
                    pct_val = float(pct_match.group(1).replace(',', ''))
                    return f"{pct_val:.2f}%"
                except ValueError:
                    return value_str

        return value_str

    def _extract_dividend_info(self, text: str) -> Dict[str, str]:
        """Extract dividend and stock buyback information with focus on accurate numbers."""
        dividend_info = {}

        # Dividend patterns
        dividend_patterns = [
            r"dividend of \$([\d\.]+)",
            r"quarterly dividend[:\s]*([\$]?[\d\.,]+)",
            r"dividend[:\s]*([\$]?[\d\.,]+) per share",
            r"increased dividend[^\.]*?([\$]?[\d\.,]+)"
        ]

        # Buyback patterns
        buyback_patterns = [
            r"stock repurchase[^\.]*?\$([\d\.,]+)\s*(?:million|billion|m|b|M|B|MM)",
            r"share buyback[^\.]*?\$([\d\.,]+)\s*(?:million|billion|m|b|M|B|MM)",
            r"buyback program[^\.]*?\$([\d\.,]+)\s*(?:million|billion|m|b|M|B|MM)"
        ]

        # Stock split patterns
        split_patterns = [
            r"stock split[^\.]*?(\d+)[:\s]*for[:\s]*(\d+)",
            r"(\d+)[:\s]*for[:\s]*(\d+) stock split"
        ]

        # Check for dividend info
        for pattern in dividend_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                value = matches[0].replace(',', '')
                try:
                    # Format as dollar amount
                    float_val = float(value.replace('$', ''))
                    dividend_info["dividend"] = f"${float_val:.2f} per share"
                except ValueError:
                    dividend_info["dividend"] = matches[0]
                break

        # Check for buyback info
        for pattern in buyback_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                value = matches[0].replace(',', '')
                try:
                    float_val = float(value)
                    # Determine unit based on context
                    if "billion" in pattern.lower() or "b" in pattern.lower():
                        dividend_info["buyback"] = f"${float_val}B"
                    else:
                        dividend_info["buyback"] = f"${float_val}M"
                except ValueError:
                    dividend_info["buyback"] = matches[0]
                break

        # Check for stock split info
        for pattern in split_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if isinstance(matches[0], tuple) and len(matches[0]) == 2:
                    ratio = f"{matches[0][0]}:{matches[0][1]}"
                    dividend_info["stock_split"] = ratio
                else:
                    dividend_info["stock_split"] = str(matches[0])
                break

        return dividend_info