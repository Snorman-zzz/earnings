import logging
import re
from typing import Dict, Any, Optional, List, Union
from openai import OpenAI

logger = logging.getLogger(__name__)


class FileSearch:
    """File search with focus on reliable financial data extraction."""

    def __init__(self, client: OpenAI, pdf_extracted_data: Dict[str, Any]):
        """
        Initialize the file search tool with focus on numerical accuracy.

        Args:
            client: OpenAI API client
            pdf_extracted_data: Dictionary of PDF data extracted directly
        """
        self.client = client
        self.pdf_extracted_data = pdf_extracted_data
        self.search_history = []
        self.search_cache = {}  # Cache search results to avoid repetitive API calls
        self.failure_count = 0  # Track failed OpenAI searches
        self.max_failures = 3  # Switch to local data only after this many failures
        self.use_local_data_only = False

    def search(self, query: str, vector_store_id: Optional[str] = None) -> str:
        """
        Search for financial information with focus on accurate numerical data.

        Args:
            query: The search query
            vector_store_id: ID of the vector store to search in

        Returns:
            Search results as a string with consistent number formatting
        """
        logger.info(f"File search initiated for query: {query}")
        logger.info(f"Using vector store: {vector_store_id or 'None'}")

        # Check cache first to avoid redundant searches
        cache_key = f"{query}_{vector_store_id}"
        if cache_key in self.search_cache:
            logger.info(f"Returning cached result for query: {query}")
            return self.search_cache[cache_key]

        # Add to search history
        self.search_history.append(query)

        # Use local data only if we've had too many failures
        if self.use_local_data_only:
            logger.warning("Using local data only due to previous search failures")
            result = self._comprehensive_local_search(query)
            self.search_cache[cache_key] = result
            return result

        # Try multiple search strategies in sequence with focus on numerical data
        result = None

        # Strategy 1: Direct metric lookup for financial metrics
        if not result:
            result = self._direct_metric_lookup(query)
            if result:
                logger.info("Found result via direct metric lookup")

        # Strategy 2: Structured data lookup for organized financial data
        if not result and "structured_data" in self.pdf_extracted_data:
            result = self._structured_data_lookup(query)
            if result:
                logger.info("Found result via structured data lookup")

        # Strategy 3: Table search specifically for financial tables
        if not result:
            result = self._financial_table_search(query)
            if result:
                logger.info("Found result via financial table search")

        # Strategy 4: OpenAI vector search with error handling
        if not result and vector_store_id:
            try:
                result = self._openai_vector_search(query, vector_store_id)
                if result:
                    logger.info("Found result via OpenAI vector search")
                    # Reset failure count on success
                    self.failure_count = 0
            except Exception as e:
                logger.error(f"OpenAI vector search failed: {str(e)}")
                self.failure_count += 1
                if self.failure_count >= self.max_failures:
                    logger.warning(f"Switching to local data only after {self.failure_count} failures")
                    self.use_local_data_only = True

        # Strategy 5: Direct text search in financial contexts
        if not result:
            result = self._financial_text_search(query)
            if result:
                logger.info("Found result via direct text search")

        # Final fallback: try a comprehensive search in all local data
        if not result:
            result = self._comprehensive_local_search(query)
            if result:
                logger.info("Found result via comprehensive local search")

        # If all strategies fail, provide a helpful response
        if not result:
            result = self._create_fallback_response(query)

        # Process result to ensure consistent number formatting
        result = self._ensure_consistent_number_formats(result)

        # Cache the result
        self.search_cache[cache_key] = result
        logger.info(f"Search complete. Result length: {len(result)}")
        return result

    def _direct_metric_lookup(self, query: str) -> Optional[str]:
        """Look up specific financial metrics with enhanced number formatting."""
        query_lower = query.lower()
        metrics = self.pdf_extracted_data.get("metrics", {})

        # Check for EPS queries with expanded pattern matching
        if any(term in query_lower for term in ["eps", "earnings per share", "earnings", "per share"]):
            # Try multiple EPS variants
            for eps_key in ["eps", "adj_eps"]:
                if eps_key in metrics and metrics[eps_key]:
                    eps_type = "adjusted" if eps_key == "adj_eps" else "reported"
                    value = metrics[eps_key]
                    logger.info(f"Found {eps_type} EPS in metrics: {value}")

                    # Ensure consistent dollar formatting
                    if not value.startswith("$"):
                        value = f"${value}"

                    return f"The {eps_type} EPS is {value}"

        # Check for revenue queries with unit standardization
        if any(term in query_lower for term in ["revenue", "sales", "top line", "turnover"]):
            if "revenue" in metrics and metrics["revenue"]:
                value = metrics["revenue"]
                logger.info(f"Found Revenue in metrics: {value}")

                # Ensure consistent formatting with units
                if not value.startswith("$"):
                    value = f"${value}"

                # Make sure unit is clear (M/B)
                if not any(unit in value for unit in ["M", "B", "million", "billion"]):
                    try:
                        num_val = float(value.replace("$", "").replace(",", ""))
                        if num_val >= 1000000000:
                            value = f"${num_val / 1000000000:.2f}B"
                        elif num_val >= 1000000:
                            value = f"${num_val / 1000000:.2f}M"
                    except ValueError:
                        pass

                return f"The reported revenue is {value}"

            # Check for revenue growth if direct revenue not found
            if "revenue_growth" in metrics and metrics["revenue_growth"]:
                value = metrics["revenue_growth"]
                logger.info(f"Found Revenue growth: {value}")

                # Ensure percentage format
                if not "%" in value:
                    value = f"{value}%"

                return f"The revenue growth is {value}"

        # Check for income/profit queries with unit standardization
        if any(term in query_lower for term in
               ["income", "net income", "profit", "earnings", "bottom line", "net profit"]):
            if "net_income" in metrics and metrics["net_income"]:
                value = metrics["net_income"]
                logger.info(f"Found Net Income: {value}")

                # Ensure consistent formatting with units
                if not value.startswith("$"):
                    value = f"${value}"

                # Make sure unit is clear (M/B)
                if not any(unit in value for unit in ["M", "B", "million", "billion"]):
                    try:
                        num_val = float(value.replace("$", "").replace(",", ""))
                        if num_val >= 1000000000:
                            value = f"${num_val / 1000000000:.2f}B"
                        elif num_val >= 1000000:
                            value = f"${num_val / 1000000:.2f}M"
                    except ValueError:
                        pass

                return f"The reported net income is {value}"

        # Check for margin queries with percentage formatting
        if any(term in query_lower for term in ["margin", "gross margin", "profit margin", "operating margin"]):
            if "gross_margin" in metrics and metrics["gross_margin"]:
                value = metrics["gross_margin"]
                logger.info(f"Found Gross Margin: {value}")

                # Ensure percentage format
                if not "%" in value:
                    value = f"{value}%"

                return f"The reported gross margin is {value}"

        # Check for operating income queries with unit standardization
        if any(term in query_lower for term in ["operating", "operating income", "operations", "operating profit"]):
            if "operating_income" in metrics and metrics["operating_income"]:
                value = metrics["operating_income"]
                logger.info(f"Found Operating Income: {value}")

                # Ensure consistent formatting with units
                if not value.startswith("$"):
                    value = f"${value}"

                # Make sure unit is clear (M/B)
                if not any(unit in value for unit in ["M", "B", "million", "billion"]):
                    try:
                        num_val = float(value.replace("$", "").replace(",", ""))
                        if num_val >= 1000000000:
                            value = f"${num_val / 1000000000:.2f}B"
                        elif num_val >= 1000000:
                            value = f"${num_val / 1000000:.2f}M"
                    except ValueError:
                        pass

                return f"The reported operating income is {value}"

        # Check for guidance/outlook queries
        if any(term in query_lower for term in ["guidance", "outlook", "forecast", "next quarter", "future"]):
            if "guidance" in metrics and metrics["guidance"]:
                value = metrics["guidance"]
                logger.info(f"Found Guidance: {value}")
                return f"The company guidance is {value}"

        # Check for dividend/buyback queries
        if any(term in query_lower for term in ["dividend", "buyback", "repurchase", "capital return", "payout"]):
            structured_data = self.pdf_extracted_data.get("structured_data", {})
            dividend_info = structured_data.get("dividend_info", {})
            if dividend_info:
                logger.info(f"Found Dividend Info: {dividend_info}")
                dividend_parts = []
                for k, v in dividend_info.items():
                    # Format currency values consistently
                    if k == "dividend" and not v.startswith("$"):
                        v = f"${v}"
                    if k == "buyback" and not v.startswith("$"):
                        v = f"${v}"
                    dividend_parts.append(f"{k.replace('_', ' ').title()}: {v}")
                return "Dividend and capital return information:\n" + "\n".join(dividend_parts)

        return None

    def _structured_data_lookup(self, query: str) -> Optional[str]:
        """Look up information in structured financial data."""
        query_lower = query.lower()
        structured_data = self.pdf_extracted_data.get("structured_data", {})

        # Empty structured data
        if not structured_data:
            return None

        # Calculate relevance scores for each section based on the query
        section_scores = {
            "reported_values": 0,
            "expected_values": 0,
            "guidance": 0,
            "quarterly_financials": 0,
            "dividend_info": 0
        }

        # Score each section based on query terms
        if any(term in query_lower for term in ["actual", "reported", "results"]):
            section_scores["reported_values"] += 3
        if any(term in query_lower for term in ["expected", "estimates", "consensus", "analyst"]):
            section_scores["expected_values"] += 3
        if any(term in query_lower for term in ["guidance", "outlook", "forecast", "future"]):
            section_scores["guidance"] += 3
        if any(term in query_lower for term in ["quarterly", "q1", "q2", "q3", "q4", "quarter"]):
            section_scores["quarterly_financials"] += 3
        if any(term in query_lower for term in ["dividend", "buyback", "split", "capital return"]):
            section_scores["dividend_info"] += 3

        # Look at specific financial metrics in the query
        if any(term in query_lower for term in ["eps", "earnings per share"]):
            section_scores["reported_values"] += 2
            section_scores["expected_values"] += 2
        if any(term in query_lower for term in ["revenue", "sales"]):
            section_scores["reported_values"] += 2
            section_scores["expected_values"] += 2
        if any(term in query_lower for term in ["income", "profit", "margin"]):
            section_scores["reported_values"] += 2
            section_scores["quarterly_financials"] += 1

        # Find highest scoring section with available data
        highest_score = 0
        best_section = None
        for section, score in section_scores.items():
            if score > highest_score and section in structured_data and structured_data[section]:
                highest_score = score
                best_section = section

        # If we found a good section, generate response
        if best_section and highest_score > 0:
            section_data = structured_data[best_section]
            if section_data:
                response = f"Here is information from the {best_section.replace('_', ' ')}:\n\n"
                for key, value in section_data.items():
                    if isinstance(value, dict):
                        response += f"- {key.replace('_', ' ').title()}:\n"
                        for subkey, subvalue in value.items():
                            # Format numbers consistently
                            if subvalue and isinstance(subvalue, str):
                                subvalue = self._format_financial_value(subvalue, key)
                            response += f"  * {subkey.title()}: {subvalue}\n"
                    else:
                        # Format numbers consistently
                        if value and isinstance(value, str):
                            value = self._format_financial_value(value, key)
                        response += f"- {key.replace('_', ' ').title()}: {value}\n"
                return response

        # If no good match, but we have specific data requested, try direct look-up
        specific_metrics = ["eps", "revenue", "net income", "operating income", "gross margin"]
        for metric in specific_metrics:
            if metric in query_lower:
                for section in structured_data.values():
                    if isinstance(section, dict):
                        for key, value in section.items():
                            if metric in key.lower() and value:
                                # Format numbers consistently
                                if isinstance(value, str):
                                    value = self._format_financial_value(value, key)
                                return f"The {key.replace('_', ' ')} is {value}"

        return None

    def _format_financial_value(self, value_str: str, metric_type: str) -> str:
        """Format financial values consistently based on the metric type."""
        if not value_str:
            return value_str

        # Format EPS values with dollar sign and 2 decimal places
        if "eps" in metric_type.lower():
            if not value_str.startswith("$"):
                try:
                    num_val = float(value_str.replace("$", "").replace(",", ""))
                    return f"${num_val:.2f}"
                except ValueError:
                    return value_str

        # Format revenue/income values with dollar sign and units
        if any(term in metric_type.lower() for term in ["revenue", "income", "profit"]):
            if not value_str.startswith("$"):
                value_str = f"${value_str}"

            # Ensure unit is visible
            if not any(unit in value_str for unit in ["M", "B", "million", "billion"]):
                try:
                    num_val = float(value_str.replace("$", "").replace(",", ""))
                    if num_val >= 1000000000:
                        return f"${num_val / 1000000000:.2f}B"
                    elif num_val >= 1000000:
                        return f"${num_val / 1000000:.2f}M"
                except ValueError:
                    pass

        # Format percentage values
        if any(term in metric_type.lower() for term in ["margin", "growth", "percentage"]):
            if not "%" in value_str:
                try:
                    num_val = float(value_str.replace("%", "").replace(",", ""))
                    return f"{num_val:.2f}%"
                except ValueError:
                    return value_str

        return value_str

    def _financial_table_search(self, query: str) -> Optional[str]:
        """Search through tables for relevant financial data."""
        query_lower = query.lower()
        tables = self.pdf_extracted_data.get("tables", [])

        if not tables:
            return None

        # Extract key terms from the query
        query_terms = []
        for term in query_lower.split():
            if len(term) > 3:  # Skip short words
                query_terms.append(term)

        # Add financial terms if they appear in the query
        financial_terms = ["eps", "revenue", "income", "margin", "profit", "sales", "growth"]
        financial_query_terms = [term for term in financial_terms if term in query_lower]
        query_terms.extend(financial_query_terms)

        # Score tables for financial relevance to the query
        scored_tables = []
        for table in tables:
            relevance_score = 0
            table_str = str(table).lower()

            # Check if query terms appear in the table
            for term in query_terms:
                if term in table_str:
                    relevance_score += 2

            # Check column headers for financial relevance
            if "columns" in table:
                column_str = str(table["columns"]).lower()
                for term in ["revenue", "income", "earnings", "eps", "financial", "quarter", "margin"]:
                    if term in column_str:
                        relevance_score += 1

                # Check if our query topic appears in columns
                for term in query_terms:
                    if term in column_str:
                        relevance_score += 3

            # Check data for financial relevance
            if "data" in table:
                data_str = str(table["data"]).lower()

                # Check if our query topic appears in data
                for term in query_terms:
                    if term in data_str:
                        relevance_score += 2

                # Bonus for tables with currency symbols (highly likely financial data)
                currency_matches = len(re.findall(r'[\$€£]\d+', data_str))
                relevance_score += min(currency_matches, 5)  # Cap at 5 points

                # Bonus for tables with percentage values
                percentage_matches = len(re.findall(r'\d+%', data_str))
                relevance_score += min(percentage_matches, 3)  # Cap at 3 points

            # Add financial relevance score from table extraction if available
            relevance_score += table.get("financial_relevance", 0)

            if relevance_score > 0:
                scored_tables.append((relevance_score, table))

        # Sort by relevance score
        scored_tables.sort(reverse=True, key=lambda x: x[0])

        if not scored_tables:
            return None

        # Format the most relevant table
        most_relevant = scored_tables[0][1]

        if "data" not in most_relevant or "columns" not in most_relevant:
            return None

        table_data = most_relevant["data"]
        columns = most_relevant["columns"]

        if not table_data:
            return None

        # Include extraction method for diagnostic info
        extraction_method = most_relevant.get("extraction_method", "unknown")
        result = f"I found a relevant financial table (extracted via {extraction_method}):\n\n"
        result += f"Columns: {', '.join(str(col) for col in columns)}\n"
        result += "Data:\n"

        # Limit to first 5 rows for readability
        for idx, row in enumerate(table_data[:5]):
            # Format currency and percentage values consistently
            formatted_row = {}
            for key, value in row.items():
                if isinstance(value, str):
                    # Format currency values
                    if '$' in value:
                        try:
                            # Extract numeric part
                            num_match = re.search(r'[\$]?([\d\.,]+)', value)
                            if num_match:
                                num_val = float(num_match.group(1).replace(',', ''))
                                if num_val >= 1000000000:
                                    formatted_row[key] = f"${num_val / 1000000000:.2f}B"
                                elif num_val >= 1000000:
                                    formatted_row[key] = f"${num_val / 1000000:.2f}M"
                                else:
                                    formatted_row[key] = f"${num_val:.2f}"
                            else:
                                formatted_row[key] = value
                        except ValueError:
                            formatted_row[key] = value
                    # Format percentage values
                    elif '%' in value:
                        try:
                            pct_match = re.search(r'([\d\.,]+)', value)
                            if pct_match:
                                pct_val = float(pct_match.group(1).replace(',', ''))
                                formatted_row[key] = f"{pct_val:.2f}%"
                            else:
                                formatted_row[key] = value
                        except ValueError:
                            formatted_row[key] = value
                    else:
                        formatted_row[key] = value
                else:
                    formatted_row[key] = value

            result += f"Row {idx + 1}: {formatted_row}\n"

        if len(table_data) > 5:
            result += f"... (and {len(table_data) - 5} more rows)\n"

        logger.info(f"Found relevant financial table with score {scored_tables[0][0]}")
        return result

    def _openai_vector_search(self, query: str, vector_store_id: str) -> Optional[str]:
        """Search using OpenAI's vector search with focus on accurate financial data."""
        max_retries = 2
        retry_count = 0
        backoff_time = 2  # seconds

        while retry_count <= max_retries:
            try:
                logger.info(f"Using OpenAI search with vector store ID {vector_store_id} (attempt {retry_count + 1})")

                # Create a message that specifically focuses on numerical financial data
                search_message = (
                    f"Find specific financial data in the earnings documents for: {query}. "
                    f"Focus on extracting precise numbers, including dollar values and percentages. "
                    f"Report exact figures with correct units (billions, millions) and maintain "
                    f"consistent number formatting in your response."
                )

                # For stability, use GPT-4 when available, fallback to other models
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system",
                             "content": "You are a financial data analyst searching through earnings reports. "
                                        "Extract and report financial numbers with high precision. "
                                        "Always use consistent formatting: $X.XXM/B for monetary values and "
                                        "X.XX% for percentages. Never round or approximate financial figures."},
                            {"role": "user", "content": search_message}
                        ],
                        tools=[{
                            "type": "retrieval"
                        }],
                        file_search={
                            "vector_store_ids": [vector_store_id]
                        }
                    )
                except Exception as model_error:
                    logger.warning(f"Error with GPT-4o, trying fallback model: {str(model_error)}")
                    # Fallback to GPT-3.5 if GPT-4 fails
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system",
                             "content": "You are a financial data analyst searching through earnings reports. "
                                        "Extract and report financial numbers with high precision. "
                                        "Always use consistent formatting: $X.XXM/B for monetary values and "
                                        "X.XX% for percentages. Never round or approximate financial figures."},
                            {"role": "user", "content": search_message}
                        ],
                        tools=[{
                            "type": "retrieval"
                        }],
                        file_search={
                            "vector_store_ids": [vector_store_id]
                        }
                    )

                content = response.choices[0].message.content
                logger.info(f"OpenAI search results: {content[:200]}...")

                # Validate response - it should contain numbers for financial queries
                if self._is_financial_query(query) and not self._contains_financial_numbers(content):
                    logger.warning("OpenAI search result doesn't contain expected financial data")
                    # Don't return invalid results - try another strategy
                    if retry_count < max_retries:
                        retry_count += 1
                        import time
                        time.sleep(backoff_time)
                        backoff_time *= 2  # Exponential backoff
                        continue
                    return None

                return content

            except Exception as e:
                logger.error(f"Error with OpenAI search (attempt {retry_count + 1}): {str(e)}")

                # Retry logic
                if retry_count < max_retries:
                    retry_count += 1
                    import time
                    time.sleep(backoff_time)
                    backoff_time *= 2  # Exponential backoff
                else:
                    # Max retries reached, give up
                    self.failure_count += 1
                    if self.failure_count >= self.max_failures:
                        logger.warning(f"Switching to local data only after {self.failure_count} failures")
                        self.use_local_data_only = True
                    return None

        return None

    def _is_financial_query(self, query: str) -> bool:
        """Check if a query is asking for financial data that should have numbers."""
        financial_terms = [
            "eps", "earnings per share", "revenue", "sales", "income", "profit",
            "margin", "growth", "increase", "decrease", "dollars", "billions", "millions"
        ]
        query_lower = query.lower()
        return any(term in query_lower for term in financial_terms)

    def _contains_financial_numbers(self, text: str) -> bool:
        """Check if the text contains properly formatted financial numbers."""
        # Look for currency values with correct formatting
        has_currency = bool(re.search(r'[\$€£]\s*\d+(?:[.,]\d+)?(?:\s*[BMK])?', text))

        # Look for percentage values
        has_percentages = bool(re.search(r'\d+(?:[.,]\d+)?\s*%', text))

        # Look for numbers with units like billion/million
        has_units = bool(re.search(r'\d+(?:[.,]\d+)?\s*(?:billion|million|B|M)', text, re.IGNORECASE))

        return has_currency or has_percentages or has_units

    def _financial_text_search(self, query: str) -> Optional[str]:
        """Search directly in the extracted text focusing on financial context."""
        text = self.pdf_extracted_data.get("text", "")

        if not text or len(text) < 100:
            return None

        # Create simplified query terms
        query_terms = [term.lower() for term in query.split() if len(term) > 3]

        if not query_terms:
            return None

        # Try to find paragraphs containing query terms with financial data
        paragraphs = text.split("\n\n")
        scored_paragraphs = []

        for paragraph in paragraphs:
            if len(paragraph.strip()) < 20:  # Skip very short paragraphs
                continue

            paragraph_lower = paragraph.lower()
            score = 0

            # Score based on query term matches
            for term in query_terms:
                occurrences = paragraph_lower.count(term)
                if occurrences > 0:
                    score += 2 * occurrences  # Increased weight for query terms

            # High bonus for paragraphs with financial numbers
            currency_matches = len(re.findall(r'[\$€£]\d+', paragraph_lower))
            score += currency_matches * 3  # Strong weight for currency values

            percentage_matches = len(re.findall(r'\d+%', paragraph_lower))
            score += percentage_matches * 2  # Medium weight for percentages

            # Bonus for financial units
            unit_matches = len(re.findall(r'\d+\s*(?:billion|million|b|m)', paragraph_lower, re.IGNORECASE))
            score += unit_matches * 2  # Medium weight for units

            # Bonus for financial terms
            for term in ["revenue", "earnings", "eps", "income", "margin", "profit", "growth"]:
                if term in paragraph_lower:
                    score += 1

            if score > 0:
                scored_paragraphs.append((score, paragraph))

        # Sort by relevance
        scored_paragraphs.sort(reverse=True, key=lambda x: x[0])

        if not scored_paragraphs:
            return None

        # Return the top relevant paragraphs (max 3)
        top_paragraphs = scored_paragraphs[:3]

        result = "Here are the most relevant financial excerpts:\n\n"
        for score, paragraph in top_paragraphs:
            # Format financial numbers consistently
            formatted_paragraph = self._format_financial_numbers(paragraph)
            result += f"{formatted_paragraph.strip()}\n\n"

        return result

    def _format_financial_numbers(self, text: str) -> str:
        """Format financial numbers in text consistently."""
        formatted = text

        # Format currency values
        currency_pattern = r'([\$€£])\s*([\d\.,]+)(?:\s*(billion|million|b|m))?'

        def format_currency_match(match):
            symbol = match.group(1)
            number = match.group(2).replace(',', '')
            unit = match.group(3).lower() if match.group(3) else ''

            try:
                value = float(number)
                if unit in ['billion', 'b']:
                    return f"{symbol}{value:.2f}B"
                elif unit in ['million', 'm']:
                    return f"{symbol}{value:.2f}M"
                elif value >= 1000000000:
                    return f"{symbol}{value / 1000000000:.2f}B"
                elif value >= 1000000:
                    return f"{symbol}{value / 1000000:.2f}M"
                else:
                    return f"{symbol}{value:.2f}"
            except ValueError:
                return match.group(0)

        formatted = re.sub(currency_pattern, format_currency_match, formatted, flags=re.IGNORECASE)

        # Format percentage values
        pct_pattern = r'([\d\.,]+)\s*%'

        def format_pct_match(match):
            number = match.group(1).replace(',', '')
            try:
                value = float(number)
                return f"{value:.2f}%"
            except ValueError:
                return match.group(0)

        formatted = re.sub(pct_pattern, format_pct_match, formatted)

        return formatted

    def _comprehensive_local_search(self, query: str) -> Optional[str]:
        """
        Comprehensive search through all local data when other methods fail.
        This combines metrics, structured data, tables, and text search.
        """
        results = []

        # Try direct metric lookup
        metric_result = self._direct_metric_lookup(query)
        if metric_result:
            results.append(f"From direct metrics extraction:\n{metric_result}")

        # Try structured data lookup
        structured_result = self._structured_data_lookup(query)
        if structured_result:
            results.append(f"From structured financial data:\n{structured_result}")

        # Try table search
        table_result = self._financial_table_search(query)
        if table_result:
            results.append(f"From financial tables:\n{table_result}")

        # Try direct text search
        text_result = self._financial_text_search(query)
        if text_result:
            results.append(f"From financial text analysis:\n{text_result}")

        # If we have results, combine them
        if results:
            combined_result = "Found financial information from multiple sources:\n\n"
            combined_result += "\n\n".join(results)
            return combined_result

        return None

    def _create_fallback_response(self, query: str) -> str:
        """Create a fallback response with all available financial information."""
        sections = []

        # Check if we have any metrics to share
        metrics = self.pdf_extracted_data.get("metrics", {})
        if metrics:
            sections.append("### Financial Metrics")
            metrics_list = []
            for key, value in metrics.items():
                if value:  # Only include non-empty values
                    formatted_key = key.replace("_", " ").title()
                    # Format values consistently
                    formatted_value = self._format_financial_value(value, key)
                    metrics_list.append(f"- {formatted_key}: {formatted_value}")
            sections.append("\n".join(metrics_list))

        # Check if we have structured data to share
        structured_data = self.pdf_extracted_data.get("structured_data", {})
        if structured_data:
            for section_name, section_data in structured_data.items():
                if section_data:
                    sections.append(f"### {section_name.replace('_', ' ').title()}")
                    section_items = []
                    for key, value in section_data.items():
                        if isinstance(value, dict):
                            section_items.append(f"- {key.replace('_', ' ').title()}:")
                            for subkey, subvalue in value.items():
                                formatted_value = self._format_financial_value(subvalue, key) if isinstance(subvalue,
                                                                                                            str) else subvalue
                                section_items.append(f"  * {subkey.title()}: {formatted_value}")
                        else:
                            formatted_value = self._format_financial_value(value, key) if isinstance(value,
                                                                                                     str) else value
                            section_items.append(f"- {key.replace('_', ' ').title()}: {formatted_value}")
                    sections.append("\n".join(section_items))

        # Check if we have any tables to mention
        tables = self.pdf_extracted_data.get("tables", [])
        if tables:
            sections.append(
                f"### Financial Tables\nI found {len(tables)} tables in the documents that might contain relevant financial information.")

        # Create the response
        if sections:
            return (f"I couldn't find specific information about '{query}' in the financial documents, "
                    f"but here's what I know:\n\n" + "\n\n".join(sections))
        else:
            return (f"I couldn't find information about '{query}' in the financial documents. "
                    f"The search didn't return any relevant financial data. Please try rephrasing your query "
                    f"or ask about a different aspect of the financial reports.")

    def _ensure_consistent_number_formats(self, text: str) -> str:
        """Ensure all financial numbers in the response have consistent formatting."""
        if not text:
            return text

        # Format currency values consistently
        formatted = self._format_financial_numbers(text)

        return formatted