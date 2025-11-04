# FinSight AI: Automated Earnings Analysis & Stock Price Prediction

FinSight AI is a comprehensive tool for automating the analysis of corporate earnings reports and predicting post-earnings stock price movements. It combines Large Language Models (LLMs) and traditional financial analysis techniques to extract key information from earnings documents, analyze financial metrics, and generate insightful reports with price predictions.

## Project Overview

FinSight AI streamlines the financial analysis workflow by:

1. Analyzing earnings press releases and presentation slides uploaded as PDFs
2. Automatically extracting key financial metrics (EPS, revenue, margins, etc.)
3. Comparing reported figures to Wall Street expectations
4. Calculating year-over-year growth rates and surprise percentages
5. Generating comprehensive analysis reports with price predictions

## System Architecture

The system is built on a modular architecture with three main components:

### 1. PDF Analysis Module (`claude_pdf_analyzer.py`)

The PDF analyzer uses Anthropic's Claude API to extract financial information from earnings documents. It:
- Processes PDF files using Claude's document understanding capabilities
- Extracts precise numerical data from complex financial documents
- Formats the data into structured tables and analysis

### 2. Financial Data Integration (`financial_agents.py`)

The financial agents framework:
- Fetches real-time market data and analyst estimates using yfinance
- Provides tools for retrieving stock prices and consensus expectations
- Implements a custom agent architecture for financial reasoning
- Structures data into a coherent workflow

### 3. Web Application (`app.py`)

The streamlit-based web application:
- Provides an intuitive interface for document uploads
- Displays financial analysis in a well-formatted layout
- Shows earnings tables, financial metrics, and price predictions
- Handles user inputs and validation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/finsight-ai.git
cd finsight-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with:
```
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key  # If using OpenAI models
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Upload documents:
   - Earnings press release (PDF)
   - Earnings presentation slides (PDF)

3. Enter company information:
   - Company name
   - Stock ticker symbol

4. Click "Generate Comprehensive Analysis" to process the documents

5. Review the analysis:
   - Earnings Calls table (Expected vs. Reported)
   - Financials table (Current Quarter vs. Previous Year)
   - Analysis summary
   - Post-earnings price prediction

## Code Structure

```
finsight-ai/
├── claude_pdf_analyzer.py  # PDF analysis using Claude API
├── financial_agents.py     # Agent framework and financial tools
├── app.py                  # Streamlit web application
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Key Features

- **Multi-model analysis**: Compares LLM-based approaches with traditional methods
- **Precise numerical extraction**: High accuracy in extracting financial metrics
- **Unit normalization**: Automatically handles different numerical formats (B vs M)
- **Structured output**: Well-formatted tables and LaTeX formulas for price predictions
- **Web-based interface**: Easy-to-use interface for document uploads and analysis

## Limitations and Future Work

- Currently optimized for quarterly earnings reports in standard formats
- Price predictions are based on earnings surprise and market data, not comprehensive market models
- Future improvements planned:
  - Support for more document types (annual reports, investor presentations)
  - Enhanced time-series analysis for better price predictions
  - Sector-specific analysis customizations
  - Expanded historical data integration

## Contributors

- Ziqi Shao: ML method development
- Zhixiao Wu: Refinement of two methods, dataset collection and evaluation
- Mingze Yuan: LLM method development

## License

This project was developed as part of the CS7180: Special Topics in Generative AI course at Northeastern University.

---

This AI-powered financial analysis tool is for informational purposes only and should not be considered financial advice.
