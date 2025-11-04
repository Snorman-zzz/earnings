"""
Market Data Tools using yfinance.
Wraps market data fetching functions as LangChain tools.
"""

import yfinance as yf
from langchain_core.tools import tool
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@tool
def fetch_street_estimates(ticker: str) -> Dict[str, any]:
    """
    Fetch Wall Street analyst estimates for a given stock ticker.

    Args:
        ticker: Stock ticker symbol (e.g., "NVDA", "AAPL")

    Returns:
        Dictionary containing:
        - eps_estimate: Consensus EPS estimate
        - revenue_estimate: Consensus revenue estimate (in billions)
        - analyst_count: Number of analysts covering the stock
    """
    logger.info(f"Fetching street estimates for {ticker}")

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        estimates = {
            "ticker": ticker,
            "eps_estimate": info.get("epsCurrentYear", "N/A"),
            "revenue_estimate": info.get("revenueEstimate", "N/A"),
            "analyst_count": info.get("numberOfAnalystOpinions", "N/A"),
        }

        logger.info(f"Street estimates: {estimates}")
        return estimates

    except Exception as e:
        logger.error(f"Error fetching estimates for {ticker}: {e}")
        return {
            "ticker": ticker,
            "eps_estimate": "N/A",
            "revenue_estimate": "N/A",
            "analyst_count": "N/A",
            "error": str(e),
        }


@tool
def fetch_stock_price(ticker: str, period: str = "1d") -> Dict[str, any]:
    """
    Get current or recent stock price for a ticker.

    Args:
        ticker: Stock ticker symbol (e.g., "NVDA", "AAPL")
        period: Time period for historical data (default "1d" for latest)

    Returns:
        Dictionary containing:
        - current_price: Most recent closing price
        - currency: Currency of the price
        - market_cap: Market capitalization
    """
    logger.info(f"Fetching stock price for {ticker}")

    try:
        stock = yf.Ticker(ticker)

        # Get latest price from history
        hist = stock.history(period=period)
        if hist.empty:
            current_price = 0.0
        else:
            current_price = float(hist["Close"].iloc[-1])

        # Get additional info
        info = stock.info

        price_data = {
            "ticker": ticker,
            "current_price": current_price,
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap", "N/A"),
        }

        logger.info(f"Stock price data: {price_data}")
        return price_data

    except Exception as e:
        logger.error(f"Error fetching price for {ticker}: {e}")
        return {
            "ticker": ticker,
            "current_price": 0.0,
            "currency": "USD",
            "market_cap": "N/A",
            "error": str(e),
        }


@tool
def fetch_historical_performance(ticker: str, period: str = "1y") -> Dict[str, any]:
    """
    Fetch historical stock performance metrics.

    Args:
        ticker: Stock ticker symbol
        period: Historical period (e.g., "1mo", "3mo", "1y")

    Returns:
        Dictionary with performance metrics
    """
    logger.info(f"Fetching historical performance for {ticker} ({period})")

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)

        if hist.empty:
            return {"ticker": ticker, "error": "No historical data available"}

        # Calculate simple metrics
        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        return_pct = ((end_price - start_price) / start_price) * 100

        performance = {
            "ticker": ticker,
            "period": period,
            "start_price": start_price,
            "end_price": end_price,
            "return_pct": round(return_pct, 2),
            "high": float(hist["High"].max()),
            "low": float(hist["Low"].min()),
        }

        logger.info(f"Historical performance: {performance}")
        return performance

    except Exception as e:
        logger.error(f"Error fetching historical data for {ticker}: {e}")
        return {"ticker": ticker, "period": period, "error": str(e)}
