"""
Financial Analysis Workflow using LangGraph.
Orchestrates document processing, metric extraction, verification, and report generation.
"""

import logging
from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, END

from .financial_research_agent import FinancialResearchAgent
from .financial_verification_agent import FinancialVerificationAgent
from tools.market_data_tools import fetch_street_estimates, fetch_stock_price
from tools.calculation_tools import calculate_surprise_percentage, calculate_yoy_growth

logger = logging.getLogger(__name__)


class FinancialAgentState(TypedDict):
    """State object passed between workflow nodes."""
    ticker: str
    company_name: str
    retriever: any  # EnsembleRetriever
    extracted_metrics: Dict
    analyst_estimates: Dict
    verification_report: Dict
    final_report: str
    needs_reextraction: bool


class FinancialWorkflow:
    """
    Workflow orchestrator for financial earnings analysis.

    Flow:
    1. Extract metrics from documents (ResearchAgent)
    2. Fetch analyst estimates (market data tools)
    3. Verify extracted metrics (VerificationAgent)
    4. Re-extract if verification fails (conditional)
    5. Generate final earnings report
    """

    def __init__(self):
        """Initialize workflow with agents."""
        logger.info("Initializing FinancialWorkflow...")

        self.research_agent = FinancialResearchAgent()
        self.verification_agent = FinancialVerificationAgent()

        # Build and compile workflow
        self.workflow = self._build_workflow()

        logger.info("FinancialWorkflow initialized successfully")

    def _build_workflow(self) -> any:
        """Build the LangGraph workflow."""
        workflow = StateGraph(FinancialAgentState)

        # Add nodes
        workflow.add_node("extract_metrics", self._extract_metrics_step)
        workflow.add_node("fetch_estimates", self._fetch_estimates_step)
        workflow.add_node("verify_metrics", self._verify_metrics_step)
        workflow.add_node("generate_report", self._generate_report_step)

        # Define flow
        workflow.set_entry_point("extract_metrics")
        workflow.add_edge("extract_metrics", "fetch_estimates")
        workflow.add_edge("fetch_estimates", "verify_metrics")

        # Conditional: re-extract if verification fails, otherwise generate report
        workflow.add_conditional_edges(
            "verify_metrics",
            self._decide_after_verification,
            {
                "reextract": "extract_metrics",
                "generate": "generate_report",
            },
        )

        workflow.add_edge("generate_report", END)

        return workflow.compile()

    def _extract_metrics_step(self, state: FinancialAgentState) -> Dict:
        """Step 1: Extract financial metrics from documents."""
        logger.info("=" * 80)
        logger.info("STEP 1: Extracting Financial Metrics")
        logger.info("=" * 80)

        extracted = self.research_agent.extract_all_metrics(
            retriever=state["retriever"],
            ticker=state["ticker"],
            company=state["company_name"],
        )

        return {"extracted_metrics": extracted}

    def _fetch_estimates_step(self, state: FinancialAgentState) -> Dict:
        """Step 2: Fetch analyst estimates from market data."""
        logger.info("=" * 80)
        logger.info("STEP 2: Fetching Analyst Estimates")
        logger.info("=" * 80)

        ticker = state["ticker"]

        # Fetch estimates
        estimates = fetch_street_estimates.invoke({"ticker": ticker})
        price_data = fetch_stock_price.invoke({"ticker": ticker})

        analyst_data = {
            **estimates,
            "current_price": price_data.get("current_price", 0.0),
            "market_cap": price_data.get("market_cap", "N/A"),
        }

        logger.info(f"Analyst estimates fetched: {analyst_data}")

        return {"analyst_estimates": analyst_data}

    def _verify_metrics_step(self, state: FinancialAgentState) -> Dict:
        """Step 3: Verify extracted metrics against source documents."""
        logger.info("=" * 80)
        logger.info("STEP 3: Verifying Extracted Metrics")
        logger.info("=" * 80)

        verification = self.verification_agent.verify_metrics(
            extracted_metrics=state["extracted_metrics"],
            retriever=state["retriever"],
        )

        # Determine if re-extraction is needed
        needs_reextraction = (
            not verification["overall_verified"]
            and len(verification.get("corrections", {})) > 0
        )

        return {
            "verification_report": verification,
            "needs_reextraction": needs_reextraction,
        }

    def _decide_after_verification(self, state: FinancialAgentState) -> str:
        """Decision point: re-extract or proceed to report generation."""
        if state.get("needs_reextraction", False):
            logger.warning("Verification failed with corrections - re-extracting metrics")
            return "reextract"
        else:
            logger.info("Verification passed - proceeding to report generation")
            return "generate"

    def _generate_report_step(self, state: FinancialAgentState) -> Dict:
        """Step 4: Generate final earnings report with tables and analysis."""
        logger.info("=" * 80)
        logger.info("STEP 4: Generating Final Earnings Report")
        logger.info("=" * 80)

        # Extract key metrics
        metrics = state["extracted_metrics"].get("metrics", {})
        estimates = state["analyst_estimates"]

        # Generate report sections
        report_sections = []

        # Header
        report_sections.append(f"# {state['company_name']} ({state['ticker']}) Earnings Report")
        report_sections.append("")
        report_sections.append("---")
        report_sections.append("")

        # Earnings Call Table
        report_sections.append("## Earnings Call Summary")
        report_sections.append("")
        report_sections.append(self._generate_earnings_table(metrics, estimates))
        report_sections.append("")

        # Financial Metrics Table
        report_sections.append("## Key Financial Metrics")
        report_sections.append("")
        report_sections.append(self._generate_financials_table(metrics))
        report_sections.append("")

        # Price Prediction (if data available)
        if "current_price" in estimates:
            report_sections.append("## Price Impact Analysis")
            report_sections.append("")
            report_sections.append(self._generate_price_analysis(metrics, estimates))
            report_sections.append("")

        # Verification Summary
        verification = state["verification_report"]
        report_sections.append("## Data Verification")
        report_sections.append("")
        report_sections.append(
            f"**Verification Status:** {verification.get('verification_rate', 'N/A')}"
        )
        if verification.get("overall_verified", False):
            report_sections.append("✅ All metrics verified against source documents")
        else:
            report_sections.append("⚠️ Some metrics require manual review")
        report_sections.append("")

        final_report = "\n".join(report_sections)

        logger.info("Final report generated successfully")
        logger.info("=" * 80)

        return {"final_report": final_report}

    def _generate_earnings_table(self, metrics: Dict, estimates: Dict) -> str:
        """Generate earnings call summary table."""
        eps_data = metrics.get("eps", {})
        revenue_data = metrics.get("revenue", {})

        # Extract values (with fallbacks)
        eps_current = eps_data.get("current_quarter", {})
        eps_value = eps_current.get("value", "N/A")
        eps_estimate = estimates.get("eps_estimate", "N/A")

        revenue_current = revenue_data.get("current_quarter", {})
        revenue_value = revenue_current.get("value", "N/A")
        revenue_unit = revenue_current.get("unit", "M")
        revenue_estimate = estimates.get("revenue_estimate", "N/A")

        # Calculate surprises (if data available)
        eps_surprise = "N/A"
        if isinstance(eps_value, (int, float)) and isinstance(eps_estimate, (int, float)):
            eps_surprise = f"{calculate_surprise_percentage.invoke({'reported': eps_value, 'expected': eps_estimate})}%"

        table = f"""
| Metric | Reported | Estimated | Surprise |
|--------|----------|-----------|----------|
| EPS | ${eps_value} | ${eps_estimate} | {eps_surprise} |
| Revenue | ${revenue_value}{revenue_unit} | ${revenue_estimate}B | N/A |
"""
        return table

    def _generate_financials_table(self, metrics: Dict) -> str:
        """Generate detailed financials table."""
        rows = []
        rows.append("| Metric | Current Quarter | Prior Year | YoY Growth |")
        rows.append("|--------|----------------|------------|------------|")

        for metric_name in ["eps", "revenue", "operating_income", "net_income"]:
            metric_data = metrics.get(metric_name, {})
            current = metric_data.get("current_quarter", {})
            prior = metric_data.get("prior_year", {})

            current_val = current.get("value", "N/A")
            current_unit = current.get("unit", "")
            prior_val = prior.get("value", "N/A")

            # Calculate YoY if both values available
            yoy = "N/A"
            if isinstance(current_val, (int, float)) and isinstance(prior_val, (int, float)):
                yoy = f"{calculate_yoy_growth.invoke({'current': current_val, 'prior': prior_val})}%"

            rows.append(
                f"| {metric_name.upper()} | ${current_val}{current_unit} | ${prior_val}{current_unit} | {yoy} |"
            )

        return "\n".join(rows)

    def _generate_price_analysis(self, metrics: Dict, estimates: Dict) -> str:
        """Generate price impact analysis section."""
        current_price = estimates.get("current_price", 0.0)

        analysis = f"""
**Current Stock Price:** ${current_price:.2f}

**Price Impact Factors:**
- Earnings surprise magnitude
- Revenue growth trajectory
- Forward guidance sentiment

*Note: Price prediction requires historical correlation analysis with earnings surprises.*
"""
        return analysis

    def run_analysis(
        self, ticker: str, company_name: str, retriever
    ) -> Dict:
        """
        Run the complete financial analysis workflow.

        Args:
            ticker: Stock ticker symbol
            company_name: Company name
            retriever: Document retriever (built from processed PDFs)

        Returns:
            Dictionary with final_report and verification_report
        """
        logger.info("=" * 80)
        logger.info(f"Starting Financial Analysis Workflow for {company_name} ({ticker})")
        logger.info("=" * 80)

        # Initialize state
        initial_state = FinancialAgentState(
            ticker=ticker,
            company_name=company_name,
            retriever=retriever,
            extracted_metrics={},
            analyst_estimates={},
            verification_report={},
            final_report="",
            needs_reextraction=False,
        )

        # Run workflow
        try:
            final_state = self.workflow.invoke(initial_state)

            logger.info("Workflow completed successfully")
            logger.info("=" * 80)

            return {
                "final_report": final_state["final_report"],
                "verification_report": self.verification_agent.format_verification_report(
                    final_state["verification_report"]
                ),
                "extracted_metrics": final_state["extracted_metrics"],
                "analyst_estimates": final_state["analyst_estimates"],
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
