import os
from dotenv import load_dotenv
try:
    from llama_index.llms.openrouter import OpenRouter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import Settings
    import nest_asyncio
    LLAMA_INDEX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LlamaIndex dependencies not available: {e}")
    LLAMA_INDEX_AVAILABLE = False

# Load environment variables
load_dotenv()

class RAGConfig:
    """Configuration class for RAG pipeline setup."""
    
    def __init__(self):
        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("LlamaIndex dependencies are not installed. Please run: pip install -r requirements.txt")
            
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        # Apply nest_asyncio for Jupyter compatibility
        nest_asyncio.apply()
        
        # Model configurations
        self.llm_model = "mistralai/mistral-7b-instruct"
        self.judge_llm_model = "qwen/qwen-turbo"
        self.embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # RAG parameters
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.similarity_top_k = 3
        self.bm25_top_k = 2
        self.max_tokens = 2048
        self.temperature = 0
        self.context_window = 4096
        
        # Initialize models
        self._setup_models()
    
    def _setup_models(self):
        """Initialize and configure LLM and embedding models."""
        # Initialize OpenRouter LLMs
        self.llm = OpenRouter(
            api_key=self.openrouter_api_key,
            model=self.llm_model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            context_window=self.context_window
        )
        
        self.judge_llm = OpenRouter(
            api_key=self.openrouter_api_key,
            model=self.judge_llm_model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            context_window=self.context_window
        )
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)
        
        # Set global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
    
    def get_financial_analysis_prompt(self, company_name: str, ticker: str, market_data: dict = None) -> str:
        """Get the financial analysis prompt template."""
        
        market_context = ""
        if market_data:
            market_context = f"""
            Current market data:
            - Expected EPS: {market_data.get('eps', 'N/A')}
            - Expected Revenue: {market_data.get('revenue', 'N/A')}
            - Current Stock Price: {market_data.get('price', 'N/A')}

            Calculate the surprise percentage for EPS and Revenue as: ((Reported - Expected) / Expected) * 100%
            """
        
        return f"""
        Please analyze these earnings documents for {company_name} ({ticker}).
        Use the provided document context to extract precise financial information.
        {market_context}

        Extract the following information with precise numerical values:
        1. Reported EPS and Revenue for the current quarter
        2. Other key financial metrics (net income, operating income, gross margin, etc.)
        3. Year-over-Year (Y/Y) changes for all metrics
        4. Forward guidance for next quarter/year if available
        5. Any announced stock splits, dividends, or buybacks

        Format your response with EXACTLY these sections using HTML formatting:

        <h2>Earnings Summary</h2>

        <h3>Earnings Calls</h3>
        <table>
        <tr>
          <th>Metric</th>
          <th>Expected</th>
          <th>Reported</th>
          <th>Surprise</th>
        </tr>
        <tr>
          <td>EPS</td>
          <td>$X.XX</td>
          <td>$X.XX</td>
          <td>X.XX%</td>
        </tr>
        <tr>
          <td>Revenue</td>
          <td>$XX.XXB</td>
          <td>$XX.XXB</td>
          <td>X.XX%</td>
        </tr>
        </table>

        <h3>Financials</h3>
        <table>
        <tr>
          <th>Metric</th>
          <th>Current Quarter</th>
          <th>Previous Year</th>
          <th>Y/Y Change</th>
        </tr>
        <tr>
          <td>Revenue</td>
          <td>$XX.XXB</td>
          <td>$XX.XXB</td>
          <td>XX.XX%</td>
        </tr>
        <tr>
          <td>Net Income</td>
          <td>$X.XXB</td>
          <td>$X.XXB</td>
          <td>XX.XX%</td>
        </tr>
        <tr>
          <td>Diluted EPS</td>
          <td>$X.XX</td>
          <td>$X.XX</td>
          <td>XX.XX%</td>
        </tr>
        <tr>
          <td>Operating Income</td>
          <td>$X.XXB</td>
          <td>$X.XXB</td>
          <td>XX.XX%</td>
        </tr>
        <tr>
          <td>Gross Margin</td>
          <td>XX.X%</td>
          <td>XX.X%</td>
          <td>X.X pts</td>
        </tr>
        </table>

        <h3>Key Findings Summary</h3>
        <p>Write a concise summary of the key findings with proper sentences and spacing between words.</p>

        <h3>Price Prediction</h3>
        <p>Price Prediction = CurrentPrice × (1 + AdjustmentFactor) = NewPrice</p>

        # IMPORTANT UNIT NORMALIZATION INSTRUCTIONS:
        Before calculating any percentage changes or surprises, normalize units first:
        1. For values with different units (like "11.89B" vs "39.33M"), convert both to the same unit first
        2. Convert all values to the same unit (millions or billions) before calculating percentages
        3. For numbers with units (B for billions, M for millions), extract the number part and apply the scale:
           - 1B = 1000M (converting billions to millions)
           - 1M = 0.001B (converting millions to billions)

        # CRUCIAL FORMATTING REQUIREMENTS:
        1. Use proper spacing between all words and numbers (e.g., "39.33 billion" NOT "39.33billion")
        2. Insert spaces between each word in your text
        3. Format numbers consistently with proper units (e.g., "$13.51B" not "$13.51billion")
        4. Ensure proper spacing after each punctuation mark
        5. Maintain proper spacing between all words in sentences
        6. Use HTML tags for all formatting, not markdown
        """

# Global configuration instance
rag_config = RAGConfig()