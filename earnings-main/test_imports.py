"""
Test script to verify all imports and basic functionality.
"""
import sys
from pathlib import Path

print("=" * 80)
print("Testing FinSight AI RAG System Imports")
print("=" * 80)
print()

# Test 1: Basic imports
print("1. Testing basic imports...")
try:
    import os
    import logging
    from dotenv import load_dotenv
    print("✅ Basic imports successful")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")
    sys.exit(1)

# Test 2: Load environment
print("\n2. Testing environment setup...")
try:
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✅ OPENAI_API_KEY found (length: {len(openai_key)})")
    else:
        print("⚠️  OPENAI_API_KEY not found in environment")
except Exception as e:
    print(f"❌ Environment setup failed: {e}")

# Test 3: Config imports
print("\n3. Testing config imports...")
try:
    from config.settings import settings
    from config.constants import FINANCIAL_METRICS, UNIT_MULTIPLIERS
    print(f"✅ Config imports successful")
    print(f"   - GPT5 Model: {settings.GPT5_MODEL}")
    print(f"   - Embedding Model: {settings.EMBEDDING_MODEL}")
    print(f"   - Chunk Size: {settings.CHUNK_SIZE}")
    print(f"   - Financial Metrics: {len(FINANCIAL_METRICS)} defined")
except Exception as e:
    print(f"❌ Config imports failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Document processor imports
print("\n4. Testing document processor imports...")
try:
    from document_processor.financial_document_processor import FinancialDocumentProcessor
    print("✅ Document processor imports successful")
except Exception as e:
    print(f"❌ Document processor imports failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Retriever imports
print("\n5. Testing retriever imports...")
try:
    from retriever.financial_retriever_builder import FinancialRetrieverBuilder
    print("✅ Retriever imports successful")
except Exception as e:
    print(f"❌ Retriever imports failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Tools imports
print("\n6. Testing tools imports...")
try:
    from tools.market_data_tools import fetch_street_estimates, fetch_stock_price
    from tools.calculation_tools import calculate_surprise_percentage, calculate_yoy_growth
    print("✅ Tools imports successful")
except Exception as e:
    print(f"❌ Tools imports failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Agents imports
print("\n7. Testing agents imports...")
try:
    from agents.financial_research_agent import FinancialResearchAgent
    from agents.financial_verification_agent import FinancialVerificationAgent
    from agents.financial_workflow import FinancialWorkflow
    print("✅ Agents imports successful")
except Exception as e:
    print(f"❌ Agents imports failed: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Component instantiation
print("\n8. Testing component instantiation...")
try:
    # This will fail without OPENAI_API_KEY, but we can test the structure
    if os.getenv("OPENAI_API_KEY"):
        processor = FinancialDocumentProcessor()
        print("✅ FinancialDocumentProcessor instantiated")

        retriever_builder = FinancialRetrieverBuilder()
        print("✅ FinancialRetrieverBuilder instantiated")

        research_agent = FinancialResearchAgent()
        print("✅ FinancialResearchAgent instantiated")

        verification_agent = FinancialVerificationAgent()
        print("✅ FinancialVerificationAgent instantiated")

        workflow = FinancialWorkflow()
        print("✅ FinancialWorkflow instantiated")
    else:
        print("⚠️  Skipping instantiation (no OPENAI_API_KEY)")
except Exception as e:
    print(f"❌ Component instantiation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Import Testing Complete")
print("=" * 80)
