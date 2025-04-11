"""Utility & helper functions."""

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
import os
import logging

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    try:
        # Parse provider and model
        parts = fully_specified_name.split("/")
        if len(parts) < 2:
            logger.warning(f"Invalid model specifier: {fully_specified_name}. Using default model.")
            provider = "openai"
            model = "gpt-4"
        else:
            provider, model = parts[0], "/".join(parts[1:])

        logger.info(f"Loading model: {provider}/{model}")

        # Select appropriate model based on provider
        if provider.lower() == "anthropic":
            # Use Anthropic model
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not found in environment. Using OpenAI fallback.")
                return ChatOpenAI(model="gpt-4")

            logger.info(f"Using Anthropic model: {model}")
            return ChatAnthropic(model=model, anthropic_api_key=api_key)

        elif provider.lower() == "openai":
            # Use OpenAI model
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY not found in environment.")

            logger.info(f"Using OpenAI model: {model}")
            return ChatOpenAI(model=model, openai_api_key=api_key)

        else:
            # Use OpenRouter to access other models
            api_key = os.getenv("OPENROUTER_API_KEY")
            api_base = os.getenv("OPENROUTER_BASE_URL")

            if not api_key or not api_base:
                logger.warning("OPENROUTER credentials not found. Using OpenAI fallback.")
                return ChatOpenAI(model="gpt-4")

            logger.info(f"Using OpenRouter for model: {fully_specified_name}")
            return ChatOpenAI(
                openai_api_key=api_key,
                openai_api_base=api_base,
                model_name=fully_specified_name,
            )

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}. Using fallback model.")
        # Fallback to OpenAI
        return ChatOpenAI(model="gpt-4")