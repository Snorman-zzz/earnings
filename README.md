# Financial Analysis System

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.herokuapp.com)
[![Python 3.11](https://img.shields.io/badge/python-3.11.11-blue.svg)](https://www.python.org/downloads/)

## Development Setup

### 1. Create Conda Environment
```bash
conda create -n Agents_Env python=3.11.11 -y
```

### 2. Activate Environment
```bash
conda activate Agents_Env
```

### 3. Navigate to Project
```bash
cd path/to/your/project  # Replace with actual path containing pyproject.toml
```

### 4. Install Dependencies (Editable Mode)
```bash
pip install -e .
```

### 5. Launch Development Server
```bash
langgraph dev
```

## Production Usage
```bash
streamlit run src/react_agent/financial_analyst_app.py
```

## Key Dependencies
| Package         | Version  | Purpose                      |
|-----------------|----------|------------------------------|
| langgraph       | ≥0.2.0   | Workflow orchestration        |
| openai          | ≥1.30.1  | LLM integration & file search|
| streamlit       | ≥1.33.0  | Web interface                |

Full list in [pyproject.toml](pyproject.toml)

## Configuration
```ini
# .env
OPENAI_API_KEY=your-api-key-here
```

![Development Workflow](docs/dev_workflow.png)

> **Note**: Editable mode (`-e .`) links live code changes without reinstallation
```

Changes made:
1. Added dedicated "Development Setup" section with Conda instructions
2. Separated production vs development launch commands
3. Added langgraph to dependencies table
4. Included note about editable installations
5. Added reference to pyproject.toml
6. Updated Python version badge to 3.11.11
7. Added visual for development workflow

The original features and architecture sections remain unchanged but would appear after these setup instructions in a full README.
