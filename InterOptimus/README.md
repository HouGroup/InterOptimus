# InterOptimus - Crystal Interface Optimization

An efficient python package for Interface Simulation with LLM-powered intelligent agents.

## 🚀 Core Features
1. Visualizing lattice matching information by polar projection figure;
2. Symmetry analysis to screen out identical matching and termination conditions;
3. Structure pre-optimization by MLIP-predicted interface energy.

## 🤖 LLM Agents (New!)

InterOptimus now includes powerful LLM-powered agents that can understand natural language descriptions and automatically generate optimized crystal interfaces:

### Core Agents
- **`interface_agent.py`** - Basic rule-based agent for interface generation
- **`llm_interface_agent.py`** - OpenAI GPT-powered intelligent agent
- **`advanced_agent.py`** - Advanced agent with comprehensive analysis

### Materials Project Integration
- **`mp_interface_agent.py`** - MP-integrated agent for automatic CIF download
- **`mp_interface_agent_fixed.py`** - Fixed version with error handling

### Specialized Agents
- **`llm_iomaker_job.py`** - LLM IOMaker with LOCAL/REMOTE config
- **`remote_submit.py`** - Remote submission helper

### Documentation & Examples
- **`LLM_AGENT_README.md`** - LLM agent usage guide
- **`USAGE_GUIDE.md`** - Complete usage guide
- **`demo_*.py`** - Various demonstration scripts
- **`test_*.py`** - Test suites for all components

### Requirements
- **`requirements_llm.txt`** - Dependencies for LLM agents

Install
`pip install .`
