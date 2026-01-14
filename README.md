# InterOptimus

Crystal Interface Optimization Toolkit with LLM-powered Intelligent Agents

## 📁 Project Structure

```
InterOptimus/
├── core/                    # Core InterOptimus functionality
│   ├── itworker.py         # Interface worker (main logic)
│   ├── matching.py         # Lattice matching algorithms
│   ├── tool.py            # Utility functions
│   ├── mlip.py            # MLIP calculator interfaces
│   ├── CNID.py            # CNID calculations
│   ├── equi_term.py       # Equivalent termination analysis
│   └── jobflow.py         # Jobflow workflows
│
├── agents/                 # LLM-powered intelligent agents
│   ├── interface_agent.py          # Basic rule-based agent
│   ├── llm_interface_agent.py      # OpenAI GPT-powered agent
│   ├── advanced_agent.py           # Advanced analysis agent
│   ├── mp_interface_agent.py       # Materials Project integration
│   ├── mp_interface_agent_fixed.py # Fixed MP agent
│   └── llm_interface_agent_yuanbao.py # Tencent YuanBao agent
│
├── demos/                 # Demonstration scripts
│   ├── demo.py            # Basic demo
│   ├── demo_llm_agent.py  # LLM agent demo
│   ├── demo_mp_agent.py   # MP integration demo
│   ├── demo_yuanbao_agent.py # YuanBao demo
│   ├── demo_custom_api.py # Custom API demo
│   └── demo_mlip.py       # MLIP optimization demo
│
├── examples/              # Example usage
│   ├── example_usage.py   # Basic usage examples
│   └── example_structures.py # Structure examples
│
├── tests/                 # Test suites
│   ├── test_*.py          # Various test files
│   └── test_mp_fix_simple.py # MP fix verification
│
├── docs/                  # Documentation
│   ├── README.md          # Main README
│   ├── LLM_AGENT_README.md # LLM agent guide
│   ├── USAGE_GUIDE.md     # Complete usage guide
│   ├── AGENT_README.md    # Agent documentation
│   ├── YUANBAO_AGENT_README.md # YuanBao guide
│   ├── INTERFACE_TYPES.md # Interface type explanations
│   └── FIXES_SUMMARY.md   # Bug fixes summary
│
└── requirements/
    └── requirements_llm.txt # LLM agent dependencies
```

## 🚀 Quick Start

### Install Core Package
```bash
cd InterOptimus
pip install .
```

### Install LLM Agents (Optional)
```bash
pip install -r requirements_llm.txt
```

### Basic Usage
```python
from InterOptimus.itworker import InterfaceWorker
from pymatgen.core.structure import Structure

# Load structures
film = Structure.from_file("film.cif")
substrate = Structure.from_file("substrate.cif")

# Create interface worker
iw = InterfaceWorker(film, substrate)

# Perform lattice matching
iw.lattice_matching(max_area=150)

# Generate interfaces
iw.parse_interface_structure_params()
```

### LLM Agent Usage
```python
from InterOptimus.mp_interface_agent_fixed import MPInterfaceAgentFixed

agent = MPInterfaceAgentFixed(
    mp_api_key="your-mp-api-key",
    api_key="your-openai-key"
)

result = agent.generate_interface_from_text_mp(
    "Create silicon on aluminum oxide interface"
)
```

## 📖 Documentation

- **[Main README](InterOptimus/README.md)** - Core package documentation
- **[LLM Agent Guide](InterOptimus/LLM_AGENT_README.md)** - LLM agent usage
- **[Usage Guide](InterOptimus/USAGE_GUIDE.md)** - Complete usage guide

## 🤖 LLM Agents

InterOptimus includes several intelligent agents:

1. **Rule-based Agent** - Traditional keyword matching
2. **OpenAI GPT Agent** - GPT-3.5/4 powered understanding
3. **Materials Project Agent** - Automatic CIF download
4. **Tencent YuanBao Agent** - Chinese language support

## 🧪 Testing

Run tests:
```bash
cd InterOptimus
python -m pytest tests/  # If pytest installed
# Or run individual tests
python test_mp_fix_simple.py
```

## 📄 License

See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please see documentation for contribution guidelines.