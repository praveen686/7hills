# QuantKubera

Institutional-grade quantitative trading platform with a focus on Deep Learning and MLOps.

## Project Structure

This project follows a modular, production-ready structure:
- **`src/quantkubera`**: Main Python package containing all source code.
- **`config/`**: Configuration files managed by [Hydra](https://hydra.cc/).
- **`data/`**: Data storage, separated into `raw` (immutable), `processed`, and `external`.
- **`notebooks/`**: Jupyter notebooks for exploration and prototyping.
- **`external/`**: Third-party repositories (e.g., `trading-momentum-transformer`).
- **`tests/`**: Unit and integration tests.

## Getting Started

### Prerequisites
- Python 3.9+

### Installation

1.  Clone the repository (if not already done).
2.  Install dependencies:
    ```bash
    pip install .
    # For development dependencies:
    pip install .[dev]
    ```

## Usage

### Running Experiments
Experiments are configured using Hydra in `config/`. To run a training job (example):
```bash
python -m quantkubera.models.train
```

### Adding New Models
1.  Define the model architecture in `src/quantkubera/models/`.
2.  Add a configuration file in `config/model/`.
3.  Register the model in the training script.

## External References
-   **Trading Momentum Transformer**: Located in `external/trading-momentum-transformer/`. See [Deep Evaluation](docs/deep_evaluation.md) for details.
