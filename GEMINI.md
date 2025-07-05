# Gemini Project Configuration

## About This Project

use conda venv genAI

This project is an implementation of a decoder-only Transformer model from scratch in Python. The primary goal is to generate coherent text, initially based on the WikiHow dataset.

<<<<<<< HEAD
## Commands

- **Run Training:** `python src/train.py --data data/processed/wikihow.txt --model_path models/`
- **Run Inference:** `python src/generate.py --model_path models/model.pt`
- **Run Tests:** `pytest`
- **Lint & Format:** `ruff check . && black .`

## Dependencies

Install dependencies using pip:
`pip install -r requirements.txt`

## Coding Style

- Follow the PEP 8 style guide.
- Use `black` for code formatting.
- All functions and classes should have docstrings.
- Use type hints for all function signatures.

## Key Files

- `src/train.py`: Main script for training the model.
- `src/model.py`: Contains the Transformer model architecture.
- `src/dataset.py`: Data loading and preprocessing logic.
=======
## Present status of the project

Trained model weight:  models/transformer.pt

## WIP

Prepare the model for huggingface-hub. The end user must be able to use the model ruchir99/transformer-101 with AutoModelForCausalLM.from_pretrained() function
>>>>>>> 06fb81e (restarting git)
