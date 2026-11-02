# Wine Quality Predictor

Binary classification: predict whether a wine is **good** (quality ≥ 6) or **not good** from its physicochemical properties.

Dataset: [Wine Quality – UCI #186](https://archive.ics.uci.edu/dataset/186/wine+quality) — red + white, 6,497 wines, 11 features — CC BY 4.0.

> Cortez et al. (2009). *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 47(4), 547-553.

## Installation

```bash
git clone https://github.com/yoramaccoh1-cpu/wine-quality.git
cd wine-quality
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Run the tests

```bash
python -m pytest
```

## Train the model

```bash
wine-quality train
```

## Evaluate on the test set
```bash
wine-quality evaluate
```

## Predict for a new wine
```bash
wine-quality predict --input demo_wine.json
```

## Project structure

```
wine-quality/
├── src/wine_quality/
│   ├── data.py       # DatasetLoader
│   ├── features.py   # ColumnTransformer
│   ├── models.py     # Trainer
│   ├── evaluate.py   # metrics and plots
│   └── cli.py        # train / evaluate / predict
├── tests/
├── notebooks/
├── reports/
│   └── report.pdf
├── config.yaml
├── pyproject.toml
├── requirements.txt
├── demo_wine.json
├── AI_USAGE.md
└── README.md
```
