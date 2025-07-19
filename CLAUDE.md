# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 【PROJECT OVERVIEW】Kaggle S5E7 Personality Prediction
- **Competition**: https://www.kaggle.com/competitions/playground-series-s5e7/overview
- **Problem**: Binary classification (Introvert vs Extrovert)
- **Metric**: Accuracy
- **Current Ranking**: 1182/2749 teams (43.0% percentile)
- **Best Score**: 0.974898
- **Bronze Target**: 0.976518 (+0.00162 improvement needed)

## 【CRITICAL - CURRENT PROJECT STATE】
### Project Has Been Reset
- **Previous Implementation Removed**: Over-engineered 76 files/15,000 lines eliminated
- **Clean Slate**: Only infrastructure files remain (Makefile, pyproject.toml, docs/)
- **No Active Code**: src/, scripts/, tests/ directories are empty and need implementation
- **Data Ready**: DuckDB with competition data is prepared at `/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb`

### Previous Problems (Now Resolved by Reset)
- Over-complexity, data leakage, factory pattern abuse, architectural issues

## 【IMPLEMENTATION PLAN】Simple & Effective Approach
```
Target Structure (To Be Built):
├── src/
│   ├── data.py          # DuckDB data loading
│   ├── features.py      # Feature engineering
│   ├── models.py        # LightGBM, XGBoost, CatBoost
│   ├── validation.py    # Cross-validation
│   ├── ensemble.py      # Model combination
│   └── submission.py    # Submission generation
├── scripts/
│   └── workflow.py      # Main pipeline
└── tests/
    └── test_*.py        # Unit tests
```

### Implementation Strategy
1. **Phase 1**: Simple LightGBM baseline (target: 0.975+)
2. **Phase 2**: Add XGBoost/CatBoost + feature engineering
3. **Phase 3**: Ensemble optimization (target: 0.976518+ for bronze)

## 【DATA MANAGEMENT】DuckDB Ready
- **Database Path**: `/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb`
- **Schema**: `playground_series_s5e7`
- **Tables**: `train`, `test`, `sample_submission`
- **Target Column**: `Personality` (Introvert/Extrovert)
- **ID Column**: `id`
- **Features**: 7 total (5 numeric + 2 categorical)

### Feature Overview
- **Numeric Features**: Time_spent_Alone, Social_event_attendance, Going_outside, Friends_circle_size, Post_frequency
- **Categorical Features**: Stage_fear (Yes/No), Drained_after_socializing (Yes/No)

### Data Access Pattern
```python
import duckdb
conn = duckdb.connect('/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb')
train = conn.execute("SELECT * FROM playground_series_s5e7.train").df()
test = conn.execute("SELECT * FROM playground_series_s5e7.test").df()
```

## 【DEVELOPMENT COMMANDS】

### Currently Available (Makefile)
```bash
make install              # Install dependencies
make dev-install         # Install with dev tools
make setup               # Create directory structure
make quick-test          # Quick single model test
make personality-prediction  # Full workflow (when implemented)
make test                # Run tests (when tests exist)
make clean               # Clean outputs
make help                # Show available commands
```

### Target Commands (After Implementation)
```bash
# Development workflow
make data-explore        # Initial data exploration
make baseline           # Run LightGBM baseline
make models             # Train all models (LGB, XGB, CatBoost)
make ensemble           # Create ensemble predictions
make submit             # Generate submission file

# Individual model tests
make model-lgb          # LightGBM only
make model-xgb          # XGBoost only
make model-cat          # CatBoost only
```

## 【DEPENDENCIES & ENVIRONMENT】

### Installation (pyproject.toml configured)
```bash
pip install -e .                    # Basic ML dependencies
pip install -e .[dev]              # + development tools
pip install -e .[optimization]     # + Optuna for tuning
pip install -e .[visualization]    # + plotting libraries
```

### Core Dependencies
- **Data**: pandas, numpy, duckdb
- **Models**: scikit-learn, xgboost, lightgbm, catboost
- **Optimization**: optuna
- **Development**: pytest, black, flake8, mypy
- **Python**: 3.8+

## 【PROVEN BENCHMARKS】Previous Performance
- **LightGBM**: 96.90% (±0.24%) ← Best single model
- **XGBoost**: 96.86% (±0.23%)
- **Random Forest**: 96.77% (±0.21%)
- **Prediction Distribution**: Extrovert 74.7%, Introvert 25.3%

## 【IMPLEMENTATION GUIDELINES】

### Design Principles
- **Keep It Simple**: Single-file modules, no over-engineering
- **Leak Prevention**: Proper CV-aware preprocessing
- **Trust Your CV**: StratifiedKFold validation
- **Data-Driven**: Focus on effective features only

### Key Implementation Notes
- **No CSV Files**: All data access through DuckDB only
- **System Python**: No virtual environments (per project history)
- **Classification Setup**: Binary classification for `Personality` target
- **Accuracy Metric**: Primary evaluation criterion

### Development Workflow
1. **Start Simple**: Basic LightGBM with minimal features
2. **Iterate Fast**: Small improvements, frequent validation
3. **Add Complexity Gradually**: Only when proven beneficial
4. **Test Everything**: Unit tests for all components
5. **Trust CV**: Don't overfit to public leaderboard

## 【SUCCESS CRITERIA】
- **Performance**: 0.976518+ accuracy (bronze medal threshold)
- **Code Quality**: Clean, simple, maintainable implementation
- **Reliability**: Reproducible results with proper CV validation
- **Efficiency**: Fast development cycles, quick iterations
