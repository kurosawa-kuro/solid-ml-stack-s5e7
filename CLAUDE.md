# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 【PROJECT OVERVIEW】Kaggle S5E7 Personality Prediction
- **Competition**: https://www.kaggle.com/competitions/playground-series-s5e7/overview
- **Problem**: Binary classification (Introvert vs Extrovert)
- **Metric**: Accuracy
- **Current Ranking**: 1182/2749 teams (43.0% percentile)
- **Current Best**: 0.9684 (CV Score)
- **Bronze Target**: 0.976518 (+0.008 improvement needed)

## 【CRITICAL - CURRENT PROJECT STATE】
### Advanced Implementation Stage
- **Mature Implementation**: Professional ML pipeline with 700+ lines of production-ready code
- **Bronze Medal Target**: 0.8% improvement needed (Current: 0.9684, Target: 0.976518)
- **Robust Architecture**: Medallion data management, pipeline integration, comprehensive CV framework
- **High Test Coverage**: 73% coverage with 475 tests across 18 test files
- **Data Ready**: DuckDB with competition data prepared at `/home/wsl/dev/my-study/ml/solid-ml-stack-s5e7/data/kaggle_datasets.duckdb`

### Current Performance Status
- **Latest CV Score**: 0.9684 ± 0.0020 (96.84% accuracy)
- **Gap to Bronze**: +0.008 improvement needed
- **Architecture Quality**: Extensible design preventing future over-engineering

## 【CURRENT ARCHITECTURE】Medallion Design & Pipeline Integration
```
Implemented Structure:
├── src/
│   ├── data/
│   │   ├── bronze.py     # ✅ Raw data processing
│   │   ├── silver.py     # ✅ Feature engineering (30+ features)
│   │   └── gold.py       # ✅ ML-ready data pipeline
│   ├── models.py         # ✅ LightGBM with pipeline integration (696 lines)
│   ├── validation.py     # ✅ CV framework with leak prevention (316 lines)
│   └── util/
│       ├── time_tracker.py   # ✅ Development efficiency tracking
│       └── notifications.py  # ✅ Workflow notifications
├── scripts/
│   ├── train.py          # ✅ Basic training
│   ├── train_light.py    # ✅ Fast iteration (322 lines)
│   ├── train_enhanced.py # ✅ Advanced pipeline
│   └── train_heavy.py    # ✅ Full optimization
└── tests/                # ✅ 73% coverage, 475 tests
```

### Medallion Data Processing Layers

#### Bronze Layer (`src/data/bronze.py`) - Raw Data Processing
**Purpose**: Raw data ingestion and **competition-grade preprocessing** with leak prevention
**Core Functions**:
- `load_data()`: DuckDB direct data loading with dtype validation
- `validate_data_quality()`: Column type & value range validation (non-negative time, realistic limits)
- `advanced_missing_strategy()`: Multi-approach missing value handling (flags, imputation, model-based)
- `encode_categorical_robust()`: Yes/No normalization with order-aware mapping
- `create_missing_indicators()`: Generate missing flags for high-impact features
- `winsorize_outliers()`: IQR-based outlier clipping for numeric stability
- `create_bronze_tables()`: Creates bronze schema with preprocessing metadata

**Advanced Missing Value Strategy** (上位勢手法):
1. **Missing Flags** (Stage_fear ~10%, Going_outside ~8%): Binary indicators for missing data
2. **Cross-Feature Imputation**: Use high correlation (多重共線性) to predict missing values
3. **Model-Specific Handling**: Keep NaN for tree models, impute for linear models
4. **Fold-Safe Processing**: All statistics computed within CV folds only

**Data Quality Pipeline** (必須ステップ):
- **Type Validation**: Explicit dtype setting (int/float/bool/category)
- **Range Guards**: Time_spent_Alone ≤ 24hrs, non-negative behavioral metrics  
- **Missing Pattern Analysis**: Identify systematic vs random missing
- **Categorical Standardization**: Unified Yes/No mapping with case handling

**Leak Prevention Architecture**:
- **Stratified K-Fold**: Maintain Introvert/Extrovert ratio across folds
- **Within-Fold Statistics**: Imputation values, encodings computed per fold
- **Pipeline Integration**: sklearn-compatible transformers for cross-validation

**Key Features**:
- **Competition-Grade Processing**: Implements top-tier Kaggle preprocessing patterns
- **Missing Intelligence**: Leverages missing patterns as prediction signals
- **Model Compatibility**: Dual preprocessing paths (tree vs linear models)
- **Quality Assurance**: Comprehensive validation preventing data corruption
- **Fast Prototyping**: Sub-second processing maintained despite advanced features

#### Silver Layer (`src/data/silver.py`) - Feature Engineering
**Purpose**: Advanced feature engineering and transformation
**Core Functions**:
- `advanced_features()`: 15+ statistical and domain features
- `enhanced_interaction_features()`: Top feature interactions
- `polynomial_features()`: Degree-2 polynomial expansion
- `scaling_features()`: Feature standardization

**Key Features**:
- **30+ engineered features** including:
  - Domain features: `extrovert_score`, `introvert_score`
  - Statistical features: `total_activity`, `avg_activity`, `activity_std`
  - Interaction features: `extrovert_social_interaction`, `fear_drain_interaction`
  - Polynomial features: degree-2 combinations of top features
- **Feature importance ordering** for systematic experimentation
- **Robust pipeline** with error handling and validation

#### Gold Layer (`src/data/gold.py`) - ML-Ready Data
**Purpose**: Production-ready ML data preparation
**Core Functions**:
- `clean_and_validate_features()`: Data quality assurance
- `select_best_features()`: Statistical feature selection (F-test + MI)
- `prepare_model_data()`: Final ML data preparation
- `get_ml_ready_data()`: X/y split with optional scaling

**Key Features**:
- **Advanced data cleaning**: Outlier handling, infinite value processing
- **Intelligent feature selection**: Combined F-statistics and mutual information
- **ML pipeline integration**: sklearn-compatible data preparation
- **Production utilities**: Submission file creation, array extraction

### Current Development Strategy
1. **Bronze Optimization** (Active): Hyperparameter tuning, feature selection → 0.976518
2. **Silver Expansion** (Ready): XGBoost/CatBoost ensemble, advanced features
3. **Gold Evolution** (Prepared): Multi-competition reusability, production deployment

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

### Available Commands (Implemented)
```bash
# Core workflow
make install              # ✅ Install dependencies
make dev-install         # ✅ Install with dev tools
make test                # ✅ Run 475 tests (73% coverage)
make quick-test          # ✅ Fast model validation
make personality-prediction  # ✅ Full training pipeline
make clean               # ✅ Clean outputs

# Training variations
python scripts/train_light.py    # ✅ Fast iteration (0.5s)
python scripts/train.py          # ✅ Standard training
python scripts/train_enhanced.py # ✅ Advanced features
python scripts/train_heavy.py    # ✅ Full optimization
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

## 【CURRENT PERFORMANCE】Recent Training Results
- **Light Enhanced Model**: 96.79% ± 0.22% (Latest run, 30 features)
- **Baseline Model**: 96.84% ± 0.20% (Best CV score, 10 features) 
- **Training Efficiency**: 0.5 seconds (light), 0.39 seconds (baseline)
- **Feature Importance**: poly_extrovert_score_Post_frequency (257.6) leads
- **Bronze Gap**: +0.8% needed (highly achievable with optimization)

### Performance Analysis
- **Consistent Results**: Low standard deviation indicates stable model
- **Fast Iteration**: Sub-second training enables rapid experimentation
- **Feature Quality**: Polynomial features show strong predictive power

## 【IMPLEMENTATION GUIDELINES】

### Design Principles (Balanced Approach)
- **Extensible Simplicity**: Clean abstractions that support growth without complexity
- **Leak Prevention**: Pipeline integration ensures CV-aware preprocessing (implemented)
- **Trust Your CV**: StratifiedKFold with integrity validation (implemented)
- **Evidence-Based**: Feature engineering based on importance analysis
- **Incremental Development**: Medallion architecture supports progressive enhancement

### Key Implementation Notes
- **No CSV Files**: All data access through DuckDB only
- **System Python**: No virtual environments (per project history)
- **Classification Setup**: Binary classification for `Personality` target
- **Accuracy Metric**: Primary evaluation criterion

### Development Workflow (Optimized)
1. **Optimize Current**: Hyperparameter tuning and feature selection for bronze
2. **Iterate Fast**: 0.5-second training cycles enable rapid experimentation
3. **Validate Rigorously**: Data leak prevention and CV integrity checks (implemented)
4. **Test Comprehensively**: 73% coverage with integration tests (implemented)
5. **Scale Thoughtfully**: Medallion architecture supports controlled expansion

### Bronze Layer Implementation Checklist (上位勢パターン)
**必須ステップ (実装優先)**:
- [ ] データ読込時dtype明示設定 (int/float/bool/category)
- [ ] 値域バリデーション (Time_spent_Alone ≤ 24hrs, 非負チェック)
- [ ] Yes/No正規化辞書 (大文字小文字統一 → {0,1})
- [ ] 欠損フラグ生成 (Stage_fear, Going_outside, Drained_after_socializing)
- [ ] Fold内統計量計算 (CV内でimputation値・エンコーディング学習)
- [ ] Stratified K-Fold設定 (クラス比率維持)

**強推奨ステップ (性能向上)**:
- [ ] 外れ値Winsorizing (IQR基準, 1%/99%分位数クリップ)
- [ ] モデル別前処理パイプライン (Tree:NaN保持, Linear:補完)
- [ ] 交差特徴補完 (高相関利用した欠損値推定)
- [ ] カテゴリエンコーディング多様化 (頻度/ターゲット統計)

**実験ステップ (微差稼ぎ)**:
- [ ] 比率特徴 (Time_spent_Alone/(Time_spent_Alone+Social_event_attendance))
- [ ] RankGauss変換 (歪度大きい特徴の正規化)
- [ ] ターゲットエンコーディング+ノイズ (高カテゴリ時)

## 【SUCCESS CRITERIA】
- **Bronze Medal**: 0.976518+ accuracy (+0.8% from current 0.9684)
- **Architecture Quality**: Extensible design with controlled complexity
- **Reliability**: Data leak prevention, reproducible CV results (implemented)
- **Development Efficiency**: Sub-second training, comprehensive testing (implemented)
- **Long-term Value**: Reusable patterns for future competitions

## 【BRONZE MEDAL ROADMAP】
### Immediate Opportunities (1-2 weeks)
1. **Advanced Bronze Layer** (High Priority +0.3-0.5% expected):
   - Missing indicators for Stage_fear, Going_outside (上位勢実証済み)
   - Cross-feature imputation using high correlation patterns
   - Winsorizing outliers (IQR-based) for numeric stability
   - Model-specific preprocessing pipelines (tree vs linear)

2. **Hyperparameter Optimization**: Leverage existing Optuna integration (+0.2-0.4%)

3. **Enhanced Data Quality** (Medium Priority +0.1-0.3%):
   - Dtype validation with range guards (Time_spent_Alone ≤ 24hrs)
   - Categorical standardization (case-insensitive Yes/No mapping)
   - Missing pattern analysis for systematic vs random detection

4. **CV Framework Enhancement** (+0.1-0.2%):
   - Stratified K-Fold with explicit Introvert/Extrovert ratio maintenance
   - Fold-safe statistics computation preventing information leakage
   - Pipeline integration ensuring consistent train/validation processing

5. **Feature Selection**: Focus on top importance features (poly_extrovert_score_*)
6. **Model Ensemble**: Combine CV folds for prediction stability
7. **Threshold Tuning**: Optimize classification threshold for accuracy

### Technical Assets Ready
- ✅ **Data Leak Prevention**: Pipeline integration implemented
- ✅ **CV Framework**: Integrity validation and stratified sampling
- ✅ **Feature Engineering**: 30+ engineered features with importance ranking
- ✅ **Optimization Infrastructure**: Optuna integration for hyperparameter tuning
- ✅ **Performance Monitoring**: Time tracking and comprehensive logging
