# Project Context

## Purpose

This is a research project that implements Recurrent Neural Networks (RNNs) for sentiment analysis in PyTorch. The objective is to extend the paper "Sentiment Analysis of COVID-19 Tweets Using Machine Learning and BERT Models" (Informatics 2024, 11, 24) by implementing and evaluating RNN-based architectures on the SENT-COVID corpus.

**Research Goal:** Develop and evaluate RNN-based models for 3-class sentiment classification (positive, negative, neutral) of Mexican Spanish tweets, comparing performance against the baseline BERT models from the original paper.

**Target Paper:** papers/informatics-11-00024/ - "Sentiment Analysis of COVID-19 Tweets Using Machine Learning and BERT Models"

**Final Deliverables:**
- Trained model implementations in `models/`
- Final research report in `report/report/rnns-for-sentiment-analysis.tex`
- Supporting utilities in `utils/`

**Non-deliverables:**
- Notebooks in `notebooks/` are for experimentation only

## Tech Stack

**Core Technologies:**
- **Language:** Python 3.x
- **Deep Learning Framework:** PyTorch
- **Word Embeddings:** GloVe pre-trained on Spanish (SBW corpus)
- **Experiment Tracking:** TensorBoard
- **Data Processing:** pandas, scikit-learn
- **Visualization:** matplotlib
- **Development:** JupyterLab, IPython
- **Utilities:** d2l (Dive into Deep Learning library)
- **Code Formatting:** black

**Dependencies (from requirements.txt):**
```
d2l==1.0.3
torch
torchvision
tensorboard
jupyterlab
ipython
matplotlib
pandas
black
scikit-learn
```

## Project Conventions

### Code Style

- **Formatter:** black (configured in project)
- **Docstrings:** NumPy/SciPy style with references to D2L book sections where applicable
- **Type hints:** Encouraged but not strictly enforced
- **Module organization:** Separate concerns (models, utils, notebooks)
- **Line length:** Follow black defaults (88 characters)
- **Imports:** Standard library → third-party → local imports

### Architecture Patterns

**Project Structure:**
```
rnns-project/
├── models/              # Model definitions (DELIVERABLE)
│   ├── __init__.py
│   └── rnns.py         # RNN implementations (RNNScratch, RNNLMScratch)
├── utils/              # Training and data utilities (DELIVERABLE)
│   ├── __init__.py
│   ├── trainer.py      # Training loops, early stopping, TensorBoard logging
│   └── vocab.py        # Vocabulary management
├── notebooks/          # Experimental notebooks (NOT DELIVERABLE)
│   └── utils/          # Notebook-specific utilities
├── data/               # Datasets (SENT-COVID corpus, TASS2020 annotations)
├── papers/             # Reference papers
│   └── informatics-11-00024/  # Target paper to extend
├── report/             # LaTeX research report (DELIVERABLE)
│   └── report/
│       └── rnns-for-sentiment-analysis.tex
├── runs/               # TensorBoard logs and saved models
├── openspec/           # OpenSpec project management
└── requirements.txt    # Python dependencies
```

**Model Architecture (as per report - subject to change):**

1. **Embedding Layer**
   - Pre-trained GloVe word vectors (300 dimensions)
   - Trained on Spanish SBW (Spanish Billion Words) corpus
   - Fixed or fine-tunable embeddings

2. **Encoder (RNN)**
   - Bidirectional GRU (Gated Recurrent Units)
   - Multiple layers (depth configurable)
   - Hidden state dimension: configurable
   - Supports gradient clipping for stability

3. **Decoder (Classifier)**
   - Multilayer Perceptron (fully connected layers)
   - Output: 3 classes (positive, negative, neutral)

4. **Training Configuration**
   - Loss function: Cross-entropy loss
   - Optimizer: Adam (typical choice, configurable)
   - Metrics: Accuracy, weighted F1-score
   - Early stopping based on validation performance
   - Gradient clipping supported

**Design Patterns:**
- Models defined as `nn.Module` subclasses in `models/`
- Training utilities separated in `utils/trainer.py`
- Vocabulary management via `utils/vocab.py`
- Modular design for easy experimentation

### Testing Strategy

**Validation Approach:**
- Train/validation/test split for model evaluation
- Validation set used for hyperparameter tuning
- Test set reserved for final performance evaluation
- Early stopping based on validation metrics

**Metrics:**
- Primary: Accuracy, weighted F1-score (handles class imbalance)
- Loss tracking: Cross-entropy loss
- Comparison baseline: BETO-uncased (73.26% accuracy from original paper)

**Experiment Tracking:**
- TensorBoard for comprehensive logging:
  - Train/validation loss and accuracy per epoch
  - F1-score (weighted) on validation set
  - Learning rate tracking
  - Gradient and parameter histograms
  - Model graph visualization
  - Sample predictions visualization

### Git Workflow

- **Main branch:** master
- **Branch strategy:** Feature branches from master (recommended)
- **Commit style:** Descriptive commit messages
- **PR process:** PRs to master for significant changes
- **Current status:** Work in progress, active development

## Domain Context

**Research Domain:** Natural Language Processing (NLP), Sentiment Analysis

**Dataset: SENT-COVID**
- Mexican Spanish tweets about COVID-19
- Annotations from TASS2020 competition
- 3-class sentiment classification:
  - Positive sentiment
  - Negative sentiment
  - Neutral sentiment
- Original paper baseline: BETO-uncased with 73.26% accuracy

**Baseline Paper Context:**
The original paper ("Sentiment Analysis of COVID-19 Tweets Using Machine Learning and BERT Models") compared various approaches:
- Lexicon-based: TextBlob, VADER, Pysentimiento
- Traditional ML: Logistic Regression, Naive Bayes, SVM, MLP
- Transformer-based: BERT models (BETO achieved best results)

**This Project's Extension:**
Implement and evaluate RNN-based architectures (GRU, potentially LSTM) with pre-trained Spanish embeddings (GloVe) as an alternative to transformer models, comparing:
- Performance vs. BERT baseline
- Computational efficiency
- Training stability and convergence

## Important Constraints

**Technical Constraints:**
- Must use PyTorch framework
- Must use pre-trained GloVe embeddings on Spanish (SBW corpus)
- Must use TASS2020 annotated data for training
- Report must be in LaTeX format

**Research Constraints:**
- Architecture must be comparable to the baseline paper
- Results must be reproducible
- Performance must be rigorously evaluated against BERT baseline
- Report is work in progress and architecture may evolve

**Development Constraints:**
- Notebooks are for experimentation only (not deliverables)
- Final code must be in `models/` and `utils/`
- Code must be well-documented
- Experiments must be tracked in TensorBoard

**Resource Constraints:**
- Training on available GPU resources (CPU fallback supported)
- Model must be trainable in reasonable time
- Gradient clipping required for RNN stability

## External Dependencies

**Pre-trained Resources:**
- **GloVe Spanish Embeddings:** Pre-trained on SBW (Spanish Billion Words) corpus
  - Dimension: 300
  - Source: External pre-trained vectors

**Datasets:**
- **SENT-COVID Corpus:** Mexican Spanish COVID-19 tweets
  - Source: https://github.com/GIL-UNAM/SENT-COVID
  - Annotations: TASS2020 competition format

**Libraries:**
- **PyTorch:** Core deep learning framework
- **TensorBoard:** Experiment tracking and visualization
- **scikit-learn:** Metrics calculation (F1-score)
- **d2l:** Dive into Deep Learning utilities

**References:**
- Target paper: `papers/informatics-11-00024/ch001.xhtml`
- Project report: `report/report/rnns-for-sentiment-analysis.tex`
- Model implementations: `models/rnns.py`
- Training utilities: `utils/trainer.py`

## Current Status

**Active Development:**
- Model architecture being refined based on report specifications
- Training utilities implemented and functional
- Report being written concurrently with implementation
- Architecture may evolve based on experimental results

**Implementation Status:**
- ✅ Basic RNN implementations (RNNScratch, RNNLMScratch)
- ✅ Training loop with TensorBoard logging
- ✅ Early stopping mechanism
- ✅ Vocabulary management
- ✅ Metrics tracking (accuracy, F1-score)
- 🚧 Final sentiment analysis model architecture
- 🚧 GloVe embedding integration
- 🚧 SENT-COVID data preprocessing pipeline
- 🚧 Research report writing

**Key Implementation Files:**
- `models/rnns.py` - RNN model definitions (lines 10-93)
- `utils/trainer.py` - Training utilities (lines 11-605)
- `utils/vocab.py` - Vocabulary management (lines 4-43)
