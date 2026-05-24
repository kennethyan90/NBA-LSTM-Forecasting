# NBA-LSTM-Forecasting
LSTM-based deep learning system for NBA time series forecasting, including points, rebounds, assists and statistical event prediction.
---

# Overview

This project was developed for an academic Deep Learning and Time Series Forecasting assignment focused on NBA statistical prediction using LSTM (Long Short-Term Memory) neural networks.

The system performs:

- Regression forecasting:
  - Points (PTS)
  - Rebounds (REB)
  - Assists (AST)

- Binary classification:
  - Probability of scoring above 100 points
  - Probability of obtaining more than 30 rebounds
  - Probability of achieving more than 20 assists

The project also includes:
- Feature engineering
- Sliding windows
- Time series preprocessing
- Model evaluation
- Statistical interpretation

---

# Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

---

# Dataset

The project uses NBA game statistics from the 2025/2026 season.

Main variables include:
- Points
- Rebounds
- Assists
- Field Goal Percentage
- Turnovers
- Home/Away condition
- Winning streak
- Rest days

---

# Data Engineering

Several feature engineering techniques were applied:

- Moving averages (5 and 10 games)
- Offensive trend analysis
- Winning streak calculation
- Home/Away encoding
- Rest day computation
- Feature scaling using StandardScaler

Selected features:
- PTS_MA5
- PTS_MA10
- FG_PCT
- AST

---

# Time Series Construction

Sliding Windows were used to transform the sequential data into temporal learning samples.

Tested window sizes:
- 5 games
- 10 games
- 15 games
- 20 games

---

# LSTM Architecture

The regression model architecture includes:

- LSTM layer (16 neurons)
- Dropout layer (0.2)
- Dense output layer

Techniques used to reduce overfitting:
- Dropout
- Early Stopping
- Feature selection
- Standardization

---

# Regression Results

| Window | MAE | RMSE | R² |
|---|---|---|---|
| 5 | 9.74 | 12.13 | 0.060 |
| 10 | 10.78 | 13.52 | -0.126 |
| 15 | 11.00 | 13.01 | -0.094 |
| 20 | 9.35 | 11.61 | 0.026 |

Best result:
- Window size = 20

---

# Classification Results

| Target | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| PTS_100 | 1.000 | 1.000 | 1.000 | 1.000 |
| REB_30 | 1.000 | 1.000 | 1.000 | 1.000 |
| AST_20 | 0.885 | 0.917 | 0.957 | 0.936 |

---

# Key Observations

- NBA data presents high natural variability
- Small datasets limit deep learning generalization
- Classification suffered from class imbalance
- LSTM captured temporal offensive patterns successfully

---

# Functional Requirements Covered

- RF1 – Statistical prediction system
- RF2 – Binary event prediction
- RF3 – Points/Rebounds/Assists forecasting
- RF4 – Data engineering
- RF5 – LSTM implementation
- RF6 – Performance evaluation
- RF7 – Technical executive reporting

---

# Future Improvements

Possible future enhancements include:

- Multi-season training
- Player-level statistics
- Advanced ensemble methods
- Hybrid statistical + deep learning models
- Attention mechanisms
- Transformer architectures

---

# Authors

- Kenneth Yan Santana Oliveira
- [Nome da colega]

---

# Academic Context

This repository was developed as part of an academic assignment involving:
- Deep Learning
- Time Series Forecasting
- Sports Analytics
- Neural Networks
- LSTM architectures

---

# References

Main references used in the project:

- Goodfellow, Bengio & Courville – Deep Learning
- Hochreiter & Schmidhuber – Long Short-Term Memory
- Chollet – Deep Learning with Python
- Géron – Hands-On Machine Learning

---

# License

This project was developed for academic and educational purposes.
