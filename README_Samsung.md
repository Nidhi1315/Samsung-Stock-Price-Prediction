# 📈 Samsung Stock Price Prediction using LSTM

A deep learning project that predicts Samsung's stock closing price using a **Long Short-Term Memory (LSTM)** neural network trained on over 20 years of historical stock data.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Workflow](#project-workflow)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [How to Run](#how-to-run)
- [Results](#results)

---

## 📌 Overview

This project uses a stacked LSTM model to learn patterns from Samsung's historical stock prices and predict future closing prices. The model is trained on 80% of historical data and evaluated on the remaining 20%.

---

## 📊 Dataset

- **File:** `Samsung.csv`
- **Source:** Historical stock data (from 2000-01-04 onwards)
- **Total Records:** 5,621 trading days

| Column | Description |
|---|---|
| Date | Trading date |
| Open | Opening price |
| High | Highest price of the day |
| Low | Lowest price of the day |
| Close | Closing price (target variable) |
| Adj Close | Adjusted closing price |
| Volume | Number of shares traded |

**Key Stats (Close Price):**
- Min: ₩2,730 | Max: ₩91,000 | Mean: ₩24,880

> ✅ No missing values found in the dataset.

---

## 🛠️ Tech Stack

- Python 3
- NumPy
- Pandas
- Matplotlib & Seaborn
- Scikit-learn (MinMaxScaler, RMSE)
- Keras / TensorFlow (LSTM, Dense, Sequential)
- Google Colab

---

## 🔄 Project Workflow

1. **Import Libraries** — NumPy, Pandas, Matplotlib, Keras, Scikit-learn
2. **Load Dataset** — Upload and read `Samsung.csv`
3. **Exploratory Data Analysis** — Shape, statistics, null checks, closing price visualization
4. **Preprocessing** — Extract `Close` column, apply `MinMaxScaler` (range 0–1)
5. **Train-Test Split** — 80% training / 20% testing
6. **Create Sequences** — Sliding window of 60 time steps to predict next price
7. **Reshape Data** — Reshape to 3D array `(samples, time_steps, features)` for LSTM input
8. **Build LSTM Model** — Stacked LSTM layers with Dense output
9. **Train Model** — 50 epochs, batch size 32, Adam optimizer, MSE loss
10. **Evaluate** — RMSE on test set, visualize actual vs predicted prices

---

## 🧠 Model Architecture

```
Model: Sequential
─────────────────────────────────────
Layer               Output Shape
─────────────────────────────────────
LSTM(50, return_sequences=True)   (None, 60, 50)
LSTM(50, return_sequences=False)  (None, 50)
Dense(25)                         (None, 25)
Dense(1)                          (None, 1)
─────────────────────────────────────
Optimizer : Adam
Loss      : Mean Squared Error
```

---

## ⚙️ Model Training

- **Epochs:** 50
- **Batch Size:** 32
- **Time Steps:** 60 (uses last 60 days to predict next day)
- Training loss steadily decreased from ~1.58e-04 (Epoch 1) across all 50 epochs

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/samsung-stock-prediction.git
   cd samsung-stock-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
   ```

3. **Add the dataset**
   Place `Samsung.csv` in the project root directory.

4. **Run the notebook**
   Open `Samsung_stock_price_prediction.ipynb` in Jupyter Notebook or Google Colab and run all cells.

   > **Note:** If running on Google Colab, the notebook uses `files.upload()` to upload `Samsung.csv` directly. On Jupyter, update the path to `pd.read_csv('Samsung.csv')`.

---

## 📉 Results

The model plots **Actual vs Predicted** closing prices on the test set, evaluated using **Root Mean Squared Error (RMSE)** to measure prediction accuracy.

---

## 📁 Project Structure

```
samsung-stock-prediction/
│
├── Samsung.csv                                  # Dataset
├── Samsung_stock_price_prediction.ipynb         # Main notebook
└── README.md                                    # Project documentation
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
