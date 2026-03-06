# 🏦 Universal Bank — Personal Loan Analytics Dashboard

A full-stack analytics dashboard built with **Streamlit** that performs Descriptive, Diagnostic, Predictive, and Prescriptive analytics on the Universal Bank dataset to identify customers likely to accept a personal loan offer.

## 🚀 Live Demo

Deploy instantly on [Streamlit Cloud](https://streamlit.io/cloud).

## 📊 Features

### Four Analytics Layers
| Type | What It Does |
|------|-------------|
| **Descriptive** | Customer demographics distribution, loan acceptance rates, financial metric averages |
| **Diagnostic** | Key drivers of loan acceptance — income, education, banking services, correlation heatmap |
| **Predictive** | Decision Tree, Random Forest & Gradient Boosting models with live customer prediction tool |
| **Prescriptive** | Customer segmentation, personalised loan offers, marketing strategy recommendations |

### Visualisations
- 📉 Age & income distribution plots
- 🍩 Interactive drill-down donut chart (by education, family size, age group)
- 🌡️ Correlation heatmap
- 📈 ROC curves for all three models
- 📊 Feature importance bar chart
- 🎯 Customer propensity segmentation
- 🔮 Live prediction gauge with personalised offer engine

### Filters
- Income range slider
- Education level selector
- Family size multi-select
- Loan status radio

## 🛠️ Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/universal-bank-dashboard.git
cd universal-bank-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud

1. Fork or push this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** — done!

> ⚠️ Make sure `UniversalBank.csv` is included in your repository root.

## 📁 Project Structure

```
universal-bank-dashboard/
├── app.py                   # Main Streamlit application
├── UniversalBank.csv        # Dataset (5,000 customers)
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme & server config
└── README.md
```

## 🧠 Models Used

| Model | Purpose |
|-------|---------|
| Decision Tree | Interpretable baseline classifier |
| Random Forest | Ensemble model for higher accuracy |
| Gradient Boosting | Best-performing model used for live predictions |

## 📌 Target Variable
**Personal Loan** — Did this customer accept the personal loan offered in the last campaign? (0 = No, 1 = Yes)

## 🎯 Key Findings
- Only **9.6%** of customers accepted the personal loan
- **Income** is the strongest predictor — customers earning $100K+ are dramatically more likely to accept
- **CD Account holders** are ~4× more likely to accept
- **Advanced education + high income** is the highest-propensity combination
- Gradient Boosting achieves **~98% AUC-ROC**
