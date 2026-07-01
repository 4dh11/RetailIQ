# Intelligent E-Commerce Analytics & Decision Support System

An end-to-end Machine Learning and Data Analytics platform built on real-world e-commerce transaction data. This project implements a complete ETL pipeline, customer segmentation, sales forecasting, churn prediction, and an interactive dashboard to transform raw retail data into actionable business insights.

---

## Overview

The objective of this project is to build an intelligent decision-support system for e-commerce businesses by combining:

* Data Engineering (ETL Pipeline)
* Exploratory Data Analysis
* Machine Learning
* Business Intelligence
* Interactive Dashboard Visualization

The system processes over **1 million retail transactions**, identifies customer behavior patterns, predicts future trends, and generates business recommendations.

---

## Features

### Data Pipeline (ETL)

* Extracts and merges multiple retail datasets
* Cleans missing, invalid, and cancelled transactions
* Performs feature engineering
* Generates ML-ready datasets

### Analytics

* Monthly sales analysis
* Product demand analysis
* Country-wise revenue analysis
* Customer purchasing behavior analysis

### Machine Learning

#### Customer Segmentation

* RFM (Recency, Frequency, Monetary) Analysis
* K-Means Clustering
* Customer group identification:

  * High Value Customers
  * Loyal Customers
  * Occasional Customers
  * At-Risk Customers

#### Sales Forecasting

* Revenue trend prediction
* Future sales estimation
* Business planning support

#### Customer Churn Prediction

* Identifies customers likely to stop purchasing
* Customer retention insights
* Risk scoring

### Decision Support System

Provides actionable recommendations such as:

* Customer retention campaigns
* Re-engagement offers
* Sales improvement strategies
* Product demand planning

### Dashboard

Interactive web dashboard displaying:

* Sales insights
* Customer segments
* Demand trends
* Forecasts
* Business recommendations

---

## Dataset

### Online Retail Dataset (2009–2011)

Source:
[https://www.kaggle.com/datasets/jillwang87/online-retail-ii](https://www.kaggle.com/datasets/jillwang87/online-retail-ii)

Dataset Characteristics:

* 1M+ transaction records
* Real-world retail sales data
* Product information
* Customer information
* Purchase history
* Country information

---

## Technology Stack

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib

### Machine Learning

* Scikit-learn

### Web Framework

* Flask

### Development Environment

* Jupyter Notebook

---

## System Architecture

```text
Raw Retail Data
        │
        ▼
Data Extraction
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Processed Dataset
        │
 ┌──────┼─────────┐
 ▼      ▼         ▼
EDA   ML Models  Dashboard
        │
 ┌──────┼─────────┐
 ▼      ▼         ▼
RFM   Sales    Churn
      Forecast Prediction
        │
        ▼
Business Insights
        │
        ▼
Decision Support
```

---

## Project Workflow

### Phase 1: Data Pipeline

* Dataset ingestion
* Data cleaning
* Missing value handling
* Duplicate removal
* Feature engineering

### Phase 2: Exploratory Data Analysis

* Revenue trends
* Product demand trends
* Customer behavior patterns
* Geographic sales distribution

### Phase 3: Machine Learning

#### RFM Analysis

Customer behavior is quantified using:

* Recency
* Frequency
* Monetary Value

#### K-Means Clustering

Customers are grouped based on purchasing patterns.

#### Sales Forecasting

Predicts future monthly revenue using historical sales data.

#### Churn Prediction

Identifies customers at risk of becoming inactive.

### Phase 4: Decision Support

Generates actionable recommendations based on:

* Customer segments
* Churn risk
* Forecasted sales
* Product demand

---

## Key Insights Generated

### Sales Analytics

* Monthly revenue trends
* Seasonal purchasing patterns
* Country-wise performance

### Customer Analytics

* High-value customers
* Loyal customers
* At-risk customers
* Customer lifetime value indicators

### Product Analytics

* Top-selling products
* Product demand trends
* Inventory planning insights

---

## Results

* Processed over 1 million transaction records
* Built a complete ETL pipeline for retail analytics
* Successfully segmented customers using RFM and K-Means
* Generated sales forecasts and churn predictions
* Delivered business recommendations through an interactive dashboard

---

## Future Enhancements

* Deep Learning-based demand forecasting
* Recommendation system using Association Rule Mining
* LLM-powered analytics assistant
* Real-time data streaming pipeline
* Cloud deployment using AWS or Azure
* Advanced business intelligence dashboards

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ecommerce-analytics-system.git
```

Navigate to the project:

```bash
cd ecommerce-analytics-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

---

## Author

**Adityaa SS**

AI/ML | Full-Stack Developer

GitHub: [https://github.com/4dh11](https://github.com/4dh11)

LinkedIn: [https://www.linkedin.com/in/adityaa-ss-30233b2b3/](https://www.linkedin.com/in/adityaa-ss-30233b2b3/)

---

## License

This project is developed for educational and research purposes.
