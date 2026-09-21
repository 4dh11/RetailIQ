import os
import pandas as pd
from flask import Flask, render_template

app = Flask(__name__)

BASE = os.path.join(os.path.dirname(__file__), "..", "data")

def load(filename):
    return pd.read_csv(os.path.join(BASE, filename))

@app.route("/")
def index():
    churn_df   = load("customer_churn_report.csv")
    forecast_df = load("sales_forecast.csv")
    yoy_df     = load("yoy_growth.csv")
    metrics_df = load("forecast_metrics.csv")

    total_customers  = len(churn_df)
    high_risk_count  = int((churn_df["Risk_Tier"] == "High").sum())
    medium_risk_count = int((churn_df["Risk_Tier"] == "Medium").sum())
    low_risk_count   = int((churn_df["Risk_Tier"] == "Low").sum())

    next_month       = forecast_df.iloc[0]
    next_month_label = next_month["Month"]
    next_month_rev   = f"{next_month['Predicted_Revenue']:,.0f}"
    next_month_lower = f"{next_month['Lower_Bound']:,.0f}"
    next_month_upper = f"{next_month['Upper_Bound']:,.0f}"

    last_yoy = yoy_df.dropna(subset=["YoY_Growth_%"]).iloc[-1]
    yoy_growth = f"{last_yoy['YoY_Growth_%']:+.1f}"

    mape = f"{metrics_df.iloc[0]['MAPE']:.1f}"

    return render_template("index.html",
        total_customers=total_customers,
        high_risk_count=high_risk_count,
        medium_risk_count=medium_risk_count,
        low_risk_count=low_risk_count,
        next_month_label=next_month_label,
        next_month_rev=next_month_rev,
        next_month_lower=next_month_lower,
        next_month_upper=next_month_upper,
        yoy_growth=yoy_growth,
        mape=mape,
    )

@app.route("/churn")
def churn():
    churn_df  = load("customer_churn_report.csv")
    summary_df = load("intervention_summary.csv")

    churn_df["Churn_Probability"] = (churn_df["Churn_Probability"] * 100).round(1)

    high   = churn_df[churn_df["Risk_Tier"] == "High"].sort_values("Churn_Probability", ascending=False).head(50)
    medium = churn_df[churn_df["Risk_Tier"] == "Medium"].sort_values("Churn_Probability", ascending=False).head(50)
    low    = churn_df[churn_df["Risk_Tier"] == "Low"].sort_values("Churn_Probability", ascending=False).head(50)

    segment_counts = churn_df["Segment_Label"].value_counts().to_dict()
    risk_counts    = churn_df["Risk_Tier"].value_counts().to_dict()

    revenue_at_risk = summary_df[summary_df["Risk_Tier"] == "High"]["Revenue_At_Risk"].values
    revenue_at_risk = f"{revenue_at_risk[0]:,.0f}" if len(revenue_at_risk) else "N/A"

    return render_template("churn.html",
        high=high.to_dict("records"),
        medium=medium.to_dict("records"),
        low=low.to_dict("records"),
        segment_counts=segment_counts,
        risk_counts=risk_counts,
        revenue_at_risk=revenue_at_risk,
        total=len(churn_df),
        high_count=len(high),
        medium_count=len(medium),
        low_count=len(low),
    )

@app.route("/forecast")
def forecast():
    forecast_df    = load("sales_forecast.csv")
    full_df        = load("sales_forecast_full.csv")
    seasonality_df = load("monthly_seasonality.csv")
    yoy_df         = load("yoy_growth.csv")
    metrics_df     = load("forecast_metrics.csv")

    forecast_records = forecast_df.to_dict("records")

    actual_months  = full_df[full_df["Month"] <= full_df["Month"].iloc[-7]]["Month"].tolist()
    actual_revenue = full_df[full_df["Month"] <= full_df["Month"].iloc[-7]]["Predicted_Revenue"].tolist()

    future_months  = forecast_df["Month"].tolist()
    future_revenue = forecast_df["Predicted_Revenue"].tolist()
    future_lower   = forecast_df["Lower_Bound"].tolist()
    future_upper   = forecast_df["Upper_Bound"].tolist()

    season_months  = seasonality_df["month_name"].tolist()
    season_revenue = seasonality_df["y"].tolist()

    yoy_records = yoy_df.to_dict("records")

    mae  = f"{metrics_df.iloc[0]['MAE']:,.0f}"
    rmse = f"{metrics_df.iloc[0]['RMSE']:,.0f}"
    mape = f"{metrics_df.iloc[0]['MAPE']:.1f}"

    return render_template("forecast.html",
        forecast_records=forecast_records,
        actual_months=actual_months,
        actual_revenue=actual_revenue,
        future_months=future_months,
        future_revenue=future_revenue,
        future_lower=future_lower,
        future_upper=future_upper,
        season_months=season_months,
        season_revenue=season_revenue,
        yoy_records=yoy_records,
        mae=mae,
        rmse=rmse,
        mape=mape,
    )

if __name__ == "__main__":
    app.run(debug=True)
