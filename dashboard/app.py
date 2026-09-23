import json
import os
import time

import pandas as pd
from flask import Flask, jsonify, render_template, request
from redis import Redis
from rq import Queue
from rq.job import Job
from rq.exceptions import NoSuchJobError

from jobs import retrain_pipeline, LAST_TRAINED_PATH

app = Flask(__name__)

BASE = os.path.join(os.path.dirname(__file__), "..", "data")

TIER_ORDER = ["Low", "Medium", "High", "Lapsed"]
SEGMENT_ORDER = ["Champions", "Promising", "At Risk", "Lost Customers"]

# --- job queue setup -------------------------------------------------------
# One Redis connection, one queue for the (slow) retraining job. Page loads
# never touch this - they only read the CSVs that the job produces.
redis_conn = Redis()
retrain_queue = Queue("retrain", connection=redis_conn)

# Key holding the id of the most recently enqueued retrain job, so a page
# refresh (or a second Flask worker process) can find and poll the same job
# instead of losing track of it.
CURRENT_JOB_KEY = "retailiq:retrain:job_id"
JOB_TIMEOUT_SECONDS = 20 * 60  # covers churn refit + Prophet CV with headroom


def load(filename):
    return pd.read_csv(os.path.join(BASE, filename))


def records(df):
    """DataFrame -> list of dicts with NaN replaced by None (safe for templates / JSON)."""
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def model_settings():
    """Read tier cut-offs / active window from the churn notebook's metadata, so the dashboard
    text can never drift out of sync with the model. Falls back to the notebook defaults."""
    defaults = {"high": 0.55, "medium": 0.35, "active_days": 135}
    try:
        with open(os.path.join(BASE, "churn_model_meta.json")) as f:
            meta = json.load(f)
        return {
            "high": meta["risk_tiers"]["high"],
            "medium": meta["risk_tiers"]["medium"],
            "active_days": meta["active_max_recency_days"],
        }
    except (OSError, KeyError, ValueError):
        return defaults


def tier_counts(churn_df):
    counts = churn_df["Risk_Tier"].value_counts()
    return {t: int(counts.get(t, 0)) for t in TIER_ORDER}


def last_trained_display():
    """Human-readable 'last trained' timestamp for the dashboard, or None if no run yet."""
    try:
        with open(LAST_TRAINED_PATH) as f:
            ts = json.load(f)["timestamp"]
        return time.strftime("%d %b %Y, %H:%M", time.localtime(ts))
    except (OSError, KeyError, ValueError):
        return None


@app.route("/")
def index():
    churn_df    = load("customer_churn_report.csv")
    forecast_df = load("sales_forecast.csv")
    yoy_df      = load("yoy_growth.csv")
    metrics_df  = load("forecast_metrics.csv")
    cfg         = model_settings()

    counts = tier_counts(churn_df)
    scored = counts["Low"] + counts["Medium"] + counts["High"]

    next_month = forecast_df.iloc[0]

    last_yoy = yoy_df.dropna(subset=["YoY_Growth_%"]).iloc[-1]
    yoy_label = last_yoy["Compared_Months"] if "Compared_Months" in yoy_df.columns else "Year-over-year change"

    m = metrics_df.iloc[0]
    mape_baseline = f"{m['MAPE_Baseline']:.1f}" if "MAPE_Baseline" in metrics_df.columns else None

    return render_template("index.html",
        total_customers=len(churn_df),
        scored_customers=scored,
        high_risk_count=counts["High"],
        medium_risk_count=counts["Medium"],
        low_risk_count=counts["Low"],
        lapsed_count=counts["Lapsed"],
        high_cut=round(cfg["high"] * 100),
        med_cut=round(cfg["medium"] * 100),
        active_days=cfg["active_days"],
        next_month_label=next_month["Month"],
        next_month_rev=f"{next_month['Predicted_Revenue']:,.0f}",
        next_month_lower=f"{next_month['Lower_Bound']:,.0f}",
        next_month_upper=f"{next_month['Upper_Bound']:,.0f}",
        yoy_growth=f"{last_yoy['YoY_Growth_%']:+.1f}",
        yoy_label=yoy_label,
        mape=f"{m['MAPE']:.1f}",
        mape_baseline=mape_baseline,
        last_trained=last_trained_display(),
    )


@app.route("/churn")
def churn():
    churn_df = load("customer_churn_report.csv")
    cfg      = model_settings()

    # Lapsed customers are not scored, so their probability is blank (NaN)
    churn_df["Churn_Probability"] = (churn_df["Churn_Probability"] * 100).round(1)

    def top(tier, sort_col, n=50):
        rows = churn_df[churn_df["Risk_Tier"] == tier].sort_values(sort_col, ascending=False)
        return records(rows.head(n))

    high   = top("High",   "Churn_Probability")
    medium = top("Medium", "Churn_Probability")
    low    = top("Low",    "Churn_Probability")
    lapsed = top("Lapsed", "Monetary")            # biggest past spenders first = best win-back targets

    counts = tier_counts(churn_df)
    scored = counts["Low"] + counts["Medium"] + counts["High"]

    seg = churn_df["Segment_Label"].value_counts()
    segment_counts = {k: int(seg[k]) for k in SEGMENT_ORDER if k in seg}
    segment_counts.update({k: int(v) for k, v in seg.items() if k not in segment_counts})

    # NOTE: Monetary is each customer's all-time spend, so this is lifetime spend of the
    # high-risk group (what they have already paid), not forecast future revenue.
    high_risk_spend = churn_df.loc[churn_df["Risk_Tier"] == "High", "Monetary"].sum()

    return render_template("churn.html",
        high=high, medium=medium, low=low, lapsed=lapsed,
        segment_counts=segment_counts,
        risk_counts=counts,
        high_risk_spend=f"{high_risk_spend:,.0f}",
        total=len(churn_df),
        scored=scored,
        high_count=counts["High"],
        medium_count=counts["Medium"],
        low_count=counts["Low"],
        lapsed_count=counts["Lapsed"],
        high_cut=round(cfg["high"] * 100),
        med_cut=round(cfg["medium"] * 100),
        active_days=cfg["active_days"],
    )


@app.route("/forecast")
def forecast():
    forecast_df    = load("sales_forecast.csv")
    full_df        = load("sales_forecast_full.csv")
    seasonality_df = load("monthly_seasonality.csv")
    yoy_df         = load("yoy_growth.csv")
    metrics_df     = load("forecast_metrics.csv")

    if "Actual_Revenue" not in full_df.columns:
        raise RuntimeError(
            "sales_forecast_full.csv has no Actual_Revenue column - re-run sales_forecasting.ipynb "
            "(updated version) so the chart can show real revenue instead of the model's fitted values."
        )

    future_set = set(forecast_df["Month"])

    # History = months with real revenue. Anything without actuals that is not in the 6-month
    # forecast is the month currently in progress (data ends part-way through it).
    history     = full_df[full_df["Actual_Revenue"].notna()]
    in_progress = full_df[full_df["Actual_Revenue"].isna() & ~full_df["Month"].isin(future_set)]
    future      = full_df[full_df["Month"].isin(future_set)]
    projected   = pd.concat([in_progress, future])

    def label(m):
        return f"{m} (in progress)" if m in set(in_progress["Month"]) else m

    yoy_records = records(yoy_df)
    for r in yoy_records:
        r["Months"] = int(r["Months"]) if r.get("Months") is not None else 12

    m = metrics_df.iloc[0]
    def get(col, fmt):
        return fmt.format(m[col]) if col in metrics_df.columns else None

    return render_template("forecast.html",
        forecast_records=records(forecast_df),
        actual_months=history["Month"].tolist(),
        actual_revenue=history["Actual_Revenue"].tolist(),
        future_months=[label(x) for x in projected["Month"]],
        future_revenue=projected["Predicted_Revenue"].tolist(),
        future_lower=projected["Lower_Bound"].tolist(),
        future_upper=projected["Upper_Bound"].tolist(),
        in_progress_month=in_progress["Month"].iloc[0] if len(in_progress) else None,
        season_months=seasonality_df["month_name"].tolist(),
        season_revenue=seasonality_df["y"].tolist(),
        yoy_records=yoy_records,
        mae=f"{m['MAE']:,.0f}",
        rmse=f"{m['RMSE']:,.0f}",
        mape=f"{m['MAPE']:.1f}",
        mape_baseline=get("MAPE_Baseline", "{:.1f}"),
        cv_mape=get("CV_MAPE", "{:.1f}"),
        cv_mape_baseline=get("CV_MAPE_Baseline", "{:.1f}"),
    )


# --- job queue endpoints ----------------------------------------------------

def _active_job():
    """The currently queued/running retrain job, if any, else None."""
    job_id = redis_conn.get(CURRENT_JOB_KEY)
    if not job_id:
        return None
    try:
        job = Job.fetch(job_id.decode(), connection=redis_conn)
    except NoSuchJobError:
        return None
    return job if job.get_status() in ("queued", "started") else job


def _job_payload(job):
    status = job.get_status(refresh=True)
    state = {"started": "running"}.get(status, status)  # queued | running | finished | failed
    payload = {"state": state, "step": job.meta.get("step"), "last_trained": last_trained_display()}
    if state == "failed":
        payload["error"] = (str(job.exc_info).strip().splitlines() or ["Unknown error"])[-1]
    return payload


@app.route("/retrain", methods=["POST"])
def start_retrain():
    """Enqueue the retraining job. A second click while one is already queued/running
    just returns the existing job's status instead of starting a duplicate."""
    existing = _active_job()
    if existing and existing.get_status() in ("queued", "started"):
        return jsonify(_job_payload(existing)), 200

    job = retrain_queue.enqueue(retrain_pipeline, job_timeout=JOB_TIMEOUT_SECONDS)
    redis_conn.set(CURRENT_JOB_KEY, job.id)
    return jsonify(_job_payload(job)), 202


@app.route("/retrain/status")
def retrain_status():
    """Polled by the dashboard. Reports idle when nothing has ever been queued."""
    job_id = redis_conn.get(CURRENT_JOB_KEY)
    if not job_id:
        return jsonify(state="idle", step=None, last_trained=last_trained_display())
    try:
        job = Job.fetch(job_id.decode(), connection=redis_conn)
    except NoSuchJobError:
        return jsonify(state="idle", step=None, last_trained=last_trained_display())
    return jsonify(_job_payload(job))


if __name__ == "__main__":
    app.run(debug=True)
