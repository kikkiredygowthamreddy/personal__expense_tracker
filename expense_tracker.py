# expense_tracker.py
"""
Simple Personal Expenses Tracker (starter)
Requirements: pandas, numpy, scikit-learn, matplotlib, seaborn
Install: pip install pandas numpy scikit-learn matplotlib seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error

sns.set(style="whitegrid")


# -------------------------
# Helper: sample CSV format
# -------------------------
# Create a CSV named `expenses.csv` with columns:
# Date,Category,Amount
# 2024-01-03,Groceries,320.50
# 2024-01-05,Travel,1200.00
# 2024-01-07,Restaurants,450.00
# 2024-02-02,Groceries,540.00
# ...
#
# Dates must be YYYY-MM-DD. Category is text. Amount numeric.
# -------------------------


def load_and_preprocess(path="expenses.csv"):
    df = pd.read_csv(path, parse_dates=["Date"])
    # Ensure types
    df["Category"] = df["Category"].astype(str).str.strip().str.title()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    # Add Year-Month column to aggregate monthly
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    return df


def monthly_category_pivot(df):
    # Aggregate to monthly category totals
    pivot = df.groupby(["YearMonth", "Category"])["Amount"].sum().reset_index()
    # Convert to wide format: rows = YearMonth, cols = Category
    wide = pivot.pivot(index="YearMonth", columns="Category", values="Amount").fillna(0).sort_index()
    # Ensure YearMonth is ordered chronologically
    wide.index = pd.to_datetime(wide.index + "-01")
    wide = wide.sort_index()
    wide.index.name = "Date"
    return wide


def plot_spending_over_time(wide):
    plt.figure(figsize=(10, 5))
    # plot top 6 categories by total spend
    totals = wide.sum().sort_values(ascending=False)
    top_cats = totals.head(6).index.tolist()
    wide[top_cats].plot(marker='o')
    plt.title("Monthly spending by category (top categories)")
    plt.ylabel("Amount")
    plt.xlabel("Month")
    plt.legend(title="Category")
    plt.tight_layout()
    plt.show()


def train_predict_next_month(wide, category):
    """
    Train a simple linear regression on month index -> amount for one category
    Returns predicted value for the next month and MAE on test split.
    """
    # Prepare time numeric feature: months since start
    df = wide[[category]].copy()
    df = df.reset_index()
    df["month_idx"] = ((df["Date"].dt.year - df["Date"].dt.year.min()) * 12 +
                       (df["Date"].dt.month - df["Date"].dt.month.min()))
    X = df[["month_idx"]].values
    y = df[category].values

    if len(df) < 3:
        # Not enough data for reliable model
        return None, None, "not_enough_data"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Predict next month
    next_idx = np.array([[X.max() + 1]])
    next_pred = float(model.predict(next_idx)[0])
    return next_pred, mae, model


def generate_insights(wide):
    # Last month actuals vs previous average
    if wide.shape[0] < 2:
        print("Not enough months to generate insights.")
        return

    last_month = wide.index.max()
    prev_months = wide.index[:-1]
    last_vals = wide.loc[last_month]
    prev_avg = wide.loc[prev_months].mean()

    insights = []
    for cat in wide.columns:
        prev = prev_avg.get(cat, 0.0)
        last = last_vals.get(cat, 0.0)
        if prev == 0 and last == 0:
            continue
        if prev == 0:
            pct = np.inf
        else:
            pct = (last - prev) / prev * 100

        if pct == np.inf:
            insights.append(f"{cat}: New spending this month: ₹{last:.2f}.")
        elif pct > 20:
            insights.append(f"{cat}: You spent {pct:.1f}% more than usual (₹{last:.2f} vs avg ₹{prev:.2f}).")
        elif pct < -20:
            insights.append(f"{cat}: You spent {abs(pct):.1f}% less than usual (₹{last:.2f} vs avg ₹{prev:.2f}).")
        # else: no notable change

    if not insights:
        print("No notable changes this month.")
    else:
        print("Insights (last month vs previous average):")
        for s in insights:
            print(" -", s)


def main(csv_path="expenses.csv"):
    df = load_and_preprocess(csv_path)
    print(f"Loaded {len(df)} expense rows.")

    wide = monthly_category_pivot(df)
    if wide.empty:
        print("No monthly data after pivot. Check your CSV.")
        return

    # Show visualization
    plot_spending_over_time(wide)

    # Generate textual insights
    generate_insights(wide)

    # Predict next month for top categories
    totals = wide.sum().sort_values(ascending=False)
    top_cats = totals.head(4).index.tolist()
    print("\nPredictions for next month (simple linear trend per category):")
    for cat in top_cats:
        pred, mae, model_or_flag = train_predict_next_month(wide, cat)
        if pred is None:
            print(f" - {cat}: Not enough data to predict.")
        elif model_or_flag == "not_enough_data":
            print(f" - {cat}: Not enough monthly points to build a model.")
        else:
            last_actual = float(wide[cat].iloc[-1])
            if last_actual == 0:
                pct_change = np.inf
            else:
                pct_change = (pred - last_actual) / last_actual * 100
            change_str = "∞" if pct_change == np.inf else f"{pct_change:.1f}%"
            print(f" - {cat}: Predicted ₹{pred:.2f} (last month ₹{last_actual:.2f}, change {change_str}, MAE {mae:.2f})")


if __name__ == "__main__":
    # call main with default CSV path
    main("expenses.csv")
