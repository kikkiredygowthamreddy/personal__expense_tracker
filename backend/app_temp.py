# backend/app_temp.py
import os
import io
import uuid
import datetime
from flask import Flask, request, jsonify, make_response, send_file
from flask_cors import CORS
from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ---------------- CONFIG ----------------
DB_URL = os.environ.get("TEMP_DB_URL", "sqlite:///temp_expenses.db")
TTL_DAYS = int(os.environ.get("TEMP_TTL_DAYS", "7"))
COOKIE_NAME = "guest_token"
FRONTEND_ORIGINS = os.environ.get(
    "FRONTEND_ORIGINS",
    "*"  # For testing, allow all. In production, set to your Netlify URL.
)
# ----------------------------------------

# create app BEFORE calling CORS
app = Flask(__name__, static_folder="../frontend", template_folder="../frontend")

# configure CORS
CORS(
    app,
    resources={r"/*": {"origins": FRONTEND__ORIGINS}},
    supports_credentials=True,
    allow_headers=["Content-Type", "X-Guest-Token", "Authorization"]
)

# ---------- Force-add CORS headers for every response (helps preflight) ----------
_allowed_origin = FRONTEND_ORIGINS if FRONTEND_ORIGINS else "*"

@app.after_request
def add_cors_headers(response):
    """
    Ensure every response includes CORS headers.
    Fixes 'No Access-Control-Allow-Origin' error in browsers.
    """
    response.headers["Access-Control-Allow-Origin"] = _allowed_origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Guest-Token, Authorization"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


# ---------- DB setup ----------
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class GuestExpense(Base):
    __tablename__ = "guest_expenses"
    id = Column(Integer, primary_key=True)
    guest_token = Column(String, index=True, nullable=False)
    date = Column(Date, nullable=False)
    category = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    created_at = Column(Date, nullable=False)

Base.metadata.create_all(bind=engine)

# ---------- Helpers ----------
def create_guest_token():
    return str(uuid.uuid4())

def get_guest_token_from_request():
    token = request.headers.get("X-Guest-Token")
    if token:
        return token
    token = request.cookies.get(COOKIE_NAME)
    return token

def set_guest_cookie(response, token):
    max_age = TTL_DAYS * 24 * 3600
    expires = datetime.datetime.utcnow() + datetime.timedelta(seconds=max_age)
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=max_age,
        expires=expires,
        httponly=True,
        samesite="Lax"
    )
    return response

def cleanup_old_data(db):
    cutoff = datetime.date.today() - datetime.timedelta(days=TTL_DAYS)
    db.query(GuestExpense).filter(GuestExpense.created_at < cutoff).delete()
    db.commit()

def parse_and_save_df(df: pd.DataFrame, guest_token: str, db_session):
    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() in ("date", "transaction_date", "timestamp")), None)
    cat_col = next((c for c in df.columns if c.lower() in ("category", "cat", "description", "vendor")), None)
    amt_col = next((c for c in df.columns if c.lower() in ("amount", "amt", "value")), None)
    if date_col is None or cat_col is None or amt_col is None:
        raise ValueError("CSV must contain Date, Category, and Amount columns (or similar).")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[amt_col] = pd.to_numeric(df[amt_col], errors="coerce")
    today = datetime.date.today()

    for _, row in df.iterrows():
        if pd.isna(row[date_col]) or pd.isna(row[amt_col]):
            continue
        exp = GuestExpense(
            guest_token=guest_token,
            date=(row[date_col].date() if isinstance(row[date_col], pd.Timestamp) else row[date_col]),
            category=str(row[cat_col]).strip().title(),
            amount=float(row[amt_col]),
            created_at=today
        )
        db_session.add(exp)
    db_session.commit()

# ---------- Analysis ----------
def build_monthly_wide(db, guest_token):
    q = db.query(GuestExpense).filter(GuestExpense.guest_token == guest_token).all()
    if not q:
        return pd.DataFrame()
    rows = [{"Date": e.date, "Category": e.category, "Amount": e.amount} for e in q]
    df = pd.DataFrame(rows)
    df["YearMonth"] = pd.to_datetime(df["Date"]).dt.to_period("M").astype(str)
    pivot = df.groupby(["YearMonth", "Category"])["Amount"].sum().reset_index()
    wide = pivot.pivot(index="YearMonth", columns="Category", values="Amount").fillna(0).sort_index()
    if wide.empty:
        return wide
    wide.index = pd.to_datetime(wide.index + "-01")
    wide.index.name = "Date"
    return wide

def generate_insights_text(wide):
    if wide.shape[0] < 2:
        return ["Not enough months to generate insights."]
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
        pct = np.inf if prev == 0 else (last - prev) / prev * 100
        if pct == np.inf:
            insights.append(f"{cat}: New spending this month: ₹{last:.2f}.")
        elif pct > 20:
            insights.append(f"{cat}: You spent {pct:.1f}% more than usual (₹{last:.2f} vs avg ₹{prev:.2f}).")
        elif pct < -20:
            insights.append(f"{cat}: You spent {abs(pct):.1f}% less than usual (₹{last:.2f} vs avg ₹{prev:.2f}).")
    if not insights:
        insights = ["No notable changes this month."]
    return insights

def predict_next_month(wide):
    preds = {}
    for cat in wide.columns:
        dfc = wide[[cat]].copy().reset_index()
        dfc["month_idx"] = ((dfc["Date"].dt.year - dfc["Date"].dt.year.min()) * 12 +
                            (dfc["Date"].dt.month - dfc["Date"].dt.month.min()))
        X = dfc[["month_idx"]].values
        y = dfc[cat].values
        if len(dfc) < 4:
            preds[cat] = {"status": "not_enough_data"}
            continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = LinearRegression().fit(X_train, y_train)
        next_idx = np.array([[X.max() + 1]])
        pred = float(model.predict(next_idx)[0])
        mae = float(mean_absolute_error(y_test, model.predict(X_test))) if len(X_test) > 0 else None
        preds[cat] = {"predicted": round(pred, 2), "mae": (round(mae, 2) if mae is not None else None)}
    return preds

# ---------- Request hooks ----------
@app.before_request
def before_request():
    db = SessionLocal()
    cleanup_old_data(db)
    db.close()

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    token = get_guest_token_from_request()
    if not token:
        token = create_guest_token()
    resp = jsonify({"message": "temporary expense tracker", "ttl_days": TTL_DAYS, "guest_token": token})
    set_guest_cookie(resp, token)
    return resp

@app.route("/get_token", methods=["GET"])
def get_token():
    token = create_guest_token()
    resp = jsonify({"guest_token": token})
    set_guest_cookie(resp, token)
    return resp

@app.route("/upload", methods=["POST"])
def upload():
    token = get_guest_token_from_request()
    if not token:
        token = create_guest_token()
    if "file" not in request.files:
        return jsonify({"error": "file field required (multipart/form-data)"}), 400
    f = request.files["file"]
    try:
        df = pd.read_csv(io.BytesIO(f.read()))
    except Exception as e:
        return jsonify({"error": f"unable to read CSV: {e}"}), 400
    db = SessionLocal()
    try:
        parse_and_save_df(df, token, db)
    except Exception as e:
        db.close()
        return jsonify({"error": str(e)}), 400
    db.close()
    resp = jsonify({"status": "ok", "message": "uploaded", "guest_token": token})
    set_guest_cookie(resp, token)
    return resp

@app.route("/expenses", methods=["GET"])
def list_expenses():
    token = get_guest_token_from_request()
    if not token:
        return jsonify([])
    db = SessionLocal()
    items = db.query(GuestExpense).filter(GuestExpense.guest_token == token).order_by(GuestExpense.date.desc()).all()
    out = [{"id": e.id, "date": e.date.isoformat(), "category": e.category, "amount": e.amount} for e in items]
    db.close()
    return jsonify(out)

@app.route("/report", methods=["GET"])
def report():
    token = get_guest_token_from_request()
    if not token:
        return jsonify({"error": "no guest token (upload first)"}), 400
    db = SessionLocal()
    wide = build_monthly_wide(db, token)
    db.close()
    if wide.empty:
        return jsonify({"error": "no expense data"}), 400
    insights = generate_insights_text(wide)
    preds = predict_next_month(wide)
    months = [d.strftime("%Y-%m") for d in wide.index]
    series = {col: [float(x) for x in wide[col].values] for col in wide.columns}
    return jsonify({"insights": insights, "predictions": preds, "months": months, "series": series})

@app.route("/export", methods=["GET"])
def export_csv():
    token = get_guest_token_from_request()
    if not token:
        return jsonify({"error": "no guest token"}), 400
    db = SessionLocal()
    rows = db.query(GuestExpense).filter(GuestExpense.guest_token == token).order_by(GuestExpense.date).all()
    db.close()
    if not rows:
        return jsonify({"error": "no data to export"}), 400
    out_df = pd.DataFrame([{"Date": r.date, "Category": r.category, "Amount": r.amount} for r in rows])
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="guest_expenses.csv"
    )

@app.route("/clear", methods=["POST"])
def clear_data():
    token = get_guest_token_from_request()
    if not token:
        return jsonify({"status": "no token"}), 200
    db = SessionLocal()
    deleted = db.query(GuestExpense).filter(GuestExpense.guest_token == token).delete()
    db.commit()
    db.close()
    resp = jsonify({"status": "cleared", "rows_deleted": deleted})
    resp.set_cookie(COOKIE_NAME, "", expires=0)
    return resp

@app.route("/ui", methods=["GET"])
def ui_index():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)



