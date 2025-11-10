# 💰 Personal Expense Tracker (Flask + Netlify)

A lightweight web app to upload and analyze personal spending.
- Backend: Flask (Render)
- Frontend: HTML + JS (Netlify)
- Data: Stored temporarily per guest token in SQLite

## 🌐 Live Demo
- Frontend: https://your-netlify-site.netlify.app
- Backend API: https://your-backend.onrender.com

## 🚀 Features
- Upload CSV (`Date,Category,Amount`)
- View monthly insights & trends
- Get predictions for next month (per category)
- Temporary storage (auto-cleared after 7 days)
- Chart.js visualization

## 🧩 Tech Stack
| Layer | Technology |
|--------|-------------|
| Backend | Flask, SQLAlchemy, Pandas, scikit-learn |
| Frontend | HTML, JS, Chart.js |
| Hosting | Render + Netlify |

## 🧠 CSV Format Example
```csv
Date,Category,Amount
2025-01-03,Groceries,2200
2025-01-05,Travel,1200
...
