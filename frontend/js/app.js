// frontend/js/app.js

// Use window.API_BASE if provided by an inline script in HTML (index.html sets it)
const API_BASE = window.API_BASE || "https://personal-expense-backend-avfq.onrender.com";
console.log("frontend/js/app.js loaded. API_BASE =", API_BASE);

// key used to store guest token locally
const TOKEN_KEY = "guest_token";

/**
 * Ensure we have a guest token. Tries localStorage first; otherwise requests one from backend.
 * Returns the token string or throws on error.
 */
async function ensureToken() {
  let token = localStorage.getItem(TOKEN_KEY);
  if (token) {
    return token;
  }

  try {
    const res = await fetch(`${API_BASE}/get_token`, { method: "GET" });
    if (!res.ok) {
      const text = await res.text().catch(()=>res.statusText);
      throw new Error(`get_token failed: ${res.status} ${text}`);
    }
    const data = await res.json();
    token = data.guest_token;
    if (!token) throw new Error("no token returned");
    localStorage.setItem(TOKEN_KEY, token);
    return token;
  } catch (err) {
    console.error("ensureToken error:", err);
    throw err;
  }
}

/**
 * Upload a CSV file element to the backend.
 * fileInputEl: <input type="file"> element
 * messageEl: DOM element for showing messages
 */
async function uploadFile(fileInputEl, messageEl) {
  const file = fileInputEl.files[0];
  if (!file) {
    messageEl.textContent = "Please choose a file.";
    return;
  }

  let token;
  try {
    token = await ensureToken();
  } catch (err) {
    messageEl.textContent = "Unable to get token from backend. Check console for details.";
    return;
  }

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: form,
      headers: {
        // attach token so backend associates uploaded data with guest
        "X-Guest-Token": token
      }
    });

    if (!res.ok) {
      // try to get server error message
      const errText = await res.text().catch(()=>res.statusText);
      console.error("Upload failed:", res.status, errText);
      messageEl.textContent = `Upload failed: ${res.status} ${errText}`;
      return;
    }

    const data = await res.json();
    if (data.guest_token) {
      localStorage.setItem(TOKEN_KEY, data.guest_token);
    }
    messageEl.textContent = "✅ Upload successful! Open Report page to view charts.";
  } catch (err) {
    console.error("Upload network/CORS error:", err);
    messageEl.textContent = "Upload failed (network/CORS). See console for details.";
  }
}

/**
 * Simple helper to call /report endpoint and return JSON or throw.
 * (Useful for the report page)
 */
async function fetchReport() {
  const token = await ensureToken();
  const res = await fetch(`${API_BASE}/report`, {
    headers: { "X-Guest-Token": token }
  });
  if (!res.ok) {
    const text = await res.text().catch(()=>res.statusText);
    throw new Error(`report failed: ${res.status} ${text}`);
  }
  return res.json();
}

// export functions for use in inline scripts/pages
window.uploadFile = uploadFile;
window.ensureToken = ensureToken;
window.fetchReport = fetchReport;
