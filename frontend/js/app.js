// frontend/js/app.js
const API_BASE = window.API_BASE || "https://<YOUR_RENDER_BACKEND_URL>"; // replace after deploy

async function ensureToken() {
  let token = localStorage.getItem("guest_token");
  if (token) return token;
  const res = await fetch(`${API_BASE}/`);
  const json = await res.json();
  token = json.guest_token;
  localStorage.setItem("guest_token", token);
  return token;
}

async function uploadFile(fileEl, onMessage) {
  const token = await ensureToken();
  const fd = new FormData();
  fd.append("file", fileEl.files[0]);
  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    headers: { "X-Guest-Token": token },
    body: fd
  });
  const json = await res.json();
  if (res.ok && json.guest_token) localStorage.setItem("guest_token", json.guest_token);
  onMessage(res.ok ? `Upload OK: ${json.message || ""}` : `Upload failed: ${json.error || JSON.stringify(json)}`);
}

async function fetchReport() {
  const token = await ensureToken();
  const res = await fetch(`${API_BASE}/report`, { headers: { "X-Guest-Token": token }});
  return res.json();
}

async function fetchExpenses() {
  const token = await ensureToken();
  const res = await fetch(`${API_BASE}/expenses`, { headers: { "X-Guest-Token": token }});
  return res.json();
}

async function clearData() {
  const token = await ensureToken();
  const res = await fetch(`${API_BASE}/clear`, { method: "POST", headers: { "X-Guest-Token": token }});
  if (res.ok) {
    localStorage.removeItem("guest_token");
  }
  return res.json();
}
