const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let message = `Request failed with ${response.status}`;
    try {
      const data = await response.json();
      message = data.detail || message;
    } catch {
      message = response.statusText || message;
    }
    throw new Error(message);
  }
  return response.json();
}

export function searchFighters(q) {
  return request(`/fighters?q=${encodeURIComponent(q)}`);
}

export function predictFight(aId, bId) {
  return request("/predict", {
    method: "POST",
    body: JSON.stringify({ fighter_a_id: aId, fighter_b_id: bId }),
  });
}

export function predictFightByRef(aRefNo, bRefNo) {
  return request("/predict", {
    method: "POST",
    body: JSON.stringify({ fighter_a_ref_no: Number(aRefNo), fighter_b_ref_no: Number(bRefNo) }),
  });
}

export function getFighterStats(id) {
  return request(`/fighter/${encodeURIComponent(id)}/stats`);
}

export function getFighterByRef(refNo) {
  return request(`/fighters/ref/${encodeURIComponent(refNo)}`);
}
