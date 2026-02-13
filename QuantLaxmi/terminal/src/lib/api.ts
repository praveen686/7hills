// ============================================================
// Terminal REST API Client
// Generic fetch wrapper for FastAPI backend
// ============================================================

const BASE_URL = import.meta.env.VITE_API_URL || "";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public endpoint: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function apiFetch<T>(
  endpoint: string,
  options?: RequestInit,
): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  const token = localStorage.getItem("ql-token");
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  const res = await fetch(url, {
    ...options,
    headers: {
      ...headers,
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "Unknown error");
    throw new ApiError(res.status, `API ${res.status}: ${body}`, endpoint);
  }

  return res.json() as Promise<T>;
}

/** Convert HTTP base URL to WebSocket URL. */
export function wsUrl(path: string): string {
  if (BASE_URL) {
    const wsBase = BASE_URL.replace(/^http/, "ws");
    return `${wsBase}${path}`;
  }
  // Relative mode: derive from current page origin
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
}

export { BASE_URL };
