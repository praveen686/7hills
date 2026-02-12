// ============================================================
// Terminal REST API Client
// Generic fetch wrapper for FastAPI backend
// ============================================================

const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

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
  const res = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "Unknown error");
    throw new ApiError(res.status, `API ${res.status}: ${body}`, endpoint);
  }

  return res.json() as Promise<T>;
}

/** Convert HTTP base URL to WebSocket URL. */
export function wsUrl(path: string): string {
  const wsBase = BASE_URL.replace(/^http/, "ws");
  return `${wsBase}${path}`;
}

export { BASE_URL };
