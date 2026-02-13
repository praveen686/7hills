import { useState } from "react";
import { useSetAtom } from "jotai";
import { apiFetch, ApiError } from "@/lib/api";
import { pageAtom, userAtom, authTokenAtom, type User } from "@/stores/auth";

/** Extract human-readable detail from FastAPI error responses. */
function extractDetail(e: unknown): string {
  if (e instanceof ApiError) {
    try {
      const body = JSON.parse(e.message.replace(/^API \d+: /, ""));
      if (body.detail) return body.detail;
    } catch { /* not JSON */ }
    return e.message;
  }
  return e instanceof Error ? e.message : "Unknown error";
}

export function LoginButtons() {
  const setPage = useSetAtom(pageAtom);
  const setUser = useSetAtom(userAtom);
  const setToken = useSetAtom(authTokenAtom);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState<string | null>(null);

  const handleAuthSuccess = async (token: string) => {
    localStorage.setItem("ql-token", token);
    setToken(token);
    try {
      const user = await apiFetch<User>("/api/auth/me", {
        headers: { Authorization: `Bearer ${token}` },
      });
      setUser(user);
    } catch {
      // Token is set; user info fetch failed but continue
    }
    setPage("terminal");
  };

  const handleGoogle = async () => {
    setError("");
    setLoading("google");
    try {
      const { url } = await apiFetch<{ url: string }>("/api/auth/google/url");
      window.location.href = url;
    } catch {
      setError("Google auth not configured — set GOOGLE_CLIENT_ID in .env");
    } finally {
      setLoading(null);
    }
  };

  const handleZerodha = async () => {
    setError("");
    setLoading("zerodha");
    try {
      const { token } = await apiFetch<{ token: string }>(
        "/api/auth/zerodha/auto",
        { method: "POST" },
      );
      await handleAuthSuccess(token);
    } catch (e) {
      setError(extractDetail(e));
    } finally {
      setLoading(null);
    }
  };

  const handleBinance = async () => {
    setError("");
    setLoading("binance");
    try {
      const { token } = await apiFetch<{ token: string }>(
        "/api/auth/binance/auto",
        { method: "POST" },
      );
      await handleAuthSuccess(token);
    } catch (e) {
      setError(extractDetail(e));
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="flex flex-col items-center gap-4 w-full max-w-md">
      <div className="flex gap-3 w-full">
        {/* Google */}
        <button
          onClick={handleGoogle}
          disabled={loading !== null}
          className="flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-white text-gray-800 font-medium text-sm border border-gray-200 hover:bg-gray-50 transition-colors shadow-sm disabled:opacity-50"
        >
          {loading === "google" ? (
            <Spinner />
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24">
              <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4"/>
              <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
              <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
              <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
            </svg>
          )}
          Google
        </button>

        {/* Zerodha */}
        <button
          onClick={handleZerodha}
          disabled={loading !== null}
          className="flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-[#387ed1] text-white font-medium text-sm hover:bg-[#2d6ab8] transition-colors shadow-sm disabled:opacity-50"
        >
          {loading === "zerodha" ? (
            <Spinner />
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M4 4h16l-8 16L4 4z"/>
            </svg>
          )}
          Zerodha
        </button>

        {/* Binance */}
        <button
          onClick={handleBinance}
          disabled={loading !== null}
          className="flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-[#F0B90B] text-[#1E2026] font-medium text-sm hover:bg-[#d4a30a] transition-colors shadow-sm disabled:opacity-50"
        >
          {loading === "binance" ? (
            <Spinner />
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2L6.5 7.5 8.5 9.5 12 6l3.5 3.5 2-2L12 2zM2 12l2-2 2 2-2 2-2-2zm16 0l2-2 2 2-2 2-2-2zM12 22l-5.5-5.5 2-2L12 18l3.5-3.5 2 2L12 22zm0-6.5L8.5 12 12 8.5l3.5 3.5L12 15.5z"/>
            </svg>
          )}
          Binance
        </button>
      </div>

      {error && (
        <p className="text-terminal-loss text-xs">{error}</p>
      )}
    </div>
  );
}

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}
