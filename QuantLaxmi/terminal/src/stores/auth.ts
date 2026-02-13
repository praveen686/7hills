import { atom } from "jotai";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface User {
  id: string;
  email: string;
  name: string;
  provider: "google" | "zerodha" | "binance";
  avatar?: string;
  // Zerodha-specific
  broker?: string;
  exchanges?: string[];
  products?: string[];
  order_types?: string[];
  // Binance-specific
  permissions?: string[];
  account_type?: string;
  can_trade?: boolean;
  can_withdraw?: boolean;
  can_deposit?: boolean;
}

// ---------------------------------------------------------------------------
// Atoms
// ---------------------------------------------------------------------------

/** Controls which page is displayed */
export type PageId = "landing" | "terminal" | "dashboard" | "strategies" | "settings" | "profile";

export const pageAtom = atom<PageId>("landing");

/** Currently authenticated user (null = anonymous) */
export const userAtom = atom<User | null>(null);

/** Whether an auth flow is in progress */
export const authLoadingAtom = atom(false);

/** JWT token persisted to localStorage */
const storedToken =
  typeof window !== "undefined" ? localStorage.getItem("ql-token") : null;

export const authTokenAtom = atom<string | null>(storedToken);

/** Check URL for OAuth callback token */
export function checkUrlToken(): string | null {
  if (typeof window === "undefined") return null;
  const params = new URLSearchParams(window.location.search);
  const token = params.get("token");
  if (token) {
    localStorage.setItem("ql-token", token);
    // Clean URL
    window.history.replaceState({}, "", window.location.pathname);
    return token;
  }
  return null;
}
