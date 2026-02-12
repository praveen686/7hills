import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

// ---------------------------------------------------------------------------
// cn — class name merge utility (clsx + tailwind-merge)
// ---------------------------------------------------------------------------

/**
 * Merge class names with Tailwind CSS conflict resolution.
 * Equivalent to shadcn/ui's `cn()` utility.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

// ---------------------------------------------------------------------------
// Currency / Price formatting
// ---------------------------------------------------------------------------

const inrFormatter = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const inrFormatterNoDecimals = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  minimumFractionDigits: 0,
  maximumFractionDigits: 0,
});

/**
 * Format a number as Indian Rupees with proper lakhs/crores grouping.
 * @example formatINR(2415000) => "₹24,15,000.00"
 */
export function formatINR(n: number): string {
  return inrFormatter.format(n);
}

/**
 * Format a number as a price with configurable decimal places.
 * Uses Indian locale grouping.
 * @example formatPrice(24150.5, 2) => "24,150.50"
 */
export function formatPrice(n: number, decimals: number = 2): string {
  return new Intl.NumberFormat("en-IN", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(n);
}

// ---------------------------------------------------------------------------
// Percentage formatting
// ---------------------------------------------------------------------------

/**
 * Format a number as a signed percentage.
 * @example formatPct(4.5)  => "+4.50%"
 * @example formatPct(-1.2) => "-1.20%"
 */
export function formatPct(n: number): string {
  const sign = n >= 0 ? "+" : "";
  return `${sign}${n.toFixed(2)}%`;
}

// ---------------------------------------------------------------------------
// P&L formatting
// ---------------------------------------------------------------------------

/**
 * Format a P&L number as signed INR (no decimals for readability).
 * @example formatPnl(32500)  => "+₹32,500"
 * @example formatPnl(-12000) => "-₹12,000"
 */
export function formatPnl(n: number): string {
  const sign = n >= 0 ? "+" : "-";
  const formatted = inrFormatterNoDecimals.format(Math.abs(n));
  // inrFormatterNoDecimals already adds ₹, just need the sign
  return `${sign}${formatted}`;
}

// ---------------------------------------------------------------------------
// Volume formatting
// ---------------------------------------------------------------------------

/**
 * Format volume with K/M/B shorthand.
 * @example formatVolume(1200000) => "1.2M"
 * @example formatVolume(350000)  => "350K"
 * @example formatVolume(999)     => "999"
 */
export function formatVolume(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 1_000_000_000) {
    return `${(n / 1_000_000_000).toFixed(1)}B`;
  }
  if (abs >= 1_000_000) {
    return `${(n / 1_000_000).toFixed(1)}M`;
  }
  if (abs >= 1_000) {
    return `${(n / 1_000).toFixed(0)}K`;
  }
  return n.toString();
}

// ---------------------------------------------------------------------------
// Timestamp formatting
// ---------------------------------------------------------------------------

/**
 * Format an ISO timestamp string to HH:MM:SS.
 * @example formatTimestamp("2026-02-12T14:32:15.123Z") => "14:32:15"
 */
export function formatTimestamp(ts: string): string {
  const date = new Date(ts);
  const h = String(date.getHours()).padStart(2, "0");
  const m = String(date.getMinutes()).padStart(2, "0");
  const s = String(date.getSeconds()).padStart(2, "0");
  return `${h}:${m}:${s}`;
}
