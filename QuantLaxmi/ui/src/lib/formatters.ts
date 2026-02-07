// ============================================================
// QuantLaxmi Formatters
// Number formatting utilities for trading display
// ============================================================

const INR = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  minimumFractionDigits: 0,
  maximumFractionDigits: 0,
});

const INR_PRECISE = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const NUM_COMPACT = new Intl.NumberFormat("en-IN", {
  notation: "compact",
  compactDisplay: "short",
  maximumFractionDigits: 1,
});

/** Format as INR currency without decimal: "12,34,567" */
export function formatCurrency(value: number): string {
  return INR.format(value);
}

/** Format as INR currency with 2 decimals: "12,34,567.89" */
export function formatCurrencyPrecise(value: number): string {
  return INR_PRECISE.format(value);
}

/** Format large numbers in compact form: "1.2Cr", "45.3L" */
export function formatCompact(value: number | null | undefined): string {
  if (value == null) return "--";
  const absValue = Math.abs(value);
  const sign = value < 0 ? "-" : "";

  if (absValue >= 1e7) {
    return `${sign}${(absValue / 1e7).toFixed(2)}Cr`;
  }
  if (absValue >= 1e5) {
    return `${sign}${(absValue / 1e5).toFixed(2)}L`;
  }
  if (absValue >= 1e3) {
    return `${sign}${(absValue / 1e3).toFixed(1)}K`;
  }
  return NUM_COMPACT.format(value);
}

/** Format percentage: "+12.34%" or "-5.67%" */
export function formatPct(value: number | null | undefined, decimals: number = 2): string {
  if (value == null) return "--";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(decimals)}%`;
}

/** Format percentage without sign: "12.34%" */
export function formatPctUnsigned(value: number | null | undefined, decimals: number = 2): string {
  if (value == null) return "--";
  return `${value.toFixed(decimals)}%`;
}

/** Format Sharpe ratio: "1.85" */
export function formatSharpe(value: number | null | undefined): string {
  if (value == null) return "--";
  return value.toFixed(2);
}

/** Format ratio to 2 decimal places */
export function formatRatio(value: number | null | undefined): string {
  if (value == null) return "--";
  return value.toFixed(2);
}

/** Format number with commas (Indian system) */
export function formatNumber(value: number, decimals: number = 0): string {
  return new Intl.NumberFormat("en-IN", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

/** Format Greeks value (small numbers with 4 decimals) */
export function formatGreek(value: number): string {
  if (Math.abs(value) < 0.001) {
    return value.toExponential(2);
  }
  return value.toFixed(4);
}

/** Format timestamp to readable time: "14:35:22" */
export function formatTime(isoString: string): string {
  try {
    const d = new Date(isoString);
    return d.toLocaleTimeString("en-IN", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  } catch {
    return "--:--:--";
  }
}

/** Format timestamp to date: "15 Jan 2025" */
export function formatDate(isoString: string): string {
  try {
    const d = new Date(isoString);
    return d.toLocaleDateString("en-IN", {
      day: "numeric",
      month: "short",
      year: "numeric",
    });
  } catch {
    return "--";
  }
}

/** Format timestamp to compact datetime: "15 Jan 14:35" */
export function formatDateTime(isoString: string): string {
  try {
    const d = new Date(isoString);
    return `${d.toLocaleDateString("en-IN", { day: "numeric", month: "short" })} ${d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: false })}`;
  } catch {
    return "--";
  }
}

/** Return CSS class for PnL coloring */
export function pnlColor(value: number | null | undefined): string {
  if (value == null) return "text-gray-400";
  if (value > 0) return "text-profit";
  if (value < 0) return "text-loss";
  return "text-gray-400";
}

/** Return CSS class for PnL background coloring */
export function pnlBg(value: number | null | undefined): string {
  if (value == null) return "bg-gray-800";
  if (value > 0) return "bg-profit/10";
  if (value < 0) return "bg-loss/10";
  return "bg-gray-800";
}

/** Format PnL with sign and color-ready: "+12,345" */
export function formatPnl(value: number | null | undefined): string {
  if (value == null) return "--";
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatCurrency(value)}`;
}
