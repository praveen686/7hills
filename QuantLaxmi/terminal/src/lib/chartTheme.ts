/** Read a CSS custom property as an rgb() color string (comma-separated for library compat). */
function cssRgb(s: CSSStyleDeclaration, prop: string, fallback: string): string {
  const v = s.getPropertyValue(prop).trim();
  if (!v) return fallback;
  // chart-grid and chart-crosshair are stored as full color values (hex/rgba)
  if (v.startsWith("#") || v.startsWith("rgb")) return v;
  // terminal-* vars are stored as space-separated RGB channels: "8 8 13"
  // Use comma-separated format for lightweight-charts compatibility
  return `rgb(${v.split(/\s+/).join(", ")})`;
}

/** Apply alpha to an rgb() color string. Returns rgba(r, g, b, alpha). */
export function withAlpha(rgbColor: string, hexAlpha: string): string {
  const alpha = parseInt(hexAlpha, 16) / 255;
  // Extract the numbers from rgb(r, g, b) or rgb(r g b)
  const match = rgbColor.match(/(\d+)[,\s]+(\d+)[,\s]+(\d+)/);
  if (!match) return rgbColor;
  return `rgba(${match[1]}, ${match[2]}, ${match[3]}, ${alpha.toFixed(2)})`;
}

/** Read chart-relevant CSS custom properties from the current theme. */
export function getChartColors() {
  const s = getComputedStyle(document.documentElement);
  return {
    grid: cssRgb(s, "--chart-grid", "#1e1e2e"),
    crosshair: cssRgb(s, "--chart-crosshair", "rgba(79,142,255,0.27)"),
    text: cssRgb(s, "--terminal-muted", "rgb(107, 107, 138)"),
    border: cssRgb(s, "--terminal-border", "rgb(30, 30, 46)"),
    bg: cssRgb(s, "--terminal-bg", "rgb(8, 8, 13)"),
    surface: cssRgb(s, "--terminal-surface", "rgb(15, 15, 23)"),
    accent: cssRgb(s, "--terminal-accent", "rgb(79, 142, 255)"),
    profit: cssRgb(s, "--terminal-profit", "rgb(0, 212, 170)"),
    loss: cssRgb(s, "--terminal-loss", "rgb(255, 77, 106)"),
    warning: cssRgb(s, "--terminal-warning", "rgb(255, 184, 77)"),
  };
}
