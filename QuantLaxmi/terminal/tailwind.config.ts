import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        terminal: {
          bg: "rgb(var(--terminal-bg) / <alpha-value>)",
          surface: "rgb(var(--terminal-surface) / <alpha-value>)",
          panel: "rgb(var(--terminal-panel) / <alpha-value>)",
          border: "rgb(var(--terminal-border) / <alpha-value>)",
          "border-bright": "rgb(var(--terminal-border-bright) / <alpha-value>)",
          muted: "rgb(var(--terminal-muted) / <alpha-value>)",
          accent: "rgb(var(--terminal-accent) / <alpha-value>)",
          "accent-dim": "rgb(var(--terminal-accent-dim) / <alpha-value>)",
          profit: "rgb(var(--terminal-profit) / <alpha-value>)",
          "profit-dim": "rgb(var(--terminal-profit-dim) / <alpha-value>)",
          loss: "rgb(var(--terminal-loss) / <alpha-value>)",
          "loss-dim": "rgb(var(--terminal-loss-dim) / <alpha-value>)",
          warning: "rgb(var(--terminal-warning) / <alpha-value>)",
          info: "rgb(var(--terminal-info) / <alpha-value>)",
          text: "rgb(var(--terminal-text) / <alpha-value>)",
          "text-secondary": "rgb(var(--terminal-text-secondary) / <alpha-value>)",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      fontSize: {
        "2xs": ["0.625rem", { lineHeight: "0.875rem" }],
      },
      animation: {
        "flash-green": "flash-green 0.3s ease-out",
        "flash-red": "flash-red 0.3s ease-out",
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
      keyframes: {
        "flash-green": {
          "0%": { backgroundColor: "rgba(0, 212, 170, 0.3)" },
          "100%": { backgroundColor: "transparent" },
        },
        "flash-red": {
          "0%": { backgroundColor: "rgba(255, 77, 106, 0.3)" },
          "100%": { backgroundColor: "transparent" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
