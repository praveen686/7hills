import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        terminal: {
          bg: "#08080d",
          surface: "#0f0f17",
          panel: "#13131d",
          border: "#1e1e2e",
          "border-bright": "#2a2a3e",
          muted: "#6b6b8a",
          accent: "#4f8eff",
          "accent-dim": "#3a6acc",
          profit: "#00d4aa",
          "profit-dim": "#00a882",
          loss: "#ff4d6a",
          "loss-dim": "#cc3d55",
          warning: "#ffb84d",
          info: "#4fc3f7",
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
