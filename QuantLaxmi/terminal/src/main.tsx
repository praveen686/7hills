import React from "react";
import ReactDOM from "react-dom/client";
import { Provider as JotaiProvider } from "jotai";
import App from "./App";
import "./styles/globals.css";

// Apply persisted theme class before first paint
const savedTheme = localStorage.getItem("ql-theme") ?? "dark";
document.documentElement.classList.toggle("dark", savedTheme === "dark");

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <JotaiProvider>
      <App />
    </JotaiProvider>
  </React.StrictMode>
);
