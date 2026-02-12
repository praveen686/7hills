import { atom } from "jotai";

/** Application mode: live trading or backtesting with historical data. */
export type AppMode = "live" | "backtest";

/** Current application mode. */
export const appModeAtom = atom<AppMode>("backtest");

/** Date for backtest mode (YYYY-MM-DD). */
export const backtestDateAtom = atom<string>("");
