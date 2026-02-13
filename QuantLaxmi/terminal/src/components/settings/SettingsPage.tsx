import { useState, useEffect, useCallback } from "react";
import * as Tabs from "@radix-ui/react-tabs";
import {
  CheckCircle,
  XCircle,
  Save,
  RefreshCw,
} from "lucide-react";
import { apiFetch } from "@/lib/api";
import { cn } from "@/lib/utils";
import { SecretInput } from "./SecretInput";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface BrokerSettings {
  zerodha_user_id: string;
  zerodha_api_key: string;
  zerodha_api_secret: string;
  binance_api_key: string;
  binance_api_secret: string;
  _zerodha_configured: boolean;
  _binance_configured: boolean;
}

interface TelegramSettings {
  telegram_api_id: string;
  telegram_api_hash: string;
  telegram_phone: string;
  _configured: boolean;
}

interface SystemSettings {
  mode: string;
  data_dir: string;
}

interface AllSettings {
  broker: BrokerSettings;
  telegram: TelegramSettings;
  system: SystemSettings;
}

interface TestResult {
  status: "ok" | "error";
  message: string;
}

// ---------------------------------------------------------------------------
// Status Badge
// ---------------------------------------------------------------------------

function StatusBadge({ configured }: { configured: boolean }) {
  return configured ? (
    <span className="inline-flex items-center gap-1 text-xs font-medium text-terminal-profit">
      <CheckCircle className="w-3.5 h-3.5" />
      Configured
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 text-xs font-medium text-terminal-warning">
      <XCircle className="w-3.5 h-3.5" />
      Not configured
    </span>
  );
}

// ---------------------------------------------------------------------------
// Inline Toast
// ---------------------------------------------------------------------------

function InlineToast({ result }: { result: TestResult | null }) {
  if (!result) return null;
  const isOk = result.status === "ok";
  return (
    <div
      className={cn(
        "mt-3 px-3 py-2 rounded-md text-xs border",
        isOk
          ? "bg-terminal-profit/10 border-terminal-profit/30 text-terminal-profit"
          : "bg-terminal-loss/10 border-terminal-loss/30 text-terminal-loss",
      )}
    >
      <span className="font-medium">{isOk ? "Success" : "Error"}:</span>{" "}
      {result.message}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Plain Input (non-secret)
// ---------------------------------------------------------------------------

function PlainInput({
  label,
  value,
  onChange,
  placeholder,
  readOnly,
}: {
  label: string;
  value: string;
  onChange?: (v: string) => void;
  placeholder?: string;
  readOnly?: boolean;
}) {
  return (
    <div className="space-y-1.5">
      <label className="text-xs font-medium text-terminal-muted uppercase tracking-wider">
        {label}
      </label>
      <input
        type="text"
        value={value}
        onChange={onChange ? (e) => onChange(e.target.value) : undefined}
        placeholder={placeholder}
        readOnly={readOnly}
        className={cn(
          "w-full px-3 py-2 text-sm font-mono rounded-md",
          "bg-terminal-panel border border-terminal-border",
          "text-terminal-text placeholder:text-terminal-muted/50",
          "focus:outline-none focus:ring-1 focus:ring-terminal-accent focus:border-terminal-accent",
          "transition-colors",
          readOnly && "opacity-60 cursor-not-allowed",
        )}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function SettingsPage() {
  const [draft, setDraft] = useState<AllSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<Record<string, TestResult | null>>({});
  const [saveError, setSaveError] = useState<string | null>(null);

  // Fetch settings on mount
  const fetchSettings = useCallback(async () => {
    try {
      setLoading(true);
      const data = await apiFetch<AllSettings>("/api/settings");
      setDraft(data);
    } catch {
      // API may be offline — leave empty
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSettings();
  }, [fetchSettings]);

  // Save handler
  const handleSave = async () => {
    if (!draft) return;
    setSaving(true);
    setSaveError(null);
    try {
      const data = await apiFetch<AllSettings>("/api/settings", {
        method: "PUT",
        body: JSON.stringify({
          broker: {
            zerodha_user_id: draft.broker.zerodha_user_id,
            zerodha_api_key: draft.broker.zerodha_api_key,
            zerodha_api_secret: draft.broker.zerodha_api_secret,
            binance_api_key: draft.broker.binance_api_key,
            binance_api_secret: draft.broker.binance_api_secret,
          },
          telegram: {
            telegram_api_id: draft.telegram.telegram_api_id,
            telegram_api_hash: draft.telegram.telegram_api_hash,
            telegram_phone: draft.telegram.telegram_phone,
          },
        }),
      });
      setDraft(data);
    } catch {
      setSaveError("Failed to save settings. Is the API running?");
    } finally {
      setSaving(false);
    }
  };

  // Test connection handler
  const handleTest = async (provider: string) => {
    setTesting(provider);
    setTestResult((prev) => ({ ...prev, [provider]: null }));
    try {
      const result = await apiFetch<TestResult>(
        `/api/settings/test/${provider}`,
        { method: "POST" },
      );
      setTestResult((prev) => ({ ...prev, [provider]: result }));
    } catch {
      setTestResult((prev) => ({
        ...prev,
        [provider]: { status: "error", message: "Failed to reach API." },
      }));
    } finally {
      setTesting(null);
    }
  };

  // Update a draft field
  const updateBroker = (key: keyof BrokerSettings, value: string) => {
    if (!draft) return;
    setDraft({
      ...draft,
      broker: { ...draft.broker, [key]: value },
    });
  };

  const updateTelegram = (key: keyof TelegramSettings, value: string) => {
    if (!draft) return;
    setDraft({
      ...draft,
      telegram: { ...draft.telegram, [key]: value },
    });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-terminal-muted text-sm">
        Loading settings...
      </div>
    );
  }

  if (!draft) {
    return (
      <div className="flex items-center justify-center h-full text-terminal-muted text-sm">
        Unable to load settings. Is the API running?
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto p-6">
      <div className="max-w-2xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-terminal-text">Settings</h1>
            <p className="text-xs text-terminal-muted mt-0.5">
              Configure broker connections, Telegram, and system preferences.
            </p>
          </div>
          <button
            onClick={handleSave}
            disabled={saving}
            className={cn(
              "inline-flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium",
              "bg-terminal-accent text-white",
              "hover:bg-terminal-accent-dim transition-colors",
              "disabled:opacity-50 disabled:cursor-not-allowed",
            )}
          >
            {saving ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Save className="w-4 h-4" />
            )}
            {saving ? "Saving..." : "Save"}
          </button>
        </div>

        {/* Save error */}
        {saveError && (
          <div className="flex items-center gap-2 px-3 py-2 rounded-md bg-terminal-loss/10 border border-terminal-loss/30 text-terminal-loss text-xs">
            {saveError}
          </div>
        )}

        {/* Tabs */}
        <Tabs.Root defaultValue="broker" className="panel">
          <Tabs.List className="flex border-b border-terminal-border bg-terminal-panel">
            <Tabs.Trigger
              value="broker"
              className={cn(
                "px-4 py-2.5 text-xs font-medium uppercase tracking-wider transition-colors",
                "text-terminal-muted hover:text-terminal-text",
                "border-b-2 border-transparent",
                "data-[state=active]:text-terminal-accent data-[state=active]:border-terminal-accent",
              )}
            >
              Broker Connections
            </Tabs.Trigger>
            <Tabs.Trigger
              value="telegram"
              className={cn(
                "px-4 py-2.5 text-xs font-medium uppercase tracking-wider transition-colors",
                "text-terminal-muted hover:text-terminal-text",
                "border-b-2 border-transparent",
                "data-[state=active]:text-terminal-accent data-[state=active]:border-terminal-accent",
              )}
            >
              Telegram
            </Tabs.Trigger>
            <Tabs.Trigger
              value="system"
              className={cn(
                "px-4 py-2.5 text-xs font-medium uppercase tracking-wider transition-colors",
                "text-terminal-muted hover:text-terminal-text",
                "border-b-2 border-transparent",
                "data-[state=active]:text-terminal-accent data-[state=active]:border-terminal-accent",
              )}
            >
              System
            </Tabs.Trigger>
          </Tabs.List>

          {/* ---- Broker Tab ---- */}
          <Tabs.Content value="broker" className="p-5 space-y-6">
            {/* Zerodha */}
            <section className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-terminal-text">Zerodha</h2>
                <StatusBadge configured={draft.broker._zerodha_configured} />
              </div>
              <div className="grid gap-3">
                <PlainInput
                  label="User ID"
                  value={draft.broker.zerodha_user_id}
                  onChange={(v) => updateBroker("zerodha_user_id", v)}
                  placeholder="e.g. AB1234"
                />
                <SecretInput
                  label="API Key"
                  value={draft.broker.zerodha_api_key}
                  onChange={(v) => updateBroker("zerodha_api_key", v)}
                  placeholder="Your Zerodha API key"
                />
                <SecretInput
                  label="API Secret"
                  value={draft.broker.zerodha_api_secret}
                  onChange={(v) => updateBroker("zerodha_api_secret", v)}
                  placeholder="Your Zerodha API secret"
                />
              </div>
              <button
                onClick={() => handleTest("zerodha")}
                disabled={testing === "zerodha"}
                className={cn(
                  "inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium",
                  "border border-terminal-border hover:border-terminal-border-bright",
                  "text-terminal-text-secondary hover:text-terminal-text transition-colors",
                  "disabled:opacity-50 disabled:cursor-not-allowed",
                )}
              >
                {testing === "zerodha" ? (
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <RefreshCw className="w-3.5 h-3.5" />
                )}
                Test Connection
              </button>
              <InlineToast result={testResult.zerodha ?? null} />
            </section>

            <hr className="border-terminal-border" />

            {/* Binance */}
            <section className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-terminal-text">Binance</h2>
                <StatusBadge configured={draft.broker._binance_configured} />
              </div>
              <div className="grid gap-3">
                <SecretInput
                  label="API Key"
                  value={draft.broker.binance_api_key}
                  onChange={(v) => updateBroker("binance_api_key", v)}
                  placeholder="Your Binance API key"
                />
                <SecretInput
                  label="API Secret"
                  value={draft.broker.binance_api_secret}
                  onChange={(v) => updateBroker("binance_api_secret", v)}
                  placeholder="Your Binance API secret"
                />
              </div>
              <button
                onClick={() => handleTest("binance")}
                disabled={testing === "binance"}
                className={cn(
                  "inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium",
                  "border border-terminal-border hover:border-terminal-border-bright",
                  "text-terminal-text-secondary hover:text-terminal-text transition-colors",
                  "disabled:opacity-50 disabled:cursor-not-allowed",
                )}
              >
                {testing === "binance" ? (
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <RefreshCw className="w-3.5 h-3.5" />
                )}
                Test Connection
              </button>
              <InlineToast result={testResult.binance ?? null} />
            </section>
          </Tabs.Content>

          {/* ---- Telegram Tab ---- */}
          <Tabs.Content value="telegram" className="p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-terminal-text">Telegram</h2>
              <StatusBadge configured={draft.telegram._configured} />
            </div>
            <div className="grid gap-3">
              <PlainInput
                label="API ID"
                value={draft.telegram.telegram_api_id}
                onChange={(v) => updateTelegram("telegram_api_id", v)}
                placeholder="e.g. 12345678"
              />
              <SecretInput
                label="API Hash"
                value={draft.telegram.telegram_api_hash}
                onChange={(v) => updateTelegram("telegram_api_hash", v)}
                placeholder="Your Telegram API hash"
              />
              <PlainInput
                label="Phone Number"
                value={draft.telegram.telegram_phone}
                onChange={(v) => updateTelegram("telegram_phone", v)}
                placeholder="e.g. +91XXXXXXXXXX"
              />
            </div>
            <button
              onClick={() => handleTest("telegram")}
              disabled={testing === "telegram"}
              className={cn(
                "inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium",
                "border border-terminal-border hover:border-terminal-border-bright",
                "text-terminal-text-secondary hover:text-terminal-text transition-colors",
                "disabled:opacity-50 disabled:cursor-not-allowed",
              )}
            >
              {testing === "telegram" ? (
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <RefreshCw className="w-3.5 h-3.5" />
              )}
              Test Connection
            </button>
            <InlineToast result={testResult.telegram ?? null} />
          </Tabs.Content>

          {/* ---- System Tab ---- */}
          <Tabs.Content value="system" className="p-5 space-y-4">
            <h2 className="text-sm font-semibold text-terminal-text">System</h2>
            <div className="grid gap-3">
              <PlainInput
                label="Mode"
                value={draft.system.mode}
                readOnly
              />
              <PlainInput
                label="Data Directory"
                value={draft.system.data_dir}
                readOnly
              />
            </div>
          </Tabs.Content>
        </Tabs.Root>
      </div>
    </div>
  );
}
