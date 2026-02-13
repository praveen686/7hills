// ============================================================
// ParamEditor — dynamic parameter controls from strategy metadata
// ============================================================

import { useCallback } from "react";
import * as Switch from "@radix-ui/react-switch";
import * as Select from "@radix-ui/react-select";
import { cn } from "@/lib/utils";
import { ChevronDown, Check, RotateCcw, Save } from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ParamMeta {
  value: number | boolean | string;
  type: "int" | "float" | "bool" | "select";
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  description?: string;
}

export interface ParamEditorProps {
  params: Record<string, ParamMeta>;
  onChange: (key: string, value: number | boolean | string) => void;
  onSave: () => void;
  onReset: () => void;
}

// ---------------------------------------------------------------------------
// Individual controls
// ---------------------------------------------------------------------------

function NumericParam({
  name,
  meta,
  onChange,
}: {
  name: string;
  meta: ParamMeta;
  onChange: (key: string, value: number) => void;
}) {
  const step = meta.step ?? (meta.type === "int" ? 1 : 0.01);
  const min = meta.min ?? 0;
  const max = meta.max ?? 100;
  const value = typeof meta.value === "number" ? meta.value : Number(meta.value);

  const handleSlider = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const v = meta.type === "int" ? parseInt(e.target.value, 10) : parseFloat(e.target.value);
      onChange(name, v);
    },
    [name, meta.type, onChange],
  );

  const handleInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const raw = e.target.value;
      if (raw === "" || raw === "-") return;
      const v = meta.type === "int" ? parseInt(raw, 10) : parseFloat(raw);
      if (!isNaN(v)) onChange(name, v);
    },
    [name, meta.type, onChange],
  );

  return (
    <div className="flex items-center gap-3">
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={handleSlider}
        className="flex-1 h-1.5 appearance-none rounded-full bg-terminal-border accent-terminal-accent cursor-pointer
                   [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:h-3.5
                   [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-terminal-accent [&::-webkit-slider-thumb]:border-0
                   [&::-webkit-slider-thumb]:shadow-sm"
      />
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={handleInput}
        className="w-20 rounded border border-terminal-border bg-terminal-panel px-2 py-1 text-xs font-mono
                   text-terminal-text focus:outline-none focus:border-terminal-accent tabular-nums"
      />
    </div>
  );
}

function BoolParam({
  name,
  meta,
  onChange,
}: {
  name: string;
  meta: ParamMeta;
  onChange: (key: string, value: boolean) => void;
}) {
  const checked = Boolean(meta.value);
  return (
    <Switch.Root
      checked={checked}
      onCheckedChange={(val) => onChange(name, val)}
      className={cn(
        "relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border border-terminal-border transition-colors",
        checked ? "bg-terminal-accent" : "bg-terminal-panel",
      )}
    >
      <Switch.Thumb
        className={cn(
          "pointer-events-none block h-4 w-4 rounded-full bg-white shadow-sm transition-transform",
          checked ? "translate-x-4" : "translate-x-0",
        )}
      />
    </Switch.Root>
  );
}

function SelectParam({
  name,
  meta,
  onChange,
}: {
  name: string;
  meta: ParamMeta;
  onChange: (key: string, value: string) => void;
}) {
  const options = meta.options ?? [];
  const value = String(meta.value);

  return (
    <Select.Root value={value} onValueChange={(val) => onChange(name, val)}>
      <Select.Trigger
        className="inline-flex items-center justify-between gap-2 rounded border border-terminal-border
                   bg-terminal-panel px-2.5 py-1.5 text-xs font-mono text-terminal-text
                   hover:border-terminal-border-bright focus:outline-none focus:border-terminal-accent min-w-[140px]"
      >
        <Select.Value />
        <Select.Icon>
          <ChevronDown className="h-3 w-3 text-terminal-muted" />
        </Select.Icon>
      </Select.Trigger>
      <Select.Portal>
        <Select.Content
          className="overflow-hidden rounded-lg border border-terminal-border bg-terminal-surface shadow-lg z-50"
          position="popper"
          sideOffset={4}
        >
          <Select.Viewport className="p-1">
            {options.map((opt) => (
              <Select.Item
                key={opt}
                value={opt}
                className="relative flex items-center gap-2 rounded px-2 py-1.5 text-xs font-mono text-terminal-text
                           cursor-pointer select-none hover:bg-terminal-panel focus:bg-terminal-panel focus:outline-none
                           data-[highlighted]:bg-terminal-panel"
              >
                <Select.ItemIndicator className="absolute left-1">
                  <Check className="h-3 w-3 text-terminal-accent" />
                </Select.ItemIndicator>
                <Select.ItemText>{opt}</Select.ItemText>
              </Select.Item>
            ))}
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function ParamEditor({ params, onChange, onSave, onReset }: ParamEditorProps) {
  const keys = Object.keys(params);

  if (keys.length === 0) {
    return (
      <div className="text-xs text-terminal-muted italic py-4">
        No configurable parameters.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Parameter rows */}
      <div className="flex flex-col gap-3">
        {keys.map((key) => {
          const meta = params[key];
          return (
            <div key={key} className="flex flex-col gap-1">
              <div className="flex items-center justify-between">
                <label className="text-xs font-medium text-terminal-text">{key}</label>
                {meta.type !== "bool" && meta.type !== "select" && (
                  <span className="text-2xs font-mono text-terminal-muted tabular-nums">
                    {String(meta.value)}
                  </span>
                )}
              </div>
              {meta.description && (
                <p className="text-2xs text-terminal-muted leading-relaxed">{meta.description}</p>
              )}
              <div className="mt-0.5">
                {(meta.type === "int" || meta.type === "float") && (
                  <NumericParam name={key} meta={meta} onChange={onChange as (k: string, v: number) => void} />
                )}
                {meta.type === "bool" && (
                  <BoolParam name={key} meta={meta} onChange={onChange as (k: string, v: boolean) => void} />
                )}
                {meta.type === "select" && (
                  <SelectParam name={key} meta={meta} onChange={onChange as (k: string, v: string) => void} />
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-2 pt-2 border-t border-terminal-border">
        <button onClick={onReset} className="btn-neutral inline-flex items-center gap-1.5">
          <RotateCcw className="h-3 w-3" />
          Reset to Default
        </button>
        <button
          onClick={onSave}
          className="inline-flex items-center gap-1.5 rounded bg-terminal-accent px-3 py-1.5
                     text-xs font-medium text-white transition-colors hover:bg-terminal-accent-dim"
        >
          <Save className="h-3 w-3" />
          Save
        </button>
      </div>
    </div>
  );
}
