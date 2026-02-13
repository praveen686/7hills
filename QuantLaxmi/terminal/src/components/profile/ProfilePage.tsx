import { useAtom } from "jotai";
import { useSetAtom } from "jotai";
import {
  LogOut,
  Sun,
  Moon,
  Shield,
  Mail,
  Globe,
  CheckCircle2,
  XCircle,
  LayoutDashboard,
} from "lucide-react";

import { useAuth } from "@/hooks/useAuth";
import { themeAtom } from "@/stores/workspace";
import { pageAtom, type User } from "@/stores/auth";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const PROVIDER_COLORS: Record<User["provider"], string> = {
  zerodha: "bg-blue-600",
  binance: "bg-yellow-500",
  google: "bg-white dark:bg-terminal-muted",
};

const PROVIDER_TEXT: Record<User["provider"], string> = {
  zerodha: "text-white",
  binance: "text-black",
  google: "text-terminal-text",
};

const PROVIDER_LABELS: Record<User["provider"], string> = {
  zerodha: "Zerodha",
  binance: "Binance",
  google: "Google",
};

function getInitials(name: string): string {
  const parts = name.trim().split(/\s+/);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return name.slice(0, 2).toUpperCase();
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function BoolBadge({ value, label }: { value: boolean | undefined; label: string }) {
  const yes = value === true;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 px-2 py-0.5 rounded text-2xs font-medium",
        yes
          ? "bg-terminal-profit/15 text-terminal-profit"
          : "bg-terminal-loss/15 text-terminal-loss",
      )}
    >
      {yes ? <CheckCircle2 size={10} /> : <XCircle size={10} />}
      {label}
    </span>
  );
}

function TagBadge({ children }: { children: string }) {
  return (
    <span className="inline-flex px-2 py-0.5 rounded bg-terminal-panel border border-terminal-border text-2xs font-mono text-terminal-text-secondary">
      {children}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Section: Header
// ---------------------------------------------------------------------------

function ProfileHeader({ user }: { user: User }) {
  return (
    <div className="panel">
      <div className="panel-body flex items-center gap-5">
        {/* Avatar */}
        {user.avatar ? (
          <img
            src={user.avatar}
            alt={user.name}
            className="w-16 h-16 rounded-full ring-2 ring-terminal-border object-cover"
          />
        ) : (
          <div
            className={cn(
              "w-16 h-16 rounded-full flex items-center justify-center text-xl font-bold select-none ring-2 ring-terminal-border",
              PROVIDER_COLORS[user.provider],
              PROVIDER_TEXT[user.provider],
            )}
          >
            {getInitials(user.name)}
          </div>
        )}

        <div className="flex flex-col gap-1.5">
          <h2 className="text-lg font-semibold text-terminal-text">{user.name}</h2>
          <p className="text-xs text-terminal-muted flex items-center gap-1.5">
            <Mail size={12} />
            {user.email}
          </p>
          <span
            className={cn(
              "inline-flex items-center gap-1 w-fit px-2 py-0.5 rounded text-2xs font-semibold uppercase tracking-wider",
              PROVIDER_COLORS[user.provider],
              PROVIDER_TEXT[user.provider],
            )}
          >
            <Shield size={10} />
            {PROVIDER_LABELS[user.provider]}
          </span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section: Connected Accounts
// ---------------------------------------------------------------------------

function ZerodhaCard({ user }: { user: User }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Zerodha</span>
        {user.broker && (
          <span className="text-2xs text-terminal-muted font-mono">{user.broker}</span>
        )}
      </div>
      <div className="panel-body space-y-3">
        {/* Exchanges */}
        {user.exchanges && user.exchanges.length > 0 && (
          <div>
            <p className="text-2xs font-medium text-terminal-muted uppercase tracking-wider mb-1.5">
              Exchanges
            </p>
            <div className="flex flex-wrap gap-1.5">
              {user.exchanges.map((ex) => (
                <TagBadge key={ex}>{ex}</TagBadge>
              ))}
            </div>
          </div>
        )}

        {/* Products */}
        {user.products && user.products.length > 0 && (
          <div>
            <p className="text-2xs font-medium text-terminal-muted uppercase tracking-wider mb-1.5">
              Products
            </p>
            <div className="flex flex-wrap gap-1.5">
              {user.products.map((p) => (
                <TagBadge key={p}>{p}</TagBadge>
              ))}
            </div>
          </div>
        )}

        {/* Order Types */}
        {user.order_types && user.order_types.length > 0 && (
          <div>
            <p className="text-2xs font-medium text-terminal-muted uppercase tracking-wider mb-1.5">
              Order Types
            </p>
            <div className="flex flex-wrap gap-1.5">
              {user.order_types.map((ot) => (
                <TagBadge key={ot}>{ot}</TagBadge>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function BinanceCard({ user }: { user: User }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Binance</span>
        {user.account_type && (
          <span className="text-2xs text-terminal-muted font-mono">{user.account_type}</span>
        )}
      </div>
      <div className="panel-body space-y-3">
        {/* Permissions */}
        {user.permissions && user.permissions.length > 0 && (
          <div>
            <p className="text-2xs font-medium text-terminal-muted uppercase tracking-wider mb-1.5">
              Permissions
            </p>
            <div className="flex flex-wrap gap-1.5">
              {user.permissions.map((perm) => (
                <TagBadge key={perm}>{perm}</TagBadge>
              ))}
            </div>
          </div>
        )}

        {/* Capabilities */}
        <div>
          <p className="text-2xs font-medium text-terminal-muted uppercase tracking-wider mb-1.5">
            Capabilities
          </p>
          <div className="flex flex-wrap gap-1.5">
            <BoolBadge value={user.can_trade} label="Trade" />
            <BoolBadge value={user.can_withdraw} label="Withdraw" />
            <BoolBadge value={user.can_deposit} label="Deposit" />
          </div>
        </div>
      </div>
    </div>
  );
}

function GoogleCard({ user }: { user: User }) {
  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Google</span>
      </div>
      <div className="panel-body flex items-center gap-3">
        {user.avatar ? (
          <img
            src={user.avatar}
            alt={user.name}
            className="w-8 h-8 rounded-full object-cover"
          />
        ) : (
          <Globe size={20} className="text-terminal-muted" />
        )}
        <div>
          <p className="text-sm font-medium text-terminal-text">{user.name}</p>
          <p className="text-xs text-terminal-muted">{user.email}</p>
        </div>
      </div>
    </div>
  );
}

function ConnectedAccounts({ user }: { user: User }) {
  return (
    <div className="space-y-3">
      <h3 className="text-xs font-medium text-terminal-muted uppercase tracking-wider">
        Connected Accounts
      </h3>
      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        {user.provider === "zerodha" && <ZerodhaCard user={user} />}
        {user.provider === "binance" && <BinanceCard user={user} />}
        {user.provider === "google" && <GoogleCard user={user} />}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section: Session
// ---------------------------------------------------------------------------

function SessionSection({ onLogout }: { onLogout: () => void }) {
  return (
    <div className="space-y-3">
      <h3 className="text-xs font-medium text-terminal-muted uppercase tracking-wider">
        Session
      </h3>
      <div className="panel">
        <div className="panel-body flex items-center justify-between">
          <div>
            <p className="text-sm text-terminal-text">Sign out of your account</p>
            <p className="text-xs text-terminal-muted mt-0.5">
              You will be redirected to the landing page.
            </p>
          </div>
          <button
            onClick={onLogout}
            className={cn(
              "btn-neutral flex items-center gap-1.5",
              "hover:!bg-terminal-loss/15 hover:!text-terminal-loss hover:!border-terminal-loss/30",
            )}
          >
            <LogOut size={13} />
            Sign Out
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section: Preferences
// ---------------------------------------------------------------------------

function PreferencesSection() {
  const [theme, setTheme] = useAtom(themeAtom);
  const setPage = useSetAtom(pageAtom);

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark");
  };

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-medium text-terminal-muted uppercase tracking-wider">
        Preferences
      </h3>
      <div className="grid gap-3 md:grid-cols-2">
        {/* Theme toggle */}
        <div className="metric-card flex items-center justify-between">
          <div className="flex items-center gap-3">
            {theme === "dark" ? (
              <Moon size={16} className="text-terminal-accent" />
            ) : (
              <Sun size={16} className="text-terminal-warning" />
            )}
            <div>
              <p className="text-sm font-medium text-terminal-text">Theme</p>
              <p className="text-xs text-terminal-muted capitalize">{theme} mode</p>
            </div>
          </div>
          <button
            onClick={toggleTheme}
            className={cn(
              "relative inline-flex h-6 w-11 items-center rounded-full transition-colors",
              theme === "dark" ? "bg-terminal-accent" : "bg-terminal-border-bright",
            )}
          >
            <span
              className={cn(
                "inline-block h-4 w-4 rounded-full bg-white transition-transform",
                theme === "dark" ? "translate-x-6" : "translate-x-1",
              )}
            />
          </button>
        </div>

        {/* Default landing page */}
        <div className="metric-card flex items-center justify-between">
          <div className="flex items-center gap-3">
            <LayoutDashboard size={16} className="text-terminal-accent" />
            <div>
              <p className="text-sm font-medium text-terminal-text">Default Page</p>
              <p className="text-xs text-terminal-muted">Page shown after login</p>
            </div>
          </div>
          <button
            onClick={() => setPage("dashboard")}
            className="btn-neutral"
          >
            Dashboard
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

export default function ProfilePage() {
  const { user, logout } = useAuth();

  if (!user) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="panel max-w-sm w-full">
          <div className="panel-body text-center py-8">
            <Shield size={32} className="mx-auto text-terminal-muted mb-3" />
            <p className="text-sm text-terminal-muted">
              Please sign in to view your profile.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-3xl mx-auto p-6 space-y-6">
        <ProfileHeader user={user} />
        <ConnectedAccounts user={user} />
        <PreferencesSection />
        <SessionSection onLogout={logout} />
      </div>
    </div>
  );
}
