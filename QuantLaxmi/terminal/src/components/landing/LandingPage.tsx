import { useState, useRef } from "react";
import { useSetAtom } from "jotai";
import { ArrowRight, Brain, Shield, Layers } from "lucide-react";
import { ThemeToggle } from "@/components/terminal/ThemeToggle";
import { LoginButtons } from "@/components/landing/LoginButtons";
import { pageAtom } from "@/stores/auth";
import { apiFetch } from "@/lib/api";

export function LandingPage() {
  const setPage = useSetAtom(pageAtom);
  const [email, setEmail] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [videoFailed, setVideoFailed] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  const handleWaitlist = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;
    try {
      await apiFetch("/api/auth/waitlist", {
        method: "POST",
        body: JSON.stringify({ email: email.trim() }),
      });
      setSubmitted(true);
    } catch {
      // Silently handle — could add toast later
      setSubmitted(true);
    }
  };

  return (
    <div className="min-h-screen bg-terminal-bg text-terminal-text flex flex-col">
      {/* ─── Nav ─── */}
      <nav className="flex items-center justify-between px-6 py-4 border-b border-terminal-border bg-terminal-surface/80 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-terminal-accent flex items-center justify-center">
            <span className="text-white font-bold text-sm">Q</span>
          </div>
          <span className="font-semibold text-lg tracking-tight">QuantLaxmi</span>
        </div>
        <div className="flex items-center gap-4">
          <ThemeToggle size="md" />
          <button
            onClick={() => setPage("terminal")}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-terminal-accent text-white text-sm font-medium hover:bg-terminal-accent-dim transition-colors"
          >
            Enter Terminal
            <ArrowRight size={14} />
          </button>
        </div>
      </nav>

      {/* ─── Hero ─── */}
      <section className="relative flex flex-col items-center justify-center min-h-[80vh] overflow-hidden">
        {/* Video background */}
        {!videoFailed && (
          <video
            ref={videoRef}
            autoPlay
            loop
            muted
            playsInline
            className="absolute inset-0 w-full h-full object-cover opacity-30"
            onError={() => setVideoFailed(true)}
          >
            <source src="/landing/hero.mp4" type="video/mp4" />
          </video>
        )}

        {/* Animated gradient fallback */}
        <div
          className={`absolute inset-0 ${videoFailed ? "opacity-40" : "opacity-0"}`}
          style={{
            background: "linear-gradient(135deg, rgb(var(--terminal-accent)) 0%, rgb(var(--terminal-profit)) 50%, rgb(var(--terminal-accent-dim)) 100%)",
            backgroundSize: "400% 400%",
            animation: "gradientShift 8s ease infinite",
          }}
        />

        {/* Overlay for readability */}
        <div className="absolute inset-0 bg-terminal-bg/60" />

        {/* Content */}
        <div className="relative z-10 flex flex-col items-center text-center px-4 max-w-3xl">
          <h1 className="text-5xl md:text-6xl font-bold tracking-tight mb-4 leading-tight">
            Institutional-Grade
            <br />
            <span className="text-terminal-accent">Trading Intelligence</span>
          </h1>
          <p className="text-lg md:text-xl text-terminal-text-secondary max-w-xl mb-10">
            25+ strategies. Real-time risk. ML-powered signals. Built for India FnO.
          </p>

          {/* Waitlist form */}
          {!submitted ? (
            <form onSubmit={handleWaitlist} className="flex gap-2 w-full max-w-md">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Enter your email for early access"
                className="flex-1 px-4 py-3 rounded-lg bg-terminal-surface border border-terminal-border text-terminal-text placeholder:text-terminal-muted focus:outline-none focus:border-terminal-accent transition-colors text-sm"
                required
              />
              <button
                type="submit"
                className="px-6 py-3 rounded-lg bg-terminal-accent text-white font-semibold text-sm hover:bg-terminal-accent-dim transition-colors whitespace-nowrap"
              >
                Join Waitlist
              </button>
            </form>
          ) : (
            <div className="px-6 py-3 rounded-lg bg-terminal-profit/20 text-terminal-profit border border-terminal-profit/30 text-sm font-medium">
              You're on the list! We'll be in touch.
            </div>
          )}
        </div>
      </section>

      {/* ─── Login Section ─── */}
      <section className="py-16 px-6 flex flex-col items-center border-t border-terminal-border">
        <h2 className="text-2xl font-bold mb-2">Get Started</h2>
        <p className="text-terminal-text-secondary mb-8 text-sm">Connect your broker or sign in to continue</p>
        <LoginButtons />
      </section>

      {/* ─── Features ─── */}
      <section className="py-16 px-6 border-t border-terminal-border bg-terminal-surface">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-2xl font-bold text-center mb-12">Why QuantLaxmi?</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <FeatureCard
              icon={<Layers size={24} className="text-terminal-accent" />}
              title="25+ Strategies"
              description="ML, RL, microstructure, statistical, and options strategies — from Hawkes processes to Temporal Fusion Transformers."
            />
            <FeatureCard
              icon={<Shield size={24} className="text-terminal-profit" />}
              title="Real-Time Risk"
              description="Live drawdown monitoring, VaR, exposure heatmaps, and automated position sizing with risk gates."
            />
            <FeatureCard
              icon={<Brain size={24} className="text-terminal-info" />}
              title="TFT-Powered"
              description="378-feature Temporal Fusion Transformer with walk-forward validation. Sharpe 1.88+ out-of-sample."
            />
          </div>
        </div>
      </section>

      {/* ─── Footer ─── */}
      <footer className="py-8 px-6 text-center text-terminal-muted text-xs border-t border-terminal-border">
        <span>QuantLaxmi &copy; 2026. All rights reserved.</span>
      </footer>

      {/* Gradient animation keyframes */}
      <style>{`
        @keyframes gradientShift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
      `}</style>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="flex flex-col gap-3 p-6 rounded-xl bg-terminal-panel border border-terminal-border hover:border-terminal-border-bright transition-colors">
      <div className="w-10 h-10 rounded-lg bg-terminal-surface flex items-center justify-center">
        {icon}
      </div>
      <h3 className="font-semibold text-lg">{title}</h3>
      <p className="text-sm text-terminal-text-secondary leading-relaxed">{description}</p>
    </div>
  );
}
