"use client";
import { useCallback, useState } from "react";

const PHASES = ["Landmarking", "rPPG Extraction", "SyncNet Alignment", "Quality Fusion"];

export default function DemoDropzone() {
  const [state, setState] = useState<"idle" | "running" | "done">("idle");
  const [phase, setPhase] = useState(0);
  const [prog, setProg] = useState(0);
  const [over, setOver] = useState(false);

  const run = useCallback(() => {
    setState("running");
    setProg(0);
    setPhase(0);
    let p = 0;
    const id = setInterval(() => {
      p += Math.random() * 7 + 3;
      if (p >= 100) {
        p = 100;
        clearInterval(id);
        setTimeout(() => setState("done"), 400);
      }
      setProg(p);
      setPhase(Math.min(3, Math.floor(p / 25)));
    }, 180);
  }, []);

  const Gauge = ({ val, label, color }: { val: number; label: string; color: string }) => {
    const r = 40;
    const c = 2 * Math.PI * r;
    const off = c - (val / 100) * c;
    return (
      <div className="flex flex-col items-center gap-2">
        <svg width="100" height="100" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r={r} fill="none" stroke="rgba(255,255,255,.06)" strokeWidth="7" />
          <circle
            cx="50"
            cy="50"
            r={r}
            fill="none"
            stroke={color}
            strokeWidth="7"
            strokeLinecap="round"
            strokeDasharray={c}
            strokeDashoffset={off}
            transform="rotate(-90 50 50)"
            style={{ transition: "stroke-dashoffset 1s cubic-bezier(0.32,0.72,0,1)" }}
          />
          <text x="50" y="55" textAnchor="middle" fill="#fff" fontSize="18" fontWeight="600">
            {val}%
          </text>
        </svg>
        <span className="text-[10px] font-mono tracking-widest text-ash uppercase">{label}</span>
      </div>
    );
  };

  return (
    <section id="live-demo" className="px-4 py-32 md:py-48">
      <div className="mx-auto max-w-4xl text-center">
        <span className="eyebrow">Live Inference Sandbox</span>
        <h2 className="mt-6 font-display font-semibold tracking-tight text-4xl md:text-6xl">
          Run the engine on your own footage.
        </h2>

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            run();
          }}
          onClick={() => state === "idle" && run()}
          className={`mt-16 bezel-shell cursor-pointer transition-all duration-700 ease-fluid ${over ? "scale-[1.01]" : ""}`}
        >
          <div className="bezel-core min-h-[360px] flex flex-col items-center justify-center gap-6 border border-dashed border-white/15">
            {state === "idle" && (
              <>
                <div className="w-16 h-16 rounded-full bg-volt/10 border border-volt/30 flex items-center justify-center text-volt text-2xl">
                  ⇪
                </div>
                <p className="font-display text-xl">
                  Drop a <span className="text-volt">.mp4</span> / <span className="text-volt">.mov</span> to begin verification
                </p>
                <p className="text-xs text-ash font-mono">PROCESSED LOCALLY · ZERO RETENTION</p>
              </>
            )}

            {state === "running" && (
              <>
                <p className="font-mono text-sm text-volt tracking-widest">{PHASES[phase].toUpperCase()}</p>
                <div className="w-full max-w-md h-1.5 rounded-full bg-white/5 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-volt to-pulse rounded-full"
                    style={{ width: `${prog}%`, transition: "width .3s cubic-bezier(0.32,0.72,0,1)" }}
                  />
                </div>
                <div className="flex gap-3 flex-wrap justify-center">
                  {PHASES.map((p, i) => (
                    <span
                      key={p}
                      className={`px-3 py-1 rounded-full border font-mono text-[10px] ${
                        i <= phase ? "bg-volt/10 border-volt/40 text-volt" : "bg-white/[0.02] border-white/10 text-ash"
                      }`}
                    >
                      {p}
                    </span>
                  ))}
                </div>
              </>
            )}

            {state === "done" && (
              <>
                <span className="px-4 py-1.5 rounded-full bg-pulse/10 border border-pulse/40 text-pulse font-mono text-sm">
                  VERDICT: LIKELY_AUTHENTIC
                </span>
                <div className="flex flex-wrap justify-center gap-8 mt-2">
                  <Gauge val={94} label="rPPG Coherence" color="#00F2FE" />
                  <Gauge val={88} label="Lip-Sync Integrity" color="#10B981" />
                  <Gauge val={91} label="Ocular Physics" color="#7928CA" />
                </div>
                <p className="max-w-md text-sm text-ash leading-relaxed">
                  Strong bilateral pulse agreement and sub-25ms audio-visual lag. No graft seams detected across 30 facial patches.
                </p>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setState("idle");
                  }}
                  className="text-xs font-mono text-volt underline underline-offset-4"
                >
                  Run another scan ↺
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
