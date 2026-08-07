"use client";
import { useEffect, useRef, useState } from "react";
import { useGSAP } from "@gsap/react";
import { gsap } from "@/lib/gsap";
import PulseGrid from "./PulseGrid";

export default function BentoGrid() {
  const root = useRef<HTMLElement>(null);
  useGSAP(() => {
    // Respect reduced-motion + responsive via matchMedia
    const mm = gsap.matchMedia();
    mm.add("(prefers-reduced-motion: no-preference)", () => {
      gsap.from(".bento-card", {
        y: 60,
        autoAlpha: 0,
        filter: "blur(6px)",
        stagger: 0.14,
        duration: 1.1,
        ease: "vanguard",
        scrollTrigger: { trigger: root.current, start: "top 75%" },
      });
    });
    return () => mm.revert();
  }, { scope: root });

  return (
    <section ref={root} id="threat-matrix" className="relative px-4 py-32 md:py-48">
      <div className="mx-auto max-w-6xl">
        <span className="eyebrow">The Dual-Branch Engine</span>
        <h2 className="mt-6 font-display font-semibold tracking-tight text-4xl md:text-6xl max-w-3xl">
          Four independent biometric witnesses. One fused verdict.
        </h2>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-12 gap-6 grid-flow-dense auto-rows-[minmax(320px,auto)]">
          <Card1 className="md:col-span-8" />
          <Card2 className="md:col-span-4" />
          <Card3 className="md:col-span-4" />
          <Card4 className="md:col-span-8" />
        </div>
      </div>
    </section>
  );
}

function Shell({ className = "", children }: { className?: string; children: React.ReactNode }) {
  return (
    <div className={`bento-card bezel-shell transition-transform duration-700 ease-fluid hover:-translate-y-1 ${className}`}>
      <div className="bezel-core h-full flex flex-col overflow-hidden">{children}</div>
    </div>
  );
}

const Head = ({ tag, title, desc }: { tag: string; title: string; desc: string }) => (
  <div className="mb-6">
    <span className="text-[10px] font-mono tracking-[0.25em] text-volt uppercase">{tag}</span>
    <h3 className="mt-2 font-display font-semibold text-2xl tracking-tight">{title}</h3>
    <p className="mt-2 text-sm text-ash leading-relaxed">{desc}</p>
  </div>
);

/* Card 1 — Bilateral rPPG Pulse Coherence (8 cols) */
function Card1({ className }: { className?: string }) {
  return (
    <Shell className={className}>
      <Head
        tag="rPPG Engine"
        title="Bilateral Pulse Coherence"
        desc="Cheek & forehead capillary blood-flow phase agreement exposes hidden face-swap seams that pixel forensics miss."
      />
      <div className="relative flex-1 rounded-2xl overflow-hidden border border-white/5 min-h-[220px] bg-zinc-900">
        <img
          src="https://images.unsplash.com/photo-1544005313-94ddf0286df2?q=80&w=1200&auto=format&fit=crop"
          alt="rPPG face stream"
          className="absolute inset-0 w-full h-full object-cover grayscale opacity-60"
        />
        <PulseGrid coherent seed={23} />
      </div>
    </Shell>
  );
}

/* Card 2 — Speech-to-Lip Lag IQR (4 cols) */
function Card2({ className }: { className?: string }) {
  const a = useRef<SVGPathElement>(null);
  const b = useRef<SVGPathElement>(null);

  const wave = () => `M0,40 ${Array.from({ length: 12 }).map((_, i) => `Q${i * 30 + 15},${i % 2 ? 10 : 70} ${(i + 1) * 30},40`).join(" ")}`;

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.to([a.current, b.current], {
        attr: { d: wave() },
        duration: 1.4,
        ease: "sine.inOut",
        repeat: -1,
        yoyo: true,
        stagger: 0.18,
      });
    });
    return () => ctx.revert();
  }, []);

  return (
    <Shell className={className}>
      <Head
        tag="Sync Discriminator"
        title="Speech-to-Lip Lag IQR"
        desc="Acoustic envelope vs mouth-aspect-ratio derivative. Drift beyond the IQR window flags dubbing."
      />
      <div className="flex-1 rounded-2xl border border-white/5 p-4 flex flex-col justify-center gap-4 bg-black/20">
        <svg viewBox="0 0 360 80" className="w-full">
          <path ref={a} d={wave()} fill="none" stroke="#00F2FE" strokeWidth="2" />
        </svg>
        <svg viewBox="0 0 360 80" className="w-full">
          <path ref={b} d={wave()} fill="none" stroke="#10B981" strokeWidth="2" />
        </svg>
        <div className="flex justify-between font-mono text-[10px] text-ash">
          <span className="text-volt">AUDIO ENVELOPE</span>
          <span className="text-pulse">MAR DERIVATIVE</span>
        </div>
      </div>
    </Shell>
  );
}

/* Card 3 — Traditional pixel forensics (4 cols)
   Replaces a fabricated "Ocular Physics / Corneal Reflection Detector" card that
   showed EAR 0.31 / BLINK 19-per-min / REFLECT OK. No ocular channel exists; the
   blink counter was a GSAP animation incrementing a random number. This card
   shows the real fourth channel and the evidence its design came from. */
function Card3({ className }: { className?: string }) {
  // Share of 4,347 human deepfake annotations reporting each artifact (ExDDV).
  const evidence = [
    { label: "warping / distortion", pct: 54.1 },
    { label: "mouth / lips", pct: 39.7 },
    { label: "eyes", pct: 38.6 },
    { label: "blur / pixelation", pct: 12.7 },
    { label: "shadow / lighting", pct: 11.9 },
  ];

  return (
    <Shell className={className}>
      <Head
        tag="Traditional Pixel Forensics"
        title="Region Texture & Warp Analysis"
        desc="Classical, no trained weights. Feature design derived from 4,347 human annotations of real deepfakes — not from assumption."
      />
      <div className="flex-1 rounded-2xl border border-white/5 bg-black/20 p-4 flex flex-col justify-center gap-2.5">
        <div className="font-mono text-[9px] text-ash tracking-widest uppercase mb-1">
          What annotators actually report
        </div>
        {evidence.map((e) => (
          <div key={e.label}>
            <div className="flex justify-between font-mono text-[10px] text-ash mb-1">
              <span>{e.label}</span>
              <span className="text-volt">{e.pct}%</span>
            </div>
            <div className="h-1 rounded-full bg-white/5 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-volt to-pulse"
                style={{ width: `${e.pct}%` }}
              />
            </div>
          </div>
        ))}
        <div className="mt-2 font-mono text-[9px] text-ash/70 leading-relaxed">
          Eyes and mouth are the top measured discriminators — matching the regions
          humans flag.
        </div>
      </div>
    </Shell>
  );
}

/* Card 4 — Quality-Weighted Fusion (8 cols) */
function Card4({ className }: { className?: string }) {
  const [q, setQ] = useState(72);
  const w = q / 100;
  const verdict = q < 30 ? "INSUFFICIENT_EVIDENCE" : q < 60 ? "UNCERTAIN" : "CONFIDENT";
  const tone =
    q < 30
      ? "text-red-300 border-red-500/40 bg-red-500/10"
      : q < 60
      ? "text-amber-300 border-amber-500/40 bg-amber-500/10"
      : "text-pulse border-pulse/40 bg-pulse/10";

  const Bar = ({ label, val, color }: { label: string; val: number; color: string }) => (
    <div>
      <div className="flex justify-between font-mono text-[10px] text-ash mb-1">
        <span>{label}</span>
        <span>{Math.round(val * 100)}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-fluid"
          style={{ width: `${val * 100}%`, background: color }}
        />
      </div>
    </div>
  );

  return (
    <Shell className={className}>
      <div className="flex flex-wrap items-start justify-between gap-4">
        <Head
          tag="Fusion Core"
          title="Quality-Weighted Dynamic Fusion"
          desc="Combines channels in log-odds so an uninformative channel contributes nothing. Drag the slider to see quality gate the contribution."
        />
        <span className={`px-3 py-1 rounded-full border font-mono text-xs ${tone}`}>{verdict}</span>
      </div>
      <div className="flex-1 grid md:grid-cols-2 gap-8 items-center">
        <div>
          <input type="range" min={0} max={100} value={q} onChange={(e) => setQ(+e.target.value)} className="w-full accent-cyan-400" />
          <p className="mt-3 font-mono text-xs text-ash">
            INPUT SIGNAL QUALITY · <span className="text-volt">{q}%</span>
          </p>
        </div>
        {/* Priors mirror config/thresholds.yaml. The previous bars used invented
            coefficients (0.9 / 0.72 / 0.55) and a channel that does not exist. */}
        <div className="space-y-4">
          <Bar label="rPPG PRIOR" val={0.3 * w} color="#00F2FE" />
          <Bar label="LIP-SYNC PRIOR" val={0.3 * w} color="#10B981" />
          <Bar label="FRAME-BY-FRAME PRIOR" val={0.4 * w} color="#7928CA" />
          <div className="pt-1 font-mono text-[9px] text-ash/70 leading-relaxed">
            Pixel forensics is calibrating and does not vote yet. A channel only
            enters fusion once it demonstrably improves the fused score.
          </div>
        </div>
      </div>
    </Shell>
  );
}
