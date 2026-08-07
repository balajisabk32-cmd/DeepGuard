"use client";
import { motion } from "motion/react";
import { FusionResultPayload } from "@/lib/api";
import ChannelPanel from "./ChannelPanel";

interface ForensicReportProps {
  result: FusionResultPayload;
  onReset: () => void;
}

const VERDICT = {
  LIKELY_AUTHENTIC: {
    label: "Likely Authentic",
    tone: "text-emerald-300",
    ring: "border-emerald-400/40 bg-emerald-400/[0.08]",
    dot: "bg-emerald-400 shadow-[0_0_10px_#10B981]",
  },
  LIKELY_MANIPULATED: {
    label: "Likely Manipulated",
    tone: "text-red-300",
    ring: "border-red-400/40 bg-red-400/[0.08]",
    dot: "bg-red-400 shadow-[0_0_10px_#EF4444]",
  },
  UNCERTAIN: {
    label: "Uncertain",
    tone: "text-amber-300",
    ring: "border-amber-400/40 bg-amber-400/[0.08]",
    dot: "bg-amber-400 shadow-[0_0_10px_#F59E0B]",
  },
  INSUFFICIENT_EVIDENCE: {
    label: "Insufficient Evidence",
    tone: "text-zinc-300",
    ring: "border-white/15 bg-white/[0.04]",
    dot: "bg-zinc-400",
  },
} as const;

function getSmoothSvgPath(
  data: number[] | undefined,
  width = 900,
  height = 140,
  padding = 10
): string {
  if (!data || data.length < 2) return "";
  const step = (width - padding * 2) / (data.length - 1);
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const pts = data.map((val, i) => {
    const norm = (val - min) / range;
    const y = height - padding - norm * (height - padding * 2);
    const x = padding + i * step;
    return { x, y };
  });

  let d = `M ${pts[0].x.toFixed(1)},${pts[0].y.toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i === 0 ? 0 : i - 1];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2] || p2;
    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${cp1x.toFixed(1)},${cp1y.toFixed(1)} ${cp2x.toFixed(1)},${cp2y.toFixed(1)} ${p2.x.toFixed(1)},${p2.y.toFixed(1)}`;
  }
  return d;
}

function getSmoothSvgArea(
  data: number[] | undefined,
  width = 900,
  height = 140,
  padding = 10
): string {
  const linePath = getSmoothSvgPath(data, width, height, padding);
  if (!linePath || !data) return "";
  const lastX = padding + (data.length - 1) * ((width - padding * 2) / (data.length - 1));
  const bottomY = height - padding;
  return `${linePath} L ${lastX.toFixed(1)},${bottomY} L ${padding},${bottomY} Z`;
}

function Section({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <motion.section
      variants={{
        hidden: { opacity: 0, y: 18 },
        show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] } },
      }}
      className="border-t border-white/[0.08] pt-10 md:pt-14"
    >
      <p className="font-mono text-[10px] tracking-[0.22em] text-volt uppercase">
        {eyebrow}
      </p>
      <h2 className="mt-2 font-display text-2xl md:text-3xl font-semibold tracking-tight text-bone">
        {title}
      </h2>
      <div className="mt-6">{children}</div>
    </motion.section>
  );
}

export default function ForensicReport({ result, onReset }: ForensicReportProps) {
  const v = VERDICT[result.verdict] ?? VERDICT.INSUFFICIENT_EVIDENCE;
  const fake = result.confidence_fake ?? 0;

  return (
    <div
      data-lenis-prevent
      className="fixed inset-0 z-50 overflow-y-auto overscroll-contain bg-void/95 backdrop-blur-3xl"
    >
      <button
        onClick={onReset}
        className="fixed top-6 right-6 z-50 w-11 h-11 rounded-full border border-white/10 bg-black/80 hover:bg-white/10 hover:border-white/30 text-ash hover:text-bone flex items-center justify-center transition-all backdrop-blur-md shadow-xl"
        aria-label="Close report"
      >
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </button>

      <motion.div
        initial="hidden"
        animate="show"
        variants={{ hidden: {}, show: { transition: { staggerChildren: 0.08 } } }}
        className="mx-auto w-full max-w-5xl px-6 md:px-12 py-20 md:py-28 space-y-12"
      >
        {/* ── Headline Banner ─────────────────────────────────────────────── */}
        <motion.header
          variants={{
            hidden: { opacity: 0, y: 18 },
            show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] } },
          }}
          className="space-y-6"
        >
          <div className="flex items-center gap-3">
            <span className="font-mono text-[10px] tracking-[0.25em] text-volt uppercase">
              Neural Forensic Ledger
            </span>
            <span className="w-1 h-1 rounded-full bg-white/30" />
            <span className="font-mono text-[10px] text-ash">VERDICT ENGINE v2.1</span>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className={`inline-flex items-center gap-3 rounded-full border px-4 py-2 ${v.ring}`}>
              <span className={`w-2 h-2 rounded-full ${v.dot}`} />
              <span className={`font-display text-sm font-semibold tracking-tight ${v.tone}`}>
                {v.label}
              </span>
            </div>

            <span className="font-mono text-xs text-ash">
              DECIMATION SAMPLE: 40 WINDOWED FRAMES
            </span>
          </div>

          {/* Probability Gauge & Metric */}
          <div className="pt-2">
            <div className="flex items-baseline gap-4">
              <span className="font-display text-7xl md:text-9xl font-bold tracking-tighter text-bone tabular-nums leading-none">
                {fake.toFixed(1)}
                <span className="text-3xl md:text-5xl text-ash/70 font-normal ml-1">%</span>
              </span>
              <div className="space-y-1">
                <p className="font-display text-lg font-medium text-bone">Estimated Manipulation Likelihood</p>
                <p className="font-body text-xs text-ash max-w-xs">
                  Calibrated quality-weighted posterior probability across all sensory channels.
                </p>
              </div>
            </div>

            <div className="mt-7 h-2.5 w-full rounded-full bg-white/[0.06] overflow-hidden p-0.5 border border-white/5 shadow-inner">
              <motion.div
                className={
                  result.verdict === "LIKELY_MANIPULATED"
                    ? "h-full bg-gradient-to-r from-amber-500 to-red-500 rounded-full shadow-[0_0_12px_rgba(239,68,68,0.5)]"
                    : "h-full bg-gradient-to-r from-cyan-400 to-pulse rounded-full shadow-[0_0_12px_rgba(16,185,129,0.5)]"
                }
                initial={{ width: 0 }}
                animate={{ width: `${fake}%` }}
                transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1], delay: 0.2 }}
              />
            </div>
          </div>

          <p className="font-body text-[16px] md:text-[17px] leading-relaxed text-ash max-w-3xl pt-2">
            {result.explanation}
          </p>
        </motion.header>

        {/* ── Evidence Channel Breakdown ──────────────────────────────────── */}
        <Section eyebrow="Evidence Weighting" title="Channel Arithmetic & Log-Odds Fusion">
          <ChannelPanel result={result} />
        </Section>

        {/* ── Pulse Coherence Map ─────────────────────────────────────────── */}
        <Section eyebrow="Channel 01 Telemetry" title="Spatial Blood-Volume Pulse Coherence">
          <p className="font-body text-[15px] leading-relaxed text-ash max-w-2xl mb-6">
            Each cell represents a facial ROI patch. Cross-region agreement drops dramatically across swapped boundaries.
          </p>

          {!result.map_coherence ? (
            <div className="rounded-2xl border border-white/[0.08] bg-white/[0.02] px-6 py-10 text-center">
              <p className="font-display text-base text-bone font-medium">Coherence Map Unavailable</p>
              <p className="mt-2 font-body text-sm text-ash max-w-md mx-auto leading-relaxed">
                Not enough usable facial skin signal in this stream. The biosignal extractor abstained rather than fabricating a pulse.
              </p>
            </div>
          ) : (
            <div className="rounded-2xl border border-white/10 bg-void/80 backdrop-blur-xl p-6 shadow-xl space-y-4">
              <div className="grid grid-cols-5 gap-2 max-w-xl mx-auto p-2 rounded-xl bg-black/60 border border-white/10 shadow-inner">
                {result.map_coherence.map((row, r) =>
                  row.map((val, c) => {
                    const ok = val !== null && val !== undefined && Number.isFinite(val);
                    let cls = "border-white/[0.06] bg-white/[0.02] text-ash/30";
                    if (ok) {
                      if (val >= 0.2) cls = "border-emerald-400/40 bg-emerald-400/15 text-emerald-300 font-semibold shadow-[0_0_8px_rgba(16,185,129,0.15)]";
                      else if (val >= -0.1) cls = "border-amber-400/40 bg-amber-400/15 text-amber-300 font-semibold";
                      else cls = "border-red-400/40 bg-red-400/15 text-red-300 font-semibold shadow-[0_0_8px_rgba(239,68,68,0.15)]";
                    }
                    return (
                      <div
                        key={`${r}-${c}`}
                        className={`aspect-[4/3] rounded-lg border flex items-center justify-center font-mono text-xs tabular-nums transition-all ${cls}`}
                      >
                        {ok ? val.toFixed(2) : "—"}
                      </div>
                    );
                  })
                )}
              </div>

              <div className="flex flex-wrap justify-center gap-6 font-mono text-[11px] text-ash pt-2">
                <span className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_6px_#10B981]" /> Coherent (≥ 0.20)
                </span>
                <span className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-amber-400" /> Marginal (-0.10 to 0.20)
                </span>
                <span className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-red-400 shadow-[0_0_6px_#EF4444]" /> Incoherent (&lt; -0.10)
                </span>
              </div>
            </div>
          )}
        </Section>

        {/* ── High-Def Lip-Sync Oscilloscope ──────────────────────────────── */}
        <Section eyebrow="Channel 02 Telemetry" title="Biomechanical Alignment: Speech vs. Mouth Aperture">
          <div className="rounded-2xl border border-white/10 bg-void/80 backdrop-blur-xl p-6 shadow-xl space-y-4">
            <div className="relative h-44 md:h-52 w-full rounded-xl bg-black/90 border border-white/10 p-3 overflow-hidden shadow-inner flex items-center justify-center">
              {/* Reference Grid */}
              <div className="absolute inset-0 grid grid-rows-4 grid-cols-8 pointer-events-none opacity-20">
                {Array.from({ length: 32 }).map((_, i) => (
                  <div key={i} className="border-b border-r border-cyan-500/20" />
                ))}
              </div>

              <svg viewBox="0 0 900 140" preserveAspectRatio="none" className="w-full h-full overflow-visible">
                <defs>
                  <linearGradient id="marAreaReport" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#00F2FE" stopOpacity="0.25" />
                    <stop offset="100%" stopColor="#00F2FE" stopOpacity="0.00" />
                  </linearGradient>
                  <linearGradient id="audioAreaReport" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#10B981" stopOpacity="0.22" />
                    <stop offset="100%" stopColor="#10B981" stopOpacity="0.00" />
                  </linearGradient>
                </defs>

                {/* Reference Baseline */}
                <line x1="10" y1="70" x2="890" y2="70" stroke="rgba(255,255,255,0.08)" strokeDasharray="4 4" />

                {/* Area Gradient Fills */}
                <path d={getSmoothSvgArea(result.mar_decimated, 900, 140, 10)} fill="url(#marAreaReport)" />
                <path d={getSmoothSvgArea(result.envelope_decimated, 900, 140, 10)} fill="url(#audioAreaReport)" />

                {/* Smooth Waveform Lines */}
                <path
                  d={getSmoothSvgPath(result.mar_decimated, 900, 140, 10)}
                  fill="none"
                  stroke="#00F2FE"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  style={{ filter: "drop-shadow(0 0 6px rgba(0, 242, 254, 0.6))" }}
                />
                <path
                  d={getSmoothSvgPath(result.envelope_decimated, 900, 140, 10)}
                  fill="none"
                  stroke="#10B981"
                  strokeWidth="2.0"
                  strokeLinecap="round"
                  style={{ filter: "drop-shadow(0 0 6px rgba(16, 185, 129, 0.5))" }}
                />
              </svg>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-4 pt-2 font-mono text-xs text-ash border-t border-white/5">
              <div className="flex items-center gap-6">
                <span className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-volt shadow-[0_0_6px_#00F2FE]" />
                  MOUTH APERTURE (MAR)
                </span>
                <span className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-pulse shadow-[0_0_6px_#10B981]" />
                  SPEECH ENVELOPE (RMS)
                </span>
              </div>

              <div className="flex items-center gap-4">
                {result.metrics?.lipsync?.lag != null && (
                  <span>
                    ALIGNMENT LAG:{" "}
                    <strong className="text-bone font-semibold">
                      {result.metrics.lipsync.lag > 0 ? "+" : ""}
                      {result.metrics.lipsync.lag.toFixed(0)} ms
                    </strong>
                  </span>
                )}
                {result.metrics?.lipsync?.iqr != null && (
                  <span>
                    JITTER IQR:{" "}
                    <strong className="text-bone font-semibold">
                      {result.metrics.lipsync.iqr.toFixed(0)} ms
                    </strong>
                  </span>
                )}
              </div>
            </div>
          </div>
        </Section>

        {/* ── Actions & Export ─────────────────────────────────────────────── */}
        <motion.div
          variants={{
            hidden: { opacity: 0, y: 18 },
            show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] } },
          }}
          className="border-t border-white/[0.08] pt-10 flex flex-wrap items-center justify-between gap-4"
        >
          <button
            onClick={() => window.print()}
            className="px-6 py-3.5 rounded-full border border-white/15 bg-white/[0.03] font-mono text-xs text-ash hover:text-bone hover:border-white/30 transition-all flex items-center gap-2"
          >
            <span>PRINT FORENSIC AUDIT</span>
            <span>↓</span>
          </button>

          <button
            onClick={onReset}
            className="px-8 py-3.5 rounded-full bg-volt text-void font-display font-semibold text-sm tracking-wide hover:bg-cyan-300 transition-all shadow-[0_0_20px_rgba(0,242,254,0.3)] hover:scale-[1.02]"
          >
            ANALYZE NEW VIDEO
          </button>
        </motion.div>
      </motion.div>
    </div>
  );
}
