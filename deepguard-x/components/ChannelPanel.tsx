"use client";
import { FusionResultPayload, MetricDetail, Verdict } from "@/lib/api";

/**
 * The four mandated channels, side by side, with the arithmetic that produced
 * the verdict — and the arithmetic that produced an ABSTENTION.
 *
 * Two rules this panel exists to enforce:
 *
 * 1. A channel that did not vote must LOOK like it did not vote. The previous
 *    report printed "0.000" for two disabled models, which reads as "measured,
 *    found nothing" when the truth is "never ran". Absent evidence and zero
 *    evidence are different claims.
 *
 * 2. Prior != influence. Priors say what a channel is worth when evidence is
 *    equal; `quality` says how much evidence there actually was. Contribution is
 *    the product, and it is the only number that moved the verdict. Showing the
 *    prior alone would overstate a channel that saw nothing.
 */

const CHANNELS: {
  key: keyof FusionResultPayload["metrics"];
  index: string;
  name: string;
  basis: string;
}[] = [
  { key: "rppg", index: "01", name: "Blood-Volume Pulse", basis: "rPPG · cross-region coherence" },
  { key: "lipsync", index: "02", name: "Lip-Sync Integrity", basis: "audio envelope × mouth aperture" },
  { key: "pixel", index: "03", name: "Pixel Forensics", basis: "classical · texture, warp, flicker" },
  { key: "visual", index: "04", name: "Frame-by-Frame", basis: "EfficientNet-B7 · per-frame CNN" },
  { key: "aigen", index: "05", name: "Synthetic Imagery", basis: "SwinV2 · full-frame, no face needed" },
];

const REGION_LABEL: Record<string, string> = {
  mouth: "Mouth",
  eye_l: "Left eye",
  eye_r: "Right eye",
  nose: "Nose",
  forehead: "Forehead",
  cheek_l: "Left cheek",
  cheek_r: "Right cheek",
};

function Bar({ value, className }: { value: number; className: string }) {
  return (
    <div className="w-full h-2 rounded-full bg-white/5 overflow-hidden p-0.5 border border-white/5">
      <div
        className={`h-full rounded-full transition-all duration-700 shadow-sm ${className}`}
        style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
      />
    </div>
  );
}

function ChannelRow({ meta, m }: { meta: (typeof CHANNELS)[number]; m?: MetricDetail }) {
  // A channel can be missing entirely from an older payload. Render the absence.
  if (!m) {
    return (
      <div className="rounded-lg border border-white/5 bg-black/20 p-4 opacity-60">
        <span className="text-[10px] font-mono tracking-[0.2em] text-ash">CHANNEL {meta.index}</span>
        <h4 className="font-display text-base text-bone mt-1">{meta.name}</h4>
        <p className="font-mono text-[10px] text-ash mt-2">NOT REPORTED BY BACKEND</p>
      </div>
    );
  }

  const quality = m.quality ?? 0;
  const contribution = m.contribution ?? 0;
  // Quality, not the prior, is what gates a vote in log-odds fusion.
  const voted = quality > 0 && contribution > 0;
  const leaning = m.score > 0.5;

  return (
    <div
      className={`rounded-xl border p-5 transition-colors ${
        voted ? "border-white/10 bg-black/30" : "border-white/5 bg-black/10"
      }`}
    >
      <div className="flex justify-between items-start gap-3">
        <div className="min-w-0">
          <span className="text-[10px] font-mono tracking-[0.2em] text-volt uppercase">
            CHANNEL {meta.index}
          </span>
          <h4 className="font-display font-semibold text-base tracking-tight text-bone mt-1 truncate">
            {meta.name}
          </h4>
          <p className="font-mono text-[9px] text-ash/70 mt-0.5 truncate">{meta.basis}</p>
        </div>
        {/* A non-voting channel can still hold a real, calibrated measurement —
            pixel forensics does. Show the number greyed rather than hiding it:
            hiding throws away a measurement, and printing it in the voting
            colours would imply it moved the verdict. */}
        <div className="text-right shrink-0">
          <div
            className={`font-mono text-2xl font-semibold tabular-nums ${
              !voted ? "text-zinc-500" : leaning ? "text-red-400" : "text-pulse"
            }`}
          >
            {m.score.toFixed(3)}
          </div>
          {voted ? (
            <div className="font-mono text-[9px] text-ash">
              {leaning ? "toward MANIPULATED" : "toward AUTHENTIC"}
            </div>
          ) : (
            <span className="inline-block mt-1 font-mono text-[9px] px-2 py-0.5 rounded-full bg-zinc-800/60 border border-zinc-600/40 text-zinc-400">
              DOES NOT VOTE
            </span>
          )}
        </div>
      </div>

      <div className="mt-5 space-y-3">
        <div>
          <div className="flex justify-between font-mono text-[10px] text-ash mb-1">
            <span>EVIDENCE QUALITY</span>
            <span className="text-bone">{quality.toFixed(3)}</span>
          </div>
          <Bar value={quality} className="bg-volt" />
        </div>
        <div>
          <div className="flex justify-between font-mono text-[10px] text-ash mb-1">
            <span>CONTRIBUTION TO VERDICT</span>
            <span className="text-bone">{(contribution * 100).toFixed(1)}%</span>
          </div>
          <Bar value={contribution} className="bg-flux" />
        </div>
        <div className="flex justify-between font-mono text-[9px] text-ash/60 pt-1">
          <span>PRIOR WEIGHT {((m.prior ?? 0) * 100).toFixed(0)}%</span>
          <span>
            {m.hr != null && `HR ${m.hr.toFixed(0)} BPM`}
            {m.lag != null && `LAG ${m.lag > 0 ? "+" : ""}${m.lag.toFixed(0)} ms`}
            {m.frames_scored != null && `${m.frames_scored} FRAMES`}
            {m.frames_used != null && m.frames_scored == null && `${m.frames_used} FRAMES`}
          </span>
        </div>
        {m.models_used && m.models_used.length > 0 && (
          <div className="font-mono text-[9px] text-ash/60">
            MODEL {m.models_used.join(", ")}
          </div>
        )}
        {/* The reason a channel is silent is more useful than its neutral score. */}
        {!voted && m.degraded_reason && (
          <p className="font-mono text-[9px] text-amber-400/80 pt-1">
            {m.degraded_reason.replace(/_/g, " ")} — reported, not counted
          </p>
        )}
      </div>
    </div>
  );
}

export default function ChannelPanel({ result }: { result: FusionResultPayload }) {
  const verdict = result.verdict as Verdict;
  const abstained = verdict === "UNCERTAIN" || verdict === "INSUFFICIENT_EVIDENCE";
  const ew = result.evidence_weight;
  const minEw = result.min_evidence_weight;
  const att = result.attribution;

  const ranked = att?.regions
    ? Object.entries(att.regions).sort((a, b) => b[1] - a[1])
    : [];
  const maxAbs = ranked.length ? Math.max(...ranked.map(([, v]) => Math.abs(v))) : 1;

  return (
    <div className="space-y-6">
      {/* ── four channels ──
          No heading or bezel of its own: the report's <Section> already titles
          this band, and nesting a second card frame inside it was most of why
          the report felt boxed-in. */}
      <div>
        <p className="font-body text-[15px] leading-relaxed text-ash max-w-2xl mb-6">
          Each channel reports a score and how much evidence it actually had.
          Contribution is the product of the two &mdash; it is the only number that
          moved the verdict.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {CHANNELS.map((c) => (
            <ChannelRow key={c.key} meta={c} m={result.metrics?.[c.key]} />
          ))}
        </div>
      </div>

      {/* ── abstention ledger ──
          UNCERTAIN and INSUFFICIENT_EVIDENCE are different failures and must not
          collapse into one "inconclusive" message. UNCERTAIN means the channels
          were heard and disagreed, or landed inside the undecided band.
          INSUFFICIENT_EVIDENCE means too little was measurable to ask. */}
      {abstained && (
        <div className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-6">
          <div>
            <div className="flex items-center gap-3 mb-3">
              <span
                className={`w-2 h-2 rounded-full ${
                  verdict === "UNCERTAIN" ? "bg-amber-400" : "bg-zinc-500"
                }`}
              />
              <h3 className="font-display font-semibold text-lg tracking-tight text-bone">
                {verdict === "UNCERTAIN"
                  ? "Why the system declined to call this"
                  : "Why the system could not assess this"}
              </h3>
            </div>
            <p className="text-ash text-sm leading-relaxed">
              {verdict === "UNCERTAIN" ? (
                <>
                  The channels returned usable evidence, but the fused probability landed
                  inside the undecided band between the authentic and manipulated
                  thresholds. Forcing a call here is how a detector produces a confident
                  error, so DeepGuard reports the uncertainty instead.
                </>
              ) : (
                <>
                  Too little measurable evidence was recovered to support any verdict —
                  the channels that could not see anything abstained rather than voting
                  a neutral score. An abstention is a correct output, not a failure.
                </>
              )}
            </p>
            {ew != null && minEw != null && (
              <div className="mt-4">
                <div className="flex justify-between font-mono text-[10px] text-ash mb-1">
                  <span>TOTAL EVIDENCE WEIGHT</span>
                  <span
                    className={ew >= minEw ? "text-pulse" : "text-amber-400"}
                  >
                    {ew.toFixed(3)} / {minEw.toFixed(2)} REQUIRED
                  </span>
                </div>
                <div className="relative w-full h-1.5 rounded-full bg-white/5 overflow-hidden">
                  <div
                    className={`h-full ${ew >= minEw ? "bg-pulse" : "bg-amber-500"}`}
                    style={{ width: `${Math.min(1, ew) * 100}%` }}
                  />
                  <div
                    className="absolute inset-y-0 w-px bg-bone/70"
                    style={{ left: `${Math.min(1, minEw) * 100}%` }}
                  />
                </div>
              </div>
            )}
            {result.warnings && result.warnings.length > 0 && (
              <ul className="mt-4 space-y-1 font-mono text-[10px] text-ash/70">
                {result.warnings.map((w) => (
                  <li key={w}>· {w.replace(/_/g, " ")}</li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}

      {/* ── explainability ── */}
      {att && !att.degraded_reason && ranked.length > 0 && (
        <div className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-6">
          <div>
            <div className="flex items-baseline justify-between flex-wrap gap-2 mb-2">
              <h3 className="font-display font-semibold text-xl tracking-tight text-bone">
                Region Attribution
              </h3>
              <span className="font-mono text-[10px] text-ash">
                OCCLUSION SENSITIVITY · {att.frames_used} FRAMES
              </span>
            </div>
            <p className="text-ash text-sm leading-relaxed mb-5">
              Each face region is masked in turn and the frame-by-frame model is re-run.
              The bar is the measured drop in manipulation score — how much that region
              was actually carrying the decision. This is a measurement, not a gradient
              approximation, so it holds for the frozen TorchScript model we deploy.
            </p>
            <div className="space-y-2.5">
              {ranked.map(([name, delta]) => {
                const pos = delta >= 0;
                return (
                  <div key={name} className="flex items-center gap-3">
                    <span className="font-mono text-[10px] text-ash w-20 shrink-0">
                      {REGION_LABEL[name] ?? name}
                    </span>
                    <div className="flex-1 h-4 flex items-center">
                      <div className="w-1/2 flex justify-end">
                        {!pos && (
                          <div
                            className="h-2 bg-zinc-600 rounded-l"
                            style={{ width: `${(Math.abs(delta) / maxAbs) * 100}%` }}
                          />
                        )}
                      </div>
                      <div className="w-px h-4 bg-white/20" />
                      <div className="w-1/2">
                        {pos && (
                          <div
                            className="h-2 bg-volt rounded-r"
                            style={{ width: `${(Math.abs(delta) / maxAbs) * 100}%` }}
                          />
                        )}
                      </div>
                    </div>
                    <span
                      className={`font-mono text-[10px] w-16 text-right shrink-0 ${
                        pos ? "text-volt" : "text-ash/60"
                      }`}
                    >
                      {delta >= 0 ? "+" : ""}
                      {delta.toFixed(3)}
                    </span>
                  </div>
                );
              })}
            </div>
            {att.top_region && (
              <p className="font-mono text-[10px] text-ash/70 mt-4">
                MOST INFLUENTIAL · {(REGION_LABEL[att.top_region] ?? att.top_region).toUpperCase()}
                {" · "}BASELINE {att.baseline.toFixed(3)}
              </p>
            )}
            {/* Attribution says WHERE the model looked, never that the region is fake. */}
            <p className="text-[10px] text-ash/50 mt-2 leading-relaxed">
              Attribution indicates influence on this score. It is not evidence that the
              named region was manipulated.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
