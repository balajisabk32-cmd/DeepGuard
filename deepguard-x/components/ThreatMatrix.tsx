"use client";
import { useRef } from "react";
import { useGSAP } from "@gsap/react";
import { gsap } from "@/lib/gsap";

// Status reports what we have MEASURED, not what the architecture aims at.
// Every row previously read "DETECTED", including Face2Face reenactment — whose
// stated vector, "Biomechanical AU Kinematics", describes a channel that does
// not exist in this codebase — and Sora/Flux fully-synthetic video, which has
// never been put through the pipeline. A judge who asks "what is your recall on
// Sora?" has to get a number or an honest "not evaluated"; a green badge that
// means "we hope so" is the single easiest claim to disprove in a live demo.
const ROWS = [
  ["Face Swaps", "SimSwap · FaceFusion · Roop", "rPPG coherence + frame-by-frame CNN",
   "EVALUATED · AUC 0.67", "pulse"],
  ["Lip-Sync & Dubbing", "Wav2Lip · SadTalker", "Audio-visual lag IQR drift",
   "EVALUATED · AUC 0.72", "pulse"],
  ["Expression Reenactment", "Face2Face", "Frame-by-frame CNN only",
   "NOT EVALUATED", "amber"],
  ["Fully Synthetic Video", "Sora · Flux", "No test corpus held",
   "NOT EVALUATED", "amber"],
  ["Voice Clone Only", "—", "Out of visual scope", "HONEST ABSTENTION", "amber"],
];

export default function ThreatMatrix() {
  const root = useRef<HTMLElement>(null);

  useGSAP(() => {
    const mm = gsap.matchMedia();
    mm.add("(prefers-reduced-motion: no-preference)", () => {
      gsap.from(".tm-row", {
        x: -40,
        autoAlpha: 0,
        stagger: 0.1,
        duration: 1,
        ease: "vanguard",
        scrollTrigger: { trigger: root.current, start: "top 70%" },
      });
      gsap.from(".tm-title", {
        y: 40,
        autoAlpha: 0,
        duration: 1.1,
        ease: "vanguard",
        scrollTrigger: { trigger: root.current, start: "top 80%" },
      });
    });
    return () => mm.revert();
  }, { scope: root });

  return (
    <section ref={root} className="px-4 py-32 md:py-48">
      <div className="mx-auto max-w-6xl">
        <span className="eyebrow">Attack Coverage</span>
        <h2 className="tm-title mt-6 font-display font-semibold tracking-tight text-4xl md:text-6xl max-w-3xl">
          Know exactly which forgeries fall — and where we abstain.
        </h2>

        <div className="mt-16 bezel-shell">
          <div className="bezel-core !p-0 overflow-x-auto">
            <table className="w-full min-w-[720px] text-left text-sm">
              <thead>
                <tr className="text-[10px] font-mono tracking-[0.25em] text-ash uppercase border-b border-white/10">
                  <th className="px-8 py-5">Attack Class</th>
                  <th className="px-8 py-5">Tooling</th>
                  <th className="px-8 py-5">Detection Vector</th>
                  <th className="px-8 py-5 text-right">Status</th>
                </tr>
              </thead>
              <tbody>
                {ROWS.map(([atk, tool, vec, st, tone]) => (
                  <tr key={atk} className="tm-row group border-b border-white/5 transition-colors duration-700 ease-fluid hover:bg-white/[0.03]">
                    <td className="px-8 py-6 font-display font-medium">{atk}</td>
                    <td className="px-8 py-6 text-ash font-mono text-xs">{tool}</td>
                    <td className="px-8 py-6 text-ash">{vec}</td>
                    <td className="px-8 py-6 text-right">
                      <span className={`px-3 py-1 rounded-full border font-mono text-[11px] ${tone === "pulse" ? "bg-pulse/10 border-pulse/40 text-pulse" : "bg-amber-500/10 border-amber-500/40 text-amber-300"}`}>
                        {st}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
}
