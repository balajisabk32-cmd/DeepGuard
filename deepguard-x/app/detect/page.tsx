"use client";
import Link from "next/link";
import { useState } from "react";
import InferenceSandbox from "@/components/InferenceSandbox";
import ForensicReport from "@/components/ForensicReport";
import { FusionResultPayload } from "@/lib/api";

/**
 * /detect — the product.
 *
 * Deliberately separate from the landing page. The landing page sells and makes
 * zero network calls; this page does the work. Keeping them in one route meant
 * the analysis widget loaded on first paint and the "launch" action was a scroll
 * rather than a navigation.
 */
export default function DetectPage() {
  const [result, setResult] = useState<FusionResultPayload | null>(null);

  return (
    <div className="grain relative min-h-[100dvh] w-full overflow-x-hidden">
      <div className="ambient-field" aria-hidden>
        <div className="orb orb-cyan" />
        <div className="orb orb-violet" />
        <div className="orb orb-emerald" />
      </div>

      <header className="relative z-20 flex items-center justify-between px-6 md:px-10 py-6">
        <Link
          href="/"
          className="flex items-center gap-2.5 font-display font-semibold tracking-tight text-bone hover:opacity-80 transition-opacity"
        >
          <span className="text-volt">←</span> DeepGuard
        </Link>
        <span className="text-[10px] font-mono tracking-[0.2em] text-ash uppercase">
          Analysis Console
        </span>
      </header>

      <main className="relative z-10">
        <InferenceSandbox onResultReady={(res) => setResult(res)} />
      </main>

      {result && (
        <ForensicReport result={result} onReset={() => setResult(null)} />
      )}
    </div>
  );
}
