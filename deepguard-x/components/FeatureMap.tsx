"use client";
import { useEffect, useRef, useState } from "react";

const GAIN = 24; // Amplification for the pulse view

type Props = {
  grid: number[][][] | null;
};

// The 7 forensic regions as fractions of the face box
export const FORENSIC_REGIONS: [string, number, number, number, number][] = [
  ["forehead", 0.25, 0.08, 0.75, 0.28],
  ["eye_l", 0.12, 0.32, 0.44, 0.5],
  ["eye_r", 0.56, 0.32, 0.88, 0.5],
  ["nose", 0.38, 0.44, 0.62, 0.66],
  ["cheek_l", 0.1, 0.48, 0.32, 0.7],
  ["cheek_r", 0.68, 0.48, 0.9, 0.7],
  ["mouth", 0.28, 0.62, 0.72, 0.92],
];

/**
 * Dedicated Card for Live Residual Pulse Sensor Grid
 */
export default function FeatureMap({ grid }: Props) {
  const [mode, setMode] = useState<"pulse" | "rgb">("pulse");
  const baseline = useRef<number[][][] | null>(null);
  const [dev, setDev] = useState<number[][] | null>(null);

  useEffect(() => {
    if (!grid || grid.length === 0) return;
    const rows = grid.length;
    const cols = grid[0].length;

    if (
      !baseline.current ||
      baseline.current.length !== rows ||
      baseline.current[0]?.length !== cols
    ) {
      baseline.current = grid.map((r) => r.map((p) => [...p]));
      return;
    }

    const A = 0.12; // EMA weight
    const next: number[][] = [];
    for (let i = 0; i < rows; i++) {
      const row: number[] = [];
      for (let j = 0; j < cols; j++) {
        const b = baseline.current[i][j];
        const p = grid[i][j];
        for (let c = 0; c < 3; c++) b[c] = b[c] * (1 - A) + p[c] * A;
        row.push((p[1] - b[1]) / 255);
      }
      next.push(row);
    }
    setDev(next);
  }, [grid]);

  const hasGrid = !!grid && grid.length > 0;

  return (
    <div className="h-full flex flex-col justify-between">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-white/5">
        <div>
          <span className="text-[10px] font-mono tracking-[0.2em] text-volt uppercase">
            Live Biosignal Array
          </span>
          <h4 className="font-display font-semibold text-base text-bone tracking-tight">
            Pulse-Residual Matrix
          </h4>
        </div>

        <div className="inline-flex rounded-lg border border-white/10 bg-black/40 p-0.5">
          {(["pulse", "rgb"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`font-mono text-[9px] px-2 py-0.5 rounded-md transition-all ${
                mode === m
                  ? "bg-volt text-void font-semibold shadow-sm"
                  : "text-ash hover:text-bone hover:bg-white/5"
              }`}
            >
              {m === "pulse" ? `PULSE ×${GAIN}` : "RAW RGB"}
            </button>
          ))}
        </div>
      </div>

      {/* Grid Display */}
      <div className="py-3 flex-1 flex items-center justify-center">
        {!hasGrid ? (
          <div className="h-28 w-full flex flex-col items-center justify-center gap-1.5 rounded-xl border border-dashed border-white/10 bg-white/[0.01]">
            <span className="w-2 h-2 rounded-full bg-volt animate-ping" />
            <span className="font-mono text-[10px] text-ash/60 tracking-wider">
              AQUIRING CHROM/POS RESIDUAL...
            </span>
          </div>
        ) : (
          <div
            className="grid gap-1.5 w-full max-w-[260px] p-2 rounded-xl border border-white/10 bg-black/70 shadow-inner"
            style={{ gridTemplateColumns: `repeat(${grid![0].length}, minmax(0, 1fr))` }}
          >
            {grid!.map((row, i) =>
              row.map((px, j) => {
                let bg: string;
                if (mode === "rgb") {
                  bg = `rgb(${px[0]}, ${px[1]}, ${px[2]})`;
                } else {
                  const d = dev?.[i]?.[j] ?? 0;
                  const v = Math.max(-1, Math.min(1, d * GAIN));
                  bg =
                    v >= 0
                      ? `rgba(0, 242, 254, ${Math.min(0.85, 0.12 + v * 0.75)})`
                      : `rgba(239, 68, 68, ${Math.min(0.85, 0.12 - v * 0.75)})`;
                }
                return (
                  <div
                    key={`${i}-${j}`}
                    className="aspect-square rounded-[3px] border border-white/10 transition-colors duration-150 shadow-sm"
                    style={{ background: bg }}
                  />
                );
              })
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between text-[10px] font-mono text-ash/70 pt-2 border-t border-white/5">
        <span>{mode === "pulse" ? "Green-Channel Deviation" : "Sensor Mean RGB"}</span>
        <span className="text-volt font-semibold">{hasGrid ? "30 PATCH SENSORS" : "STANDBY"}</span>
      </div>
    </div>
  );
}

/**
 * Dedicated Card for 7 Forensic Facial Occlusion Regions
 */
export function ForensicRegionMap() {
  return (
    <div className="h-full flex flex-col justify-between">
      <div className="flex items-center justify-between pb-3 border-b border-white/5">
        <div>
          <span className="text-[10px] font-mono tracking-[0.2em] text-volt uppercase">
            Occlusion Spatial Grid
          </span>
          <h4 className="font-display font-semibold text-base text-bone tracking-tight">
            7 Forensic Face Regions
          </h4>
        </div>
        <span className="text-[9px] font-mono px-2 py-0.5 rounded bg-volt/10 text-volt border border-volt/20">
          CNN MASK
        </span>
      </div>

      <div className="py-3 flex-1 flex items-center justify-center">
        <div className="relative w-36 aspect-[3/4] rounded-xl border border-white/15 bg-black/60 overflow-hidden shadow-inner p-1">
          <div className="absolute inset-0 bg-[radial-gradient(#00f2fe_1px,transparent_1px)] [background-size:8px_8px] opacity-20" />
          {FORENSIC_REGIONS.map(([name, x0, y0, x1, y1], i) => (
            <div
              key={name}
              className="absolute rounded border border-volt/60 bg-volt/10 transition-all hover:bg-volt/30 flex items-center justify-center"
              style={{
                left: `${x0 * 100}%`,
                top: `${y0 * 100}%`,
                width: `${(x1 - x0) * 100}%`,
                height: `${(y1 - y0) * 100}%`,
                animation: `fmPulse 2.5s ease-in-out ${i * 0.15}s infinite`,
              }}
            >
              <span className="font-mono text-[7px] text-volt font-medium uppercase tracking-tighter truncate px-0.5">
                {name}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="flex items-center justify-between text-[10px] font-mono text-ash/70 pt-2 border-t border-white/5">
        <span>TorchScript Spatial Anchors</span>
        <span className="text-volt font-semibold">7 ROIs ACTIVE</span>
      </div>

      <style jsx>{`
        @keyframes fmPulse {
          0%,
          100% {
            opacity: 0.35;
            transform: scale(0.98);
          }
          50% {
            opacity: 1;
            transform: scale(1.02);
          }
        }
      `}</style>
    </div>
  );
}
