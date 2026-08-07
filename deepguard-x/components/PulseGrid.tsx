"use client";
import { useEffect, useRef } from "react";
import { gsap } from "@/lib/gsap";
import { mulberry32 } from "@/lib/tokens";

export default function PulseGrid({ coherent, seed = 1 }: { coherent: boolean; seed?: number }) {
  const grid = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const cells = grid.current?.querySelectorAll(".pp");
    if (!cells) return;
    const rand = mulberry32(seed);
    const ctx = gsap.context(() => {
      cells.forEach((c) => {
        gsap.to(c, {
          opacity: coherent ? 0.85 : 0.2 + rand() * 0.5,
          duration: coherent ? 0.9 : 0.5 + rand() * 0.9,
          delay: coherent ? 0 : rand() * 1.2, // coherent = same phase; swap = desynced
          repeat: -1,
          yoyo: true,
          ease: "sine.inOut",
        });
      });
    }, grid);
    return () => ctx.revert();
  }, [coherent, seed]);

  return (
    <div ref={grid} className="absolute inset-0 grid grid-cols-6 grid-rows-5 gap-1 p-6 pointer-events-none">
      {Array.from({ length: 30 }).map((_, i) => (
        <div key={i} className="pp rounded-md border border-volt/30 bg-volt/10 opacity-20" />
      ))}
    </div>
  );
}
