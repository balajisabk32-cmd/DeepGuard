"use client";
import { useRef } from "react";
import { useGSAP } from "@gsap/react";
import { gsap } from "@/lib/gsap";

export default function Footer() {
  const root = useRef<HTMLElement>(null);
  useGSAP(() => {
    const mm = gsap.matchMedia();
    mm.add("(prefers-reduced-motion: no-preference)", () => {
      gsap.from(".cta-rise", {
        y: 60,
        autoAlpha: 0,
        stagger: 0.12,
        duration: 1.2,
        ease: "vanguard",
        scrollTrigger: { trigger: root.current, start: "top 80%" },
      });
    });
    return () => mm.revert();
  }, { scope: root });

  return (
    <footer ref={root} className="relative px-4 pt-32 md:pt-48 pb-16 text-center">
      <h2 className="cta-rise font-display font-semibold tracking-tight text-5xl md:text-7xl max-w-4xl mx-auto leading-[1.05]">
        Stop guessing.
        <br />
        <span className="text-transparent bg-clip-text bg-gradient-to-r from-volt to-pulse">Start verifying.</span>
      </h2>
      <div className="cta-rise mt-12">
        <a
          href="#"
          className="group inline-flex items-center gap-4 rounded-full bg-bone text-void font-semibold px-8 py-4 text-lg transition-all duration-700 ease-fluid active:scale-[0.98] hover:bg-white"
        >
          Deploy DeepGuard-X
          <span className="w-9 h-9 rounded-full bg-void/10 flex items-center justify-center transition-transform duration-700 ease-fluid group-hover:translate-x-1 group-hover:-translate-y-0.5 group-hover:scale-105">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#030305" strokeWidth="2.5">
              <path d="M7 17L17 7M17 7H8M17 7v9" />
            </svg>
          </span>
        </a>
      </div>
      <div className="mt-24 flex flex-wrap items-center justify-center gap-x-8 gap-y-3 text-xs text-ash font-mono">
        <span>DeepGuard-X © 2026</span>
        <span>Research</span>
        <span>API Docs</span>
        <span>Trust & Safety</span>
        <span>Privacy</span>
      </div>
    </footer>
  );
}
