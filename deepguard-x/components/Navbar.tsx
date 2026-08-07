"use client";
import { useRef } from "react";
import { useGSAP } from "@gsap/react";
import { gsap } from "@/lib/gsap";

interface NavbarProps {
  hideLogo?: boolean;
  hideLinks?: boolean;
}

export default function Navbar({ hideLogo = false, hideLinks = false }: NavbarProps) {
  const nav = useRef<HTMLElement>(null);
  
  useGSAP(() => {
    if (!hideLinks) {
      // Animate the navbar container background and the items when revealed
      gsap.fromTo(
        nav.current,
        { backgroundColor: "rgba(0, 0, 0, 0)", borderColor: "rgba(255, 255, 255, 0)" },
        { backgroundColor: "rgba(0, 0, 0, 0.4)", borderColor: "rgba(255, 255, 255, 0.1)", duration: 1.2, ease: "vanguard" }
      );
      
      gsap.fromTo(
        ".nav-anim-item",
        { y: -12, opacity: 0 },
        { y: 0, opacity: 1, stagger: 0.1, duration: 1.0, ease: "vanguard" }
      );
    }
  }, { scope: nav, dependencies: [hideLinks] });

  return (
    <header className="fixed inset-x-0 top-0 z-40 px-4">
      <nav 
        ref={nav}
        className="mt-6 mx-auto max-w-5xl rounded-full border border-transparent bg-transparent backdrop-blur-xl px-6 py-3.5 flex justify-between items-center transition-all duration-1000"
      >
        {/* Logo & Name Container */}
        <a href="#" className="flex items-center gap-2.5 font-display font-semibold tracking-tight text-bone select-none">
          <div 
            id="nav-logo-target" 
            className={`flex items-center justify-center transition-opacity duration-500 ${hideLogo ? "opacity-0" : "opacity-100"}`}
            style={{ width: "28px", height: "28px" }}
          >
            <img src="/LOGO.png" alt="Logo" className="w-full h-full object-contain invert" />
          </div>
          <div 
            id="nav-name-target" 
            className={`flex items-center transition-opacity duration-500 ${hideLogo ? "opacity-0" : "opacity-100"}`}
            style={{ height: "16px" }}
          >
            <img src="/NAME.png" alt="DeepGuard" className="h-full w-auto object-contain invert" />
          </div>
        </a>

        {/* Links (Staggered animation) */}
        <div className="hidden md:flex items-center gap-1 text-sm text-ash">
          {["Engine", "Threat Matrix", "Research"].map((l) => (
            <a 
              key={l} 
              href={`#${l.toLowerCase().replace(" ", "-")}`}
              className="nav-anim-item px-4 py-2 rounded-full transition-all duration-700 ease-fluid hover:text-bone hover:bg-white/[0.06] opacity-0"
            >
              {l}
            </a>
          ))}
        </div>

        {/* System Status Indicator (Staggered animation) */}
        <div className="nav-anim-item opacity-0 flex items-center gap-2.5 px-4.5 py-2 rounded-full border border-white/10 bg-white/[0.02] select-none">
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full rounded-full bg-pulse opacity-75 animate-ping" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-pulse" />
          </span>
          <span className="text-[11px] font-mono tracking-wider text-ash uppercase">
            DEFENSE: <span className="text-pulse font-semibold">ONLINE</span>
          </span>
        </div>
      </nav>
    </header>
  );
}
