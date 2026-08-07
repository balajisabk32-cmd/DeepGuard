"use client";
import { useState } from "react";
import SmoothScroll from "@/components/SmoothScroll";
import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import BentoGrid from "@/components/BentoGrid";
import ThreatMatrix from "@/components/ThreatMatrix";
import Footer from "@/components/Footer";
import IntroScreen from "@/components/IntroScreen";

/**
 * Landing page. Presentation only.
 *
 * It must make NO network calls and render NO analysis widget — the product
 * lives at /detect. Previously InferenceSandbox was mounted here, so the landing
 * page loaded the analysis client on first paint and "Launch DeepGuard" was a
 * scroll anchor rather than a navigation.
 */
export default function Page() {
  const [introActive, setIntroActive] = useState(true);

  return (
    <SmoothScroll locked={introActive}>
      <div
        className={`grain relative min-h-[100dvh] w-full ${
          introActive ? "overflow-hidden h-screen" : "overflow-x-hidden"
        }`}
      >
        <div className="ambient-field" aria-hidden>
          <div className="orb orb-cyan" />
          <div className="orb orb-violet" />
          <div className="orb orb-emerald" />
        </div>
        <Navbar hideLogo={introActive} hideLinks={introActive} />
        {introActive && <IntroScreen onComplete={() => setIntroActive(false)} />}
        <main className="relative z-10">
          <Hero />
          <BentoGrid />
          <ThreatMatrix />
        </main>
        <Footer />
      </div>
    </SmoothScroll>
  );
}
