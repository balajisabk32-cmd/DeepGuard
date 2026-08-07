"use client";
import { useEffect, useRef } from "react";
import Lenis from "lenis";
import { gsap, ScrollTrigger } from "@/lib/gsap";

interface SmoothScrollProps {
  children: React.ReactNode;
  locked?: boolean;
}

export default function SmoothScroll({ children, locked = false }: SmoothScrollProps) {
  const frame = useRef<number | null>(null);
  const lenisRef = useRef<Lenis | null>(null);

  useEffect(() => {
    const lenis = new Lenis({ lerp: 0.09, smoothWheel: true });
    lenisRef.current = lenis;
    lenis.on("scroll", ScrollTrigger.update);
    
    const loop = (time: number) => {
      lenis.raf(time);
      frame.current = requestAnimationFrame(loop);
    };
    frame.current = requestAnimationFrame(loop);
    
    const tickerUpdate = (time: number) => lenis.raf(time * 1000);
    gsap.ticker.add(tickerUpdate);
    gsap.ticker.lagSmoothing(0);

    return () => {
      if (frame.current !== null) cancelAnimationFrame(frame.current);
      gsap.ticker.remove(tickerUpdate);
      lenis.destroy();
    };
  }, []);

  useEffect(() => {
    const lenis = lenisRef.current;
    if (!lenis) return;

    if (locked) {
      lenis.stop();
      document.documentElement.classList.add("lenis-stopped");
      document.body.style.overflow = "hidden";
    } else {
      lenis.start();
      document.documentElement.classList.remove("lenis-stopped");
      document.body.style.overflow = "";
    }
  }, [locked]);

  return <>{children}</>;
}
