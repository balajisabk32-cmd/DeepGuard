"use client";
import { useEffect, useRef, useState } from "react";
import { useGSAP } from "@gsap/react";
import { gsap } from "@/lib/gsap";

interface IntroScreenProps {
  onComplete: () => void;
}

export default function IntroScreen({ onComplete }: IntroScreenProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const logoRef = useRef<HTMLDivElement>(null);
  const nameRef = useRef<HTMLDivElement>(null);
  const bypassRef = useRef<HTMLButtonElement>(null);
  const [isAnimatingOut, setIsAnimatingOut] = useState(false);

  useGSAP(() => {
    // 1. Initial cinematic 3D entrance
    const tl = gsap.timeline({
      onComplete: () => {
        // Automatically trigger the flight after the cinematic phase completes + a short delay
        gsap.delayedCall(0.8, startFlight);
      }
    });

    // Animate background glow and grid overlay
    tl.fromTo(
      ".intro-bg-glow",
      { opacity: 0, scale: 0.8 },
      { opacity: 0.6, scale: 1.1, duration: 2.2, ease: "power3.out" }
    );

    // 3D rotation and scale rise for Logo and Name
    tl.fromTo(
      logoRef.current,
      {
        transform: "perspective(1000px) translateZ(-800px) rotateY(-55deg) rotateX(25deg)",
        opacity: 0,
        filter: "blur(15px)",
      },
      {
        transform: "perspective(1000px) translateZ(0px) rotateY(0deg) rotateX(0deg)",
        opacity: 1,
        filter: "blur(0px)",
        duration: 2.0,
        ease: "vanguard",
      },
      0.2
    );

    tl.fromTo(
      nameRef.current,
      {
        transform: "perspective(1000px) translateZ(-600px) rotateY(-40deg) rotateX(15deg)",
        opacity: 0,
        filter: "blur(12px)",
      },
      {
        transform: "perspective(1000px) translateZ(0px) rotateY(0deg) rotateX(0deg)",
        opacity: 1,
        filter: "blur(0px)",
        duration: 1.8,
        ease: "vanguard",
      },
      0.4
    );

    // Sweep scanline across the branding elements
    tl.fromTo(
      ".intro-scanline",
      { top: "-20%", opacity: 0 },
      { top: "120%", opacity: 0.8, duration: 1.8, ease: "power2.inOut" },
      0.6
    );

    // Set up a gentle floating (yoyo) idle loop while waiting
    const floatTween = gsap.fromTo(
      [logoRef.current, nameRef.current],
      { y: 0 },
      {
        y: -10,
        duration: 2.0,
        ease: "sine.inOut",
        yoyo: true,
        repeat: -1,
        paused: true
      }
    );

    // Play float loop after entrance
    tl.add(() => floatTween.play());

    // Fade in skip button
    gsap.fromTo(
      bypassRef.current,
      { opacity: 0, y: 15 },
      { opacity: 1, y: 0, duration: 0.8, ease: "power2.out", delay: 1.0 }
    );

    // Cleanup animations
    return () => {
      floatTween.kill();
      tl.kill();
    };
  }, { scope: containerRef });

  const startFlight = () => {
    if (isAnimatingOut) return;
    setIsAnimatingOut(true);

    const logo = logoRef.current;
    const name = nameRef.current;
    const container = containerRef.current;
    const targetLogo = document.getElementById("nav-logo-target");
    const targetName = document.getElementById("nav-name-target");

    if (!logo || !name || !container || !targetLogo || !targetName) {
      // Fallback if elements aren't in DOM
      onComplete();
      return;
    }

    // Measure start coordinates (centered on screen)
    const logoStartRect = logo.getBoundingClientRect();
    const nameStartRect = name.getBoundingClientRect();

    // Measure final target coordinates in the navbar
    const logoTgtRect = targetLogo.getBoundingClientRect();
    const nameTgtRect = targetName.getBoundingClientRect();

    // Calculate delta positions and scale factors for GPU-safe transform flight (FLIP)
    const logoDeltaX = logoTgtRect.left - logoStartRect.left;
    const logoDeltaY = logoTgtRect.top - logoStartRect.top;
    const logoScaleX = logoTgtRect.width / logoStartRect.width;
    const logoScaleY = logoTgtRect.height / logoStartRect.height;

    const nameDeltaX = nameTgtRect.left - nameStartRect.left;
    const nameDeltaY = nameTgtRect.top - nameStartRect.top;
    const nameScaleX = nameTgtRect.width / nameStartRect.width;
    const nameScaleY = nameTgtRect.height / nameStartRect.height;

    // Trigger GSAP Flight timeline
    const flightTl = gsap.timeline({
      onComplete: () => {
        // Complete state and hide intro screen completely
        onComplete();
      }
    });

    // Fade out background glow, bypass button, and main backdrop gradient
    flightTl.to(
      [bypassRef.current, ".intro-bg-glow", ".intro-scan-pattern"],
      { opacity: 0, duration: 0.6, ease: "power2.out" }
    );

    flightTl.to(
      container,
      { backgroundColor: "rgba(3, 3, 5, 0)", duration: 1.2, ease: "vanguard" },
      0
    );

    // Fly Logo using transform delta
    flightTl.to(
      logo,
      {
        x: logoDeltaX,
        y: logoDeltaY,
        scaleX: logoScaleX,
        scaleY: logoScaleY,
        transformOrigin: "top left",
        duration: 1.4,
        ease: "vanguard",
      },
      0
    );

    // Fly Name using transform delta
    flightTl.to(
      name,
      {
        x: nameDeltaX,
        y: nameDeltaY,
        scaleX: nameScaleX,
        scaleY: nameScaleY,
        transformOrigin: "top left",
        duration: 1.4,
        ease: "vanguard",
      },
      0
    );
  };

  return (
    <div
      ref={containerRef}
      className={`fixed inset-0 z-50 flex flex-col items-center justify-center bg-void overflow-hidden transition-all duration-[1200ms] pointer-events-auto`}
      style={{
        zIndex: 100,
        backgroundColor: "rgba(3, 3, 5, 1)",
      }}
    >
      {/* Cinematic Ambient Background */}
      <div 
        className="intro-bg-glow absolute inset-0 pointer-events-none opacity-0 select-none"
        style={{
          background: "radial-gradient(circle at center, rgba(0, 242, 254, 0.12) 0%, rgba(121, 40, 202, 0.08) 40%, transparent 70%)",
        }}
      />
      
      {/* Glowing Scanline Scanner Matrix */}
      <div className="intro-scan-pattern absolute inset-0 pointer-events-none opacity-20 bg-[linear-gradient(rgba(255,255,255,0.01)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.01)_1px,transparent_1px)] bg-[size:32px_32px] select-none" />

      {/* Main 3D Stage */}
      <div 
        className="relative flex flex-col items-center gap-6 md:gap-8 max-w-lg px-6 select-none"
        style={{ 
          transformStyle: "preserve-3d",
        }}
      >
        {/* Glow behind the logo */}
        <div className="intro-scan-pattern absolute -inset-24 rounded-full bg-volt/5 blur-[80px] pointer-events-none" />
        
        {/* Scanline Sweep */}
        <div className="intro-scanline absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-volt to-transparent shadow-[0_0_12px_#00F2FE] pointer-events-none opacity-0 z-20" />

        {/* 3D Logo Element */}
        <div
          ref={logoRef}
          className="relative transition-shadow duration-500 rounded-2xl flex items-center justify-center"
          style={{
            transformStyle: "preserve-3d",
            willChange: "transform, opacity",
            width: "110px",
            height: "110px",
          }}
        >
          <img
            src="/LOGO.png"
            alt="DeepGuard Logo"
            className="w-28 h-28 md:w-32 md:h-32 object-contain invert filter drop-shadow-[0_0_24px_rgba(0,242,254,0.35)]"
          />
        </div>

        {/* 3D Name Element */}
        <div
          ref={nameRef}
          className="relative flex items-center justify-center"
          style={{
            transformStyle: "preserve-3d",
            willChange: "transform, opacity",
            width: "200px",
            height: "32px",
          }}
        >
          <img
            src="/NAME.png"
            alt="DeepGuard"
            className="h-8 md:h-9 w-auto object-contain invert filter drop-shadow-[0_0_15px_rgba(255,255,255,0.2)]"
          />
        </div>
      </div>

      {/* Skip Button */}
      <button
        ref={bypassRef}
        onClick={startFlight}
        className="absolute bottom-12 px-6 py-2 rounded-full border border-white/10 bg-white/[0.03] text-ash text-xs hover:text-bone hover:border-white/20 transition-all duration-300 backdrop-blur-md opacity-0 z-30 font-mono tracking-wider"
      >
        SKIP INTRO
      </button>
    </div>
  );
}
