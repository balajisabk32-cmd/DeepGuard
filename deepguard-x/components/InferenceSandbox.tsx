"use client";
import { useCallback, useRef, useState, useEffect } from "react";
import { motion } from "motion/react";
import FeatureMap, { ForensicRegionMap } from "./FeatureMap";
import { uploadVideo, connectAnalysisWS, FrameData, FusionResultPayload } from "@/lib/api";

const cardIn = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.22, 1, 0.36, 1] as const } },
};

interface InferenceSandboxProps {
  onResultReady: (result: FusionResultPayload) => void;
}

/**
 * Generates an ultra-smooth cubic bezier SVG path from raw data points
 */
function getSmoothSvgPath(
  data: number[],
  width = 400,
  height = 120,
  minVal = 0,
  maxVal = 1,
  padding = 10
): string {
  if (data.length < 2) return "";
  const step = (width - padding * 2) / (data.length - 1);
  const range = maxVal - minVal || 1;
  const pts = data.map((val, i) => {
    const norm = Math.max(0, Math.min(1, (val - minVal) / range));
    const y = height - padding - norm * (height - padding * 2);
    const x = padding + i * step;
    return { x, y };
  });

  let d = `M ${pts[0].x.toFixed(1)},${pts[0].y.toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i === 0 ? 0 : i - 1];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2] || p2;
    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${cp1x.toFixed(1)},${cp1y.toFixed(1)} ${cp2x.toFixed(1)},${cp2y.toFixed(1)} ${p2.x.toFixed(1)},${p2.y.toFixed(1)}`;
  }
  return d;
}

/**
 * Generates a closed area SVG path for glowing translucent underfills
 */
function getSmoothSvgArea(
  data: number[],
  width = 400,
  height = 120,
  minVal = 0,
  maxVal = 1,
  padding = 10
): string {
  const linePath = getSmoothSvgPath(data, width, height, minVal, maxVal, padding);
  if (!linePath) return "";
  const lastX = padding + (data.length - 1) * ((width - padding * 2) / (data.length - 1));
  const bottomY = height - padding;
  return `${linePath} L ${lastX.toFixed(1)},${bottomY} L ${padding},${bottomY} Z`;
}

export default function InferenceSandbox({ onResultReady }: InferenceSandboxProps) {
  const [state, setState] = useState<"idle" | "uploading" | "scanning" | "completed" | "error">("idle");
  const [progress, setProgress] = useState(0);
  const [phase, setPhase] = useState("Awaiting Video Input...");
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [isMuted, setIsMuted] = useState(false);
  const activeSessionId = useRef<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [resultData, setResultData] = useState<FusionResultPayload | null>(null);
  const activeFrameSize = useRef<[number, number]>([320, 240]);
  const liveCoherence = useRef<(number | null)[][] | null>(null);
  const frameBoxesRef = useRef<Array<{ timestamp: number; box: [number, number, number, number] | null; width: number; height: number }>>([]);

  const [waveHistory, setWaveHistory] = useState<Array<{ mar: number; audio: number }>>([]);
  const [rgbGrid, setRgbGrid] = useState<number[][][] | null>(null);
  const [videoAspect, setVideoAspect] = useState(16 / 9);
  const activeBox = useRef<[number, number, number, number] | null>(null);
  const scanLineY = useRef(0);
  const scanDirection = useRef(1);

  const handleVideoMeta = useCallback(() => {
    const v = videoRef.current;
    if (v?.videoWidth && v?.videoHeight) {
      setVideoAspect(v.videoWidth / v.videoHeight);
    }
  }, []);

  // Animate the laser scan line
  useEffect(() => {
    let animId: number;
    const animateLaser = () => {
      scanLineY.current += 1.8 * scanDirection.current;
      if (scanLineY.current > 100) {
        scanLineY.current = 100;
        scanDirection.current = -1;
      } else if (scanLineY.current < 0) {
        scanLineY.current = 0;
        scanDirection.current = 1;
      }
      animId = requestAnimationFrame(animateLaser);
    };
    if (state === "scanning") {
      animateLaser();
    }
    return () => cancelAnimationFrame(animId);
  }, [state]);

  // Real-time canvas drawing overlay
  useEffect(() => {
    if ((state !== "scanning" && state !== "completed") || !canvasRef.current || !videoRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const video = videoRef.current;

    let active = true;
    const drawOverlay = () => {
      if (!active || !ctx || !video) return;

      const rect = video.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      let box: [number, number, number, number] | null = null;
      let frameW = 320;
      let frameH = 240;

      if (state === "scanning") {
        box = activeBox.current;
        frameW = activeFrameSize.current[0];
        frameH = activeFrameSize.current[1];
      } else if (state === "completed" && frameBoxesRef.current.length > 0) {
        const curTime = video.currentTime;
        let closest = frameBoxesRef.current[0];
        let minDist = Math.abs(closest.timestamp - curTime);
        for (let i = 1; i < frameBoxesRef.current.length; i++) {
          const dist = Math.abs(frameBoxesRef.current[i].timestamp - curTime);
          if (dist < minDist) {
            minDist = dist;
            closest = frameBoxesRef.current[i];
          }
        }
        box = closest.box;
        frameW = closest.width;
        frameH = closest.height;
      }

      if (box) {
        const scaleX = canvas.width / frameW;
        const scaleY = canvas.height / frameH;
        const bx = box[0] * scaleX;
        const by = box[1] * scaleY;
        const bw = box[2] * scaleX;
        const bh = box[3] * scaleY;

        // 1. Sleek Cybernetic Corner Brackets
        ctx.strokeStyle =
          state === "completed" && resultData?.verdict === "LIKELY_MANIPULATED" ? "#EF4444" : "#00F2FE";
        ctx.lineWidth = 2.5;
        const edge = 16;

        // Top Left
        ctx.beginPath();
        ctx.moveTo(bx, by + edge);
        ctx.lineTo(bx, by);
        ctx.lineTo(bx + edge, by);
        ctx.stroke();

        // Top Right
        ctx.beginPath();
        ctx.moveTo(bx + bw - edge, by);
        ctx.lineTo(bx + bw, by);
        ctx.lineTo(bx + bw, by + edge);
        ctx.stroke();

        // Bottom Left
        ctx.beginPath();
        ctx.moveTo(bx, by + bh - edge);
        ctx.lineTo(bx, by + bh);
        ctx.lineTo(bx + edge, by + bh);
        ctx.stroke();

        // Bottom Right
        ctx.beginPath();
        ctx.moveTo(bx + bw - edge, by + bh);
        ctx.lineTo(bx + bw, by + bh);
        ctx.lineTo(bx + bw, by + bh - edge);
        ctx.stroke();

        // 2. 6x5 scanning grid matrix
        const cols = 5;
        const rows = 6;
        const cellW = bw / cols;
        const cellH = bh / rows;
        ctx.lineWidth = 0.5;

        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            const cx = bx + j * cellW;
            const cy = by + i * cellH;

            ctx.strokeStyle =
              state === "completed" && resultData?.verdict === "LIKELY_MANIPULATED"
                ? "rgba(239, 68, 68, 0.12)"
                : "rgba(0, 242, 254, 0.12)";
            ctx.strokeRect(cx, cy, cellW, cellH);

            const grid =
              state === "completed" && resultData?.map_coherence
                ? resultData.map_coherence
                : liveCoherence.current;
            if (grid) {
              const val = grid[i]?.[j];
              if (val !== null && val !== undefined && Number.isFinite(val)) {
                if (val >= 0.2) {
                  ctx.fillStyle = `rgba(16, 185, 129, ${Math.min(0.25 + (val - 0.2) * 0.45, 0.55)})`;
                } else if (val >= -0.1) {
                  ctx.fillStyle = "rgba(245, 158, 11, 0.30)";
                } else {
                  ctx.fillStyle = `rgba(239, 68, 68, ${Math.min(0.25 + -val * 0.45, 0.55)})`;
                }
                ctx.fillRect(cx + 1, cy + 1, cellW - 2, cellH - 2);
              }
            } else {
              const sweep = 0.5 + 0.5 * Math.sin(Date.now() / 320 - (i + j) * 0.5);
              ctx.fillStyle = `rgba(0, 242, 254, ${0.04 + sweep * 0.07})`;
              ctx.fillRect(cx + 1, cy + 1, cellW - 2, cellH - 2);
            }
          }
        }

        // 3. Central Target Crosshair
        ctx.fillStyle =
          state === "completed" && resultData?.verdict === "LIKELY_MANIPULATED" ? "#EF4444" : "#00F2FE";
        ctx.beginPath();
        ctx.arc(bx + bw / 2, by + bh / 2, 2.5, 0, Math.PI * 2);
        ctx.fill();

        // 4. Laser Scanning Line
        if (state === "scanning") {
          const laserY = by + (bh * scanLineY.current) / 100;
          const grad = ctx.createLinearGradient(bx, laserY, bx + bw, laserY);
          grad.addColorStop(0, "rgba(0, 242, 254, 0)");
          grad.addColorStop(0.5, "rgba(0, 242, 254, 0.85)");
          grad.addColorStop(1, "rgba(0, 242, 254, 0)");
          ctx.strokeStyle = grad;
          ctx.lineWidth = 2.0;
          ctx.shadowColor = "#00F2FE";
          ctx.shadowBlur = 10;
          ctx.beginPath();
          ctx.moveTo(bx, laserY);
          ctx.lineTo(bx + bw, laserY);
          ctx.stroke();
          ctx.shadowBlur = 0;
        }
      }

      requestAnimationFrame(drawOverlay);
    };

    drawOverlay();
    return () => {
      active = false;
    };
  }, [state, resultData]);

  const processFile = useCallback(
    async (file: File) => {
      if (!file.type.startsWith("video/")) {
        setErrorMsg("Please upload a valid MP4 or MOV video file.");
        setState("error");
        return;
      }

      try {
        setState("uploading");
        setProgress(0);
        setPhase("Caching Video Stream...");
        setErrorMsg("");
        setWaveHistory([]);
        activeBox.current = null;
        setResultData(null);
        frameBoxesRef.current = [];

        const localUrl = URL.createObjectURL(file);
        setVideoUrl(localUrl);

        const uploadRes = await uploadVideo(file);
        activeSessionId.current = uploadRes.session_id;

        setTimeout(() => {
          document.getElementById("live-sandbox")?.scrollIntoView({ behavior: "smooth", block: "center" });
        }, 50);

        if (videoRef.current) {
          videoRef.current.muted = isMuted;
          videoRef.current.volume = 1.0;
          videoRef.current.play().catch(() => {
            // Autoplay policy fallback: if unmuted autoplay is blocked by browser, start muted
            if (videoRef.current) {
              videoRef.current.muted = true;
              setIsMuted(true);
              videoRef.current.play().catch(() => {});
            }
          });
        }

        setState("scanning");
        setPhase("Initializing Biometric Extraction...");

        connectAnalysisWS(uploadRes.session_id, {
          onProgress: (p, currentPhase) => {
            setProgress(p);
            setPhase(currentPhase);
          },
          onFrameData: (frame: FrameData) => {
            if (frame.box) {
              activeBox.current = frame.box;
            }
            if (frame.width && frame.height) {
              activeFrameSize.current = [frame.width, frame.height];
            }
            if (frame.coherence) {
              liveCoherence.current = frame.coherence;
            }
            if (frame.rgb_grid) {
              setRgbGrid(frame.rgb_grid);
            }
            if (frame.mar != null || frame.audio_envelope != null) {
              setWaveHistory((prev) => {
                const next = [
                  ...prev,
                  { mar: frame.mar ?? 0, audio: frame.audio_envelope ?? 0 },
                ];
                return next.slice(-60); // 60 points for expansive, fluid waveform
              });
            }
            frameBoxesRef.current.push({
              timestamp: frame.timestamp,
              box: frame.box,
              width: frame.width || 320,
              height: frame.height || 240,
            });
          },
          onResult: (result) => {
            setState("completed");
            setProgress(100);
            setPhase("Analysis Completed");
            setResultData(result);
          },
          onError: (err) => {
            setErrorMsg(err);
            setState("error");
          },
          onClose: () => {
            setState((prev) => (prev === "scanning" ? "error" : prev));
          },
        });
      } catch (e: any) {
        setErrorMsg(e.message || "Failed to run video pipeline.");
        setState("error");
      }
    },
    [onResultReady]
  );

  const handleReset = () => {
    setState("idle");
    setProgress(0);
    setPhase("Awaiting Video Input...");
    setVideoUrl(null);
    setResultData(null);
    liveCoherence.current = null;
    frameBoxesRef.current = [];
    activeBox.current = null;
    setWaveHistory([]);
    setRgbGrid(null);
  };

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        processFile(e.dataTransfer.files[0]);
      }
    },
    [processFile]
  );

  // Latest numerical readouts for the oscilloscope
  const latestMar = waveHistory.length > 0 ? waveHistory[waveHistory.length - 1].mar : 0;
  const latestAudio = waveHistory.length > 0 ? waveHistory[waveHistory.length - 1].audio : 0;
  const marData = waveHistory.map((w) => w.mar);
  const audioData = waveHistory.map((w) => w.audio);

  return (
    <section id="live-sandbox" className="px-4 py-16 md:py-24 max-w-7xl mx-auto w-full">
      {/* Section Header */}
      <div className="text-center mb-10 md:mb-14">
        <span className="eyebrow inline-flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-volt animate-ping" />
          Interactive Telemetry Engine
        </span>
        <h2 className="mt-4 font-display font-semibold tracking-tight text-3xl md:text-5xl text-bone">
          Live Multi-Modal Neural Inspection
        </h2>
        <p className="mt-2.5 text-sm md:text-base text-ash max-w-xl mx-auto font-body">
          Cross-region pulse coherence paired with speech-to-lip biomechanical alignment.
        </p>
      </div>

      {/* Main Container */}
      <div className="w-full">
        {state === "idle" && (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`bezel-shell max-w-3xl mx-auto cursor-pointer transition-all duration-700 ease-fluid overflow-hidden ${
              dragOver ? "scale-[1.01] border-volt/60 shadow-[0_0_40px_rgba(0,242,254,0.15)]" : ""
            }`}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => e.target.files?.[0] && processFile(e.target.files[0])}
              className="hidden"
              accept="video/mp4,video/quicktime"
            />

            <div className="bezel-core min-h-[380px] flex flex-col items-center justify-center p-8 md:p-14 text-center">
              <div className="w-20 h-20 rounded-full bg-volt/10 border border-volt/30 flex items-center justify-center text-volt text-3xl shadow-[0_0_30px_rgba(0,242,254,0.2)] mb-6 transition-transform hover:scale-110">
                ⇪
              </div>
              <p className="font-display text-2xl md:text-3xl font-semibold tracking-tight text-bone">
                Drop your <span className="text-volt">.mp4</span> or <span className="text-volt">.mov</span> file
              </p>
              <p className="text-xs md:text-sm text-ash mt-2.5 font-body max-w-md">
                Clips from 4s to 60s. Real-time extraction runs locally on your stream.
              </p>
              <div className="mt-8 px-5 py-2 rounded-full border border-white/10 bg-white/[0.03] text-[11px] text-ash font-mono tracking-widest uppercase flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-pulse" />
                SECURE SANDBOX · ZERO CLOUD EXFILTRATION
              </div>
            </div>
          </div>
        )}

        {state === "uploading" && (
          <div className="max-w-2xl mx-auto bezel-shell p-1">
            <div className="bezel-core p-12 flex flex-col items-center gap-6 text-center">
              <span className="w-10 h-10 rounded-full border-2 border-volt/20 border-t-volt animate-spin" />
              <div className="space-y-2">
                <p className="font-mono text-sm text-volt tracking-widest uppercase">{phase}</p>
                <p className="text-xs text-ash">Buffering raw frames and preparing landmark geometry...</p>
              </div>
            </div>
          </div>
        )}

        {(state === "scanning" || state === "completed") && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            className="space-y-8"
          >
            {/* ── TOP SECTION: Video Viewport Stage + Expansive Oscilloscope Graph ── */}
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 lg:gap-8 items-start">
              {/* Standalone Video Stage (7 Cols) */}
              <div className="xl:col-span-7 flex flex-col gap-4">
                <div className="rounded-2xl border border-white/10 bg-void/90 p-2.5 md:p-3.5 shadow-2xl backdrop-blur-xl relative overflow-hidden">
                  {/* Stage Header Tags */}
                  <div className="flex items-center justify-between px-3 py-2 text-[10px] font-mono text-ash border-b border-white/5 mb-2">
                    <div className="flex items-center gap-2">
                      <span
                        className={`w-2 h-2 rounded-full ${
                          state === "scanning" ? "bg-volt animate-ping" : "bg-emerald-400"
                        }`}
                      />
                      <span className="text-bone uppercase font-semibold">
                        {state === "scanning" ? "LIVE STREAM INGEST" : "TELEMETRY SETTLED"}
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          if (videoRef.current) {
                            const next = !isMuted;
                            videoRef.current.muted = next;
                            setIsMuted(next);
                            if (!next) {
                              videoRef.current.volume = 1.0;
                              videoRef.current.play().catch(() => {});
                            }
                          }
                        }}
                        className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-[10px] font-mono transition-all select-none ${
                          !isMuted
                            ? "bg-volt/15 border-volt/50 text-volt shadow-[0_0_10px_rgba(0,242,254,0.2)]"
                            : "bg-white/5 border-white/10 text-ash hover:text-bone hover:bg-white/10"
                        }`}
                        title={isMuted ? "Click to Unmute Sound" : "Click to Mute Sound"}
                      >
                        <span>{!isMuted ? "🔊" : "🔇"}</span>
                        <span>{!isMuted ? "SOUND ON" : "MUTED"}</span>
                      </button>
                      <span className="text-volt font-medium hidden sm:inline">30 FPS</span>
                      <span className="border-l border-white/10 pl-3 hidden sm:inline">FACIAL ROI LOCK</span>
                    </div>
                  </div>

                  {/* Video Stage Frame */}
                  <div
                    className="relative rounded-xl overflow-hidden bg-black/95 flex items-center justify-center border border-white/5"
                    style={{
                      aspectRatio: String(videoAspect),
                      maxHeight: "min(52vh, 480px)",
                      minHeight: "260px",
                    }}
                  >
                    <video
                      ref={videoRef}
                      src={videoUrl ?? undefined}
                      onLoadedMetadata={handleVideoMeta}
                      autoPlay
                      className="w-full h-full object-contain"
                      muted={isMuted}
                      playsInline
                      loop
                      controls={true}
                    />
                    <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none z-10" />
                  </div>

                  {/* Bottom Pipeline Phase Progress Bar */}
                  <div className="p-3 mt-2 rounded-xl bg-black/40 border border-white/5 space-y-2">
                    <div className="flex justify-between items-center text-xs font-mono">
                      <span className="text-volt uppercase tracking-wider font-medium">{phase}</span>
                      <span className="text-bone font-semibold">{Math.round(progress)}%</span>
                    </div>
                    <div className="w-full h-2 rounded-full bg-white/5 overflow-hidden p-0.5 border border-white/5">
                      <div
                        className="h-full bg-gradient-to-r from-volt via-cyan-400 to-pulse rounded-full transition-all duration-300 shadow-[0_0_12px_rgba(0,242,254,0.4)]"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Standalone Oscilloscope Graph (5 Cols) */}
              <div className="xl:col-span-5 flex flex-col gap-4">
                <motion.div
                  variants={cardIn}
                  initial="hidden"
                  animate="show"
                  className="rounded-2xl border border-white/10 bg-void/80 backdrop-blur-xl p-5 md:p-6 shadow-xl relative overflow-hidden h-full flex flex-col justify-between"
                >
                  {/* Oscilloscope Header */}
                  <div className="flex items-center justify-between pb-3 border-b border-white/5">
                    <div className="space-y-0.5">
                      <span className="text-[10px] font-mono tracking-[0.2em] text-volt uppercase">
                        Channel 02 Telemetry
                      </span>
                      <h4 className="font-display font-semibold text-lg text-bone tracking-tight">
                        Lip-Sync & Speech Oscilloscope
                      </h4>
                    </div>
                    <div className="flex items-center gap-2 font-mono text-[9px]">
                      <span className="flex items-center gap-1 text-volt px-2 py-0.5 rounded-full bg-volt/10 border border-volt/20">
                        MAR: {latestMar.toFixed(3)}
                      </span>
                      <span className="flex items-center gap-1 text-emerald-400 px-2 py-0.5 rounded-full bg-emerald-400/10 border border-emerald-400/20">
                        ENV: {latestAudio.toFixed(3)}
                      </span>
                    </div>
                  </div>

                  {/* Expansive SVG Oscilloscope Canvas */}
                  <div className="relative mt-4 h-48 md:h-56 w-full rounded-xl bg-black/90 border border-white/10 p-3 overflow-hidden shadow-inner flex items-center justify-center">
                    {/* Background Grid Lines */}
                    <div className="absolute inset-0 grid grid-rows-4 grid-cols-6 pointer-events-none opacity-20">
                      {Array.from({ length: 24 }).map((_, i) => (
                        <div key={i} className="border-b border-r border-cyan-500/20" />
                      ))}
                    </div>

                    {waveHistory.length < 2 ? (
                      <div className="flex flex-col items-center gap-2 text-ash/60 font-mono text-[10px] uppercase">
                        <span className="w-2 h-2 rounded-full bg-volt animate-ping" />
                        SYNCHRONIZING AUDIO-VISUAL SAMPLER...
                      </div>
                    ) : (
                      <svg viewBox="0 0 400 140" preserveAspectRatio="none" className="w-full h-full overflow-visible">
                        <defs>
                          <linearGradient id="marAreaGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#00F2FE" stopOpacity="0.30" />
                            <stop offset="100%" stopColor="#00F2FE" stopOpacity="0.00" />
                          </linearGradient>
                          <linearGradient id="audioAreaGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#10B981" stopOpacity="0.25" />
                            <stop offset="100%" stopColor="#10B981" stopOpacity="0.00" />
                          </linearGradient>
                        </defs>

                        {/* Baseline Reference Line */}
                        <line x1="10" y1="70" x2="390" y2="70" stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" />

                        {/* Area Glowing Fills */}
                        <path d={getSmoothSvgArea(marData, 400, 140, 0.05, 0.45, 10)} fill="url(#marAreaGrad)" />
                        <path d={getSmoothSvgArea(audioData, 400, 140, 0.0, 0.35, 10)} fill="url(#audioAreaGrad)" />

                        {/* Smooth Bezier Waveform Lines */}
                        <path
                          d={getSmoothSvgPath(marData, 400, 140, 0.05, 0.45, 10)}
                          fill="none"
                          stroke="#00F2FE"
                          strokeWidth="2.2"
                          strokeLinecap="round"
                          style={{ filter: "drop-shadow(0 0 6px rgba(0, 242, 254, 0.6))" }}
                        />
                        <path
                          d={getSmoothSvgPath(audioData, 400, 140, 0.0, 0.35, 10)}
                          fill="none"
                          stroke="#10B981"
                          strokeWidth="2.0"
                          strokeLinecap="round"
                          style={{ filter: "drop-shadow(0 0 6px rgba(16, 185, 129, 0.5))" }}
                        />
                      </svg>
                    )}
                  </div>

                  {/* Oscilloscope Legend */}
                  <div className="mt-3 flex items-center justify-between text-[10px] font-mono text-ash/80 pt-2 border-t border-white/5">
                    <div className="flex items-center gap-4">
                      <span className="flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-volt shadow-[0_0_6px_#00F2FE]" />
                        LIP APERTURE (MAR)
                      </span>
                      <span className="flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-pulse shadow-[0_0_6px_#10B981]" />
                        SPEECH ENVELOPE
                      </span>
                    </div>
                    <span className="text-ash/50">VAD GATED SYNC</span>
                  </div>
                </motion.div>
              </div>
            </div>

            {/* ── BOTTOM SECTION: Separate Standalone Telemetry Grids ── */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 items-stretch">
              {/* Card 1: Dedicated 6×5 rPPG Pulse Coherence Seam Matrix */}
              <motion.div
                variants={cardIn}
                initial="hidden"
                animate="show"
                className="rounded-2xl border border-white/10 bg-void/80 backdrop-blur-xl p-5 shadow-xl flex flex-col justify-between"
              >
                <div className="flex items-center justify-between pb-3 border-b border-white/5">
                  <div>
                    <span className="text-[10px] font-mono tracking-[0.2em] text-volt uppercase">
                      Channel 01 Sensor
                    </span>
                    <h4 className="font-display font-semibold text-base text-bone tracking-tight">
                      6×5 Pulse Coherence Grid
                    </h4>
                  </div>
                  <span className="text-[9px] font-mono px-2 py-0.5 rounded bg-volt/10 text-volt border border-volt/20">
                    30 SEAM MATRIX
                  </span>
                </div>

                <div className="py-4 flex-1 flex items-center justify-center">
                  <div className="w-full max-w-[260px] p-2.5 rounded-xl bg-black/70 border border-white/10 shadow-inner">
                    <div className="grid grid-cols-5 gap-1.5">
                      {Array.from({ length: 6 }).map((_, rIdx) =>
                        Array.from({ length: 5 }).map((_, cIdx) => {
                          const grid =
                            state === "completed" && resultData?.map_coherence
                              ? resultData.map_coherence
                              : liveCoherence.current;
                          const val = grid?.[rIdx]?.[cIdx];
                          const hasVal = val !== null && val !== undefined && Number.isFinite(val);
                          let cellCls = "bg-white/[0.02] border border-white/5 text-ash/30";
                          if (hasVal) {
                            if (val >= 0.2) {
                              cellCls =
                                "bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 font-semibold shadow-[0_0_8px_rgba(16,185,129,0.15)]";
                            } else if (val >= -0.1) {
                              cellCls =
                                "bg-amber-500/20 border border-amber-500/40 text-amber-300 font-semibold";
                            } else {
                              cellCls =
                                "bg-red-500/20 border border-red-500/40 text-red-300 font-semibold shadow-[0_0_8px_rgba(239,68,68,0.15)]";
                            }
                          }
                          return (
                            <div
                              key={`grid-${rIdx}-${cIdx}`}
                              className={`h-6 rounded-[4px] flex items-center justify-center text-[9px] font-mono transition-all duration-300 ${cellCls}`}
                            >
                              {hasVal ? val.toFixed(2) : "—"}
                            </div>
                          );
                        })
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex justify-between items-center text-[10px] font-mono text-ash/70 pt-2 border-t border-white/5">
                  <span>Cross-Region Phase Match</span>
                  <span className="text-volt font-medium">CHROM / POS</span>
                </div>
              </motion.div>

              {/* Card 2: Dedicated Live Pulse-Residual Sensor Grid */}
              <motion.div
                variants={cardIn}
                initial="hidden"
                animate="show"
                className="rounded-2xl border border-white/10 bg-void/80 backdrop-blur-xl p-5 shadow-xl flex flex-col justify-between"
              >
                <FeatureMap grid={rgbGrid} />
              </motion.div>

              {/* Card 3: Dedicated 7 Forensic Face Regions Map */}
              <motion.div
                variants={cardIn}
                initial="hidden"
                animate="show"
                className="rounded-2xl border border-white/10 bg-void/80 backdrop-blur-xl p-5 shadow-xl flex flex-col justify-between"
              >
                <ForensicRegionMap />
              </motion.div>
            </div>

            {/* ── COMPLETED STAGE: Final Verdict Banner & Actions ── */}
            {state === "completed" && resultData && (
              <motion.div
                variants={cardIn}
                initial="hidden"
                animate="show"
                className="rounded-2xl border border-volt/30 bg-volt/[0.04] p-6 md:p-8 shadow-2xl backdrop-blur-xl flex flex-col md:flex-row items-center justify-between gap-6"
              >
                <div className="space-y-1.5 text-center md:text-left">
                  <span className="text-[10px] font-mono tracking-[0.2em] text-volt uppercase">
                    Consensus State
                  </span>
                  <h3 className="font-display font-bold text-2xl md:text-3xl text-bone tracking-tight">
                    Multi-Modal Analysis Settled
                  </h3>
                  <p className="text-xs text-ash font-body max-w-lg">
                    Full probability weights, regional attributions, and forensic decimation channels ready for inspection.
                  </p>
                </div>

                <div className="flex flex-col sm:flex-row items-center gap-3 w-full md:w-auto">
                  <button
                    onClick={() => onResultReady(resultData)}
                    className="w-full sm:w-auto px-8 py-4 rounded-xl bg-volt text-void font-display font-bold text-sm tracking-wide hover:bg-cyan-300 transition-all flex items-center justify-center gap-2 shadow-[0_0_20px_rgba(0,242,254,0.3)] hover:scale-[1.02]"
                  >
                    <span>VIEW FULL FORENSIC REPORT</span>
                    <span className="w-6 h-6 rounded-full bg-void/10 flex items-center justify-center text-xs font-bold">
                      ↗
                    </span>
                  </button>

                  <button
                    onClick={handleReset}
                    className="w-full sm:w-auto px-6 py-4 rounded-xl border border-white/10 hover:border-white/20 bg-white/[0.02] hover:bg-white/[0.05] text-xs font-mono text-ash hover:text-bone transition-all"
                  >
                    RESET & DETECT NEW
                  </button>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}

        {state === "error" && (
          <div className="max-w-md mx-auto bezel-shell p-1">
            <div className="bezel-core p-8 flex flex-col items-center justify-center gap-4 text-center">
              <div className="w-12 h-12 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center text-red-400 font-mono text-lg">
                !
              </div>
              <div>
                <p className="font-display text-lg font-medium text-red-300">Pipeline Failed</p>
                <p className="text-xs text-ash mt-1 leading-relaxed">{errorMsg}</p>
              </div>
              <button
                onClick={handleReset}
                className="mt-2 px-6 py-2 rounded-full bg-white/10 border border-white/15 text-xs font-mono text-volt hover:bg-white/20 transition-all"
              >
                TRY AGAIN
              </button>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
