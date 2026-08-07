import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        void: "#030305",
        ink: "#070709",
        panel: "#0D0E15",
        core: "#0B0C12",
        volt: "#00F2FE",
        flux: "#7928CA",
        pulse: "#10B981",
        bone: "#EDEDF2",
        ash: "#8A8F98",
      },
      fontFamily: {
        display: ["var(--font-display)", "sans-serif"],
        body: ["var(--font-body)", "sans-serif"],
      },
      transitionTimingFunction: {
        fluid: "cubic-bezier(0.32,0.72,0,1)",
      },
      borderRadius: {
        bezel: "2.5rem",
      },
    },
  },
  plugins: [],
};

export default config;
