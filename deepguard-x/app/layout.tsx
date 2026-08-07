import type { Metadata } from "next";
import { Outfit, Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";

const display = Outfit({ subsets: ["latin"], variable: "--font-display", weight: ["400", "500", "600", "700", "800"] });
const body = Plus_Jakarta_Sans({ subsets: ["latin"], variable: "--font-body", weight: ["300", "400", "500", "600"] });

export const metadata: Metadata = {
  title: "DeepGuard-X — Decode Truth From Synthetic Pixels",
  description: "Multi-modal deepfake defense engine.",
  icons: {
    icon: [
      { url: "/LOGO.png", href: "/LOGO.png" },
      { url: "/icon.png", href: "/icon.png" },
    ],
    shortcut: "/LOGO.png",
    apple: "/LOGO.png",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} bg-void text-bone antialiased`}>
      <body className="font-body overflow-x-hidden">{children}</body>
    </html>
  );
}
