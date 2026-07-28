import type { Metadata } from "next";
import { JetBrains_Mono, Space_Grotesk } from "next/font/google";
import "./globals.css";

// Chartered Vectorial's actual UI font ("Stage Grotesk") is a proprietary,
// self-hosted file — not something to download and reuse. Space Grotesk is the
// closest freely-licensed (SIL OFL, via Google Fonts) geometric grotesk with a
// similar character, used everywhere the brand reference uses one grotesk family
// throughout (headings, chrome, and body copy alike, not just display type).
const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "KnowledgeHub",
  description: "Multi-document RAG assistant with chat memory and source citations",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      {/* The workspace owns the full viewport and manages its own scroll regions,
          so the document itself never scrolls. */}
      <body
        className={`${spaceGrotesk.variable} ${jetbrainsMono.variable} h-dvh overflow-hidden bg-surface font-sans text-text antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
