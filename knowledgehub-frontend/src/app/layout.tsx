import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
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
        className={`${inter.variable} ${jetbrainsMono.variable} h-dvh overflow-hidden bg-surface font-sans text-text antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
