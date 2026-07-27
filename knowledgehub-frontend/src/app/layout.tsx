import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
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
      <body
        className={`${geistSans.variable} ${geistMono.variable} min-h-screen bg-neutral-50 text-neutral-900 antialiased dark:bg-neutral-950 dark:text-neutral-100`}
      >
        <header className="border-b border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-900">
          <nav className="mx-auto flex max-w-5xl items-center gap-6 px-6 py-4">
            <span className="font-semibold tracking-tight">KnowledgeHub</span>
            <Link
              href="/documents"
              className="text-sm text-neutral-600 transition hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100"
            >
              Documents
            </Link>
            <Link
              href="/chat"
              className="text-sm text-neutral-600 transition hover:text-neutral-900 dark:text-neutral-400 dark:hover:text-neutral-100"
            >
              Chat
            </Link>
          </nav>
        </header>
        <main className="mx-auto max-w-5xl px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
