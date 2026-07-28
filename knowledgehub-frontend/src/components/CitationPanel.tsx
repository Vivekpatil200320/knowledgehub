"use client";

import { useState } from "react";
import { FileText } from "lucide-react";
import type { Citation } from "@/lib/api";

export function CitationPanel({ citations }: { citations: Citation[] }) {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  if (citations.length === 0) return null;

  return (
    <div className="mt-3 border-t border-border pt-3">
      <p className="mb-2 font-mono text-[9px] uppercase tracking-[0.45px] text-accent">
        Sources
      </p>
      <div className="flex flex-wrap gap-1.5">
        {citations.map((citation, index) => {
          const isOpen = openIndex === index;
          return (
            <button
              key={`${citation.document_id}-${citation.chunk_index}`}
              type="button"
              onClick={() => setOpenIndex(isOpen ? null : index)}
              aria-expanded={isOpen}
              title={`${citation.filename} — chunk ${citation.chunk_index}`}
              className={`inline-flex max-w-full items-center gap-1.5 rounded-full border px-[11px] py-[5px] text-[10px] transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-surface-raised ${
                isOpen
                  ? "border-accent bg-accent-soft text-accent"
                  : "border-white/[0.14] text-text hover:border-border-strong"
              }`}
            >
              <FileText aria-hidden="true" className="size-3.5 shrink-0" />
              <span className="truncate">{citation.filename}</span>
              <span className="shrink-0 tabular-nums opacity-70">
                #{citation.chunk_index}
              </span>
            </button>
          );
        })}
      </div>

      {openIndex !== null && (
        <p className="mt-2 rounded-lg bg-surface-sunken px-3 py-2 font-mono text-xs leading-relaxed text-text-muted">
          {citations[openIndex].snippet}…
        </p>
      )}
    </div>
  );
}
