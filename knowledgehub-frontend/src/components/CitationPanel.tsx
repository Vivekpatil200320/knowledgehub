"use client";

import { useState } from "react";
import type { Citation } from "@/lib/api";

export function CitationPanel({ citations }: { citations: Citation[] }) {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  if (citations.length === 0) return null;

  return (
    <div className="mt-3 border-t border-neutral-200 pt-3 dark:border-neutral-700">
      <p className="mb-2 text-xs font-medium uppercase tracking-wide text-neutral-500">
        Sources
      </p>
      <div className="flex flex-wrap gap-2">
        {citations.map((citation, index) => (
          <button
            key={`${citation.document_id}-${citation.chunk_index}`}
            onClick={() => setOpenIndex(openIndex === index ? null : index)}
            className={`rounded-full border px-2.5 py-1 text-xs transition ${
              openIndex === index
                ? "border-neutral-900 bg-neutral-900 text-white dark:border-neutral-100 dark:bg-neutral-100 dark:text-neutral-900"
                : "border-neutral-300 text-neutral-600 hover:border-neutral-500 dark:border-neutral-600 dark:text-neutral-300"
            }`}
          >
            {citation.filename} · chunk {citation.chunk_index}
          </button>
        ))}
      </div>
      {openIndex !== null && (
        <p className="mt-2 rounded-md bg-neutral-100 px-3 py-2 font-mono text-xs leading-relaxed text-neutral-700 dark:bg-neutral-800 dark:text-neutral-300">
          {citations[openIndex].snippet}…
        </p>
      )}
    </div>
  );
}
