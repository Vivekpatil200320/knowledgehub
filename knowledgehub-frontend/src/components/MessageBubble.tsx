"use client";

import ReactMarkdown from "react-markdown";
import { Search } from "lucide-react";
import { CitationPanel } from "@/components/CitationPanel";
import type { Citation } from "@/lib/api";

/**
 * Rewrite literal bullet characters ("•" and its common lookalikes) into Markdown
 * "- " list syntax, preserving indentation.
 *
 * The grounding prompt asks the model for "- " bullets, but an 8B model doesn't
 * reliably follow that — it sometimes emits "•" instead. CommonMark only recognises
 * hyphen, asterisk or plus as list markers, so a "•" line is just ordinary paragraph text: a single
 * newline between two such lines collapses to a space, and a whole multi-point
 * answer renders as one run-on paragraph instead of a list. This is presentational
 * only — normalise at render time, not on the stored message, so the DB keeps the
 * model's actual output for evals/debugging.
 */
function normalizeBullets(text: string): string {
  return text
    .split("\n")
    .map((line) => {
      const match = line.match(/^(\s*)[•●▪‣∙◦]\s+(.*)$/);
      return match ? `${match[1]}- ${match[2]}` : line;
    })
    .join("\n");
}

export function TypingIndicator() {
  return (
    <span className="inline-flex items-center gap-1 py-1" aria-label="Generating answer">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="typing-dot size-1.5 rounded-full bg-text-subtle"
          style={{ animationDelay: `${i * 0.16}s` }}
        />
      ))}
    </span>
  );
}

export function MessageBubble({
  role,
  content,
  citations,
  condensedQuery,
  streaming = false,
}: {
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  condensedQuery: string | null;
  streaming?: boolean;
}) {
  if (role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-accent px-4 py-2.5 text-sm leading-relaxed text-accent-fg">
          <p className="whitespace-pre-wrap break-words">{content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[90%] rounded-2xl rounded-bl-sm border border-border bg-surface-raised px-4 py-3">
        {content ? (
          <div className="space-y-2 text-sm leading-relaxed break-words [&_a]:text-accent [&_a]:underline [&_code]:rounded [&_code]:bg-surface-sunken [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-xs [&_li]:ml-4 [&_li]:list-disc [&_ol_li]:list-decimal [&_p+p]:mt-2 [&_strong]:font-semibold [&_ul]:space-y-1">
            <ReactMarkdown>{normalizeBullets(content)}</ReactMarkdown>
          </div>
        ) : (
          <TypingIndicator />
        )}

        {/* The condensed query is the memory mechanism made visible: it shows the
            follow-up that was actually sent to retrieval. Only the label gets the
            brand's lime-monospace-uppercase tag treatment (mirroring their
            "CONTEXT: FILENAME.PDF" provenance tag) — the query itself is a
            variable-length sentence, not a short fixed label, so uppercasing the
            whole thing would read as shouting rather than as a quiet tag. */}
        {condensedQuery && condensedQuery !== content && (
          <p className="mt-2.5 flex items-start gap-1.5 text-xs">
            <Search aria-hidden="true" className="mt-0.5 size-3 shrink-0 text-accent" />
            <span>
              <span className="font-mono text-[9px] uppercase tracking-[0.45px] text-accent">
                Retrieved using:
              </span>{" "}
              <span className="italic text-text-subtle">“{condensedQuery}”</span>
            </span>
          </p>
        )}

        {!streaming && <CitationPanel citations={citations} />}
      </div>
    </div>
  );
}
