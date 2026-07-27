"use client";

import ReactMarkdown from "react-markdown";
import { Search } from "lucide-react";
import { CitationPanel } from "@/components/CitationPanel";
import type { Citation } from "@/lib/api";

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
            <ReactMarkdown>{content}</ReactMarkdown>
          </div>
        ) : (
          <TypingIndicator />
        )}

        {/* The condensed query is the memory mechanism made visible: it shows the
            follow-up that was actually sent to retrieval. */}
        {condensedQuery && condensedQuery !== content && (
          <p className="mt-2.5 flex items-start gap-1.5 text-xs text-text-subtle">
            <Search aria-hidden="true" className="mt-0.5 size-3 shrink-0" />
            <span>
              Retrieved using <span className="italic">“{condensedQuery}”</span>
            </span>
          </p>
        )}

        {!streaming && <CitationPanel citations={citations} />}
      </div>
    </div>
  );
}
