"use client";

import { useEffect, useRef, useState } from "react";
import { AlertCircle, ArrowUp, Sparkles } from "lucide-react";
import { MessageBubble } from "@/components/MessageBubble";
import type { ChatMessage, Citation } from "@/lib/api";

export interface PendingReply {
  content: string;
  citations: Citation[];
  condensedQuery: string | null;
}

export interface StarterDocument {
  filename: string;
  title: string | null;
}

/**
 * Starters are built from the documents actually present. Generic prompts
 * ("what does the data say about pricing?") are frequently unanswerable by a
 * small corpus, so the first thing a new user sees would be a refusal.
 *
 * Prefer the ingested `title` (the document's own opening line) over the filename
 * as the subject. A résumé named "candidate-profile.pdf" never contains the phrase
 * "candidate profile", so a filename-derived starter about it can score below the
 * refusal threshold — the retrieved-content check, not the wording, is what fails.
 * `title` is only set once ingestion has actually completed, so older or
 * still-processing documents fall back to the filename heuristic.
 */
function startersFor(documents: StarterDocument[]): string[] {
  const subject = (doc: StarterDocument) =>
    doc.title ?? doc.filename.replace(/\.[^.]+$/, "").replace(/[-_]/g, " ");

  if (documents.length === 0) return [];
  if (documents.length === 1) {
    return [
      `What is ${subject(documents[0])} about?`,
      `Summarise the key points of ${subject(documents[0])}.`,
    ];
  }
  return [
    `What is ${subject(documents[0])} about?`,
    `Summarise the key points of ${subject(documents[1])}.`,
    `Compare ${subject(documents[0])} and ${subject(documents[1])}.`,
  ];
}

export function ChatPanel({
  messages,
  pending,
  error,
  readyDocuments,
  onSend,
}: {
  messages: ChatMessage[];
  pending: PendingReply | null;
  error: string | null;
  readyDocuments: StarterDocument[];
  onSend: (content: string) => void;
}) {
  const hasDocuments = readyDocuments.length > 0;
  const starters = startersFor(readyDocuments);
  const [draft, setDraft] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pending]);

  // Grow with the content, but stop before the composer eats the conversation.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [draft]);

  function submit() {
    const content = draft.trim();
    if (!content || pending) return;
    setDraft("");
    onSend(content);
  }

  const isEmpty = messages.length === 0 && !pending;

  return (
    <div className="flex h-full min-w-0 flex-col">
      <div
        className="scrollbar-slim min-h-0 flex-1 overflow-y-auto"
        aria-live="polite"
        aria-busy={pending !== null}
      >
        <div className="mx-auto w-full max-w-3xl px-4 py-6">
          {isEmpty ? (
            <div className="flex flex-col items-center py-16 text-center">
              <span className="grid size-11 place-items-center rounded-xl bg-accent-soft">
                <Sparkles aria-hidden="true" className="size-5 text-accent" />
              </span>
              <h2 className="mt-4 text-lg font-semibold">
                {hasDocuments ? "Ask about your documents" : "Upload a document first"}
              </h2>
              <p className="mt-1 max-w-sm text-sm text-text-muted">
                {hasDocuments
                  ? "Answers are grounded in what you uploaded, with sources cited. Follow-up questions keep their context."
                  : "Add a PDF, TXT or Markdown file in the Documents panel, then ask a question about it."}
              </p>

              {hasDocuments && (
                <div className="mt-6 flex w-full max-w-md flex-col gap-2">
                  {starters.map((starter) => (
                    <button
                      key={starter}
                      type="button"
                      onClick={() => onSend(starter)}
                      className="rounded border border-border bg-surface-raised px-3 py-2.5 text-left text-sm text-text-muted transition-colors duration-150 hover:border-accent hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    >
                      {starter}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((message) => (
                <MessageBubble
                  key={message.id}
                  role={message.role}
                  content={message.content}
                  citations={message.citations ?? []}
                  condensedQuery={message.condensed_query}
                />
              ))}

              {pending && (
                <MessageBubble
                  role="assistant"
                  content={pending.content}
                  citations={pending.citations}
                  condensedQuery={pending.condensedQuery}
                  streaming
                />
              )}
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      <div className="shrink-0 border-t border-border bg-surface px-4 py-3">
        <div className="mx-auto w-full max-w-3xl">
          {error && (
            <p
              role="alert"
              className="mb-2 flex items-start gap-2 rounded-lg bg-danger-soft px-3 py-2 text-sm text-danger"
            >
              <AlertCircle aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
              <span>{error}</span>
            </p>
          )}

          <div className="flex items-end gap-2 rounded-xl border border-border bg-surface-raised p-2 transition-colors duration-150 focus-within:border-accent">
            <label htmlFor="composer" className="sr-only">
              Ask a question about your documents
            </label>
            <textarea
              id="composer"
              ref={textareaRef}
              value={draft}
              rows={1}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit();
                }
              }}
              placeholder="Ask a question…"
              className="max-h-[200px] min-h-[24px] flex-1 resize-none bg-transparent px-2 py-1.5 text-sm leading-relaxed outline-none placeholder:text-text-subtle"
            />
            <button
              type="button"
              onClick={submit}
              disabled={!draft.trim() || pending !== null}
              aria-label="Send message"
              className="grid size-9 shrink-0 place-items-center rounded border border-accent bg-surface text-accent transition-colors duration-150 hover:border-accent-hover hover:bg-accent-soft hover:text-accent-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-surface-raised disabled:cursor-not-allowed disabled:opacity-40"
            >
              <ArrowUp aria-hidden="true" className="size-4" />
            </button>
          </div>

          <p className="mt-1.5 px-1 text-[11px] text-text-subtle">
            Enter to send · Shift+Enter for a new line
          </p>
        </div>
      </div>
    </div>
  );
}
