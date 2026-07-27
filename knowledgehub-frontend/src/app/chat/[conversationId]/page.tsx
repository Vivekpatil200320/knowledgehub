"use client";

import { use, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { CitationPanel } from "@/components/CitationPanel";
import {
  listMessages,
  streamMessage,
  type ChatMessage,
  type Citation,
} from "@/lib/api";

interface PendingReply {
  content: string;
  citations: Citation[];
  condensedQuery: string | null;
}

export default function ConversationPage({
  params,
}: {
  params: Promise<{ conversationId: string }>;
}) {
  const { conversationId } = use(params);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState<PendingReply | null>(null);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listMessages(conversationId)
      .then(setMessages)
      .catch((err) => setError(err.message));
  }, [conversationId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pending]);

  async function send() {
    const content = draft.trim();
    if (!content || pending) return;

    setDraft("");
    setError(null);
    setMessages((prev) => [
      ...prev,
      {
        id: `local-${Date.now()}`,
        conversation_id: conversationId,
        role: "user",
        content,
        citations: null,
        condensed_query: null,
        created_at: new Date().toISOString(),
      },
    ]);
    setPending({ content: "", citations: [], condensedQuery: null });

    try {
      await streamMessage(conversationId, content, {
        onToken: (token) =>
          setPending((prev) =>
            prev ? { ...prev, content: prev.content + token } : prev,
          ),
        onCitations: (citations) =>
          setPending((prev) => (prev ? { ...prev, citations } : prev)),
        onCondensedQuery: (query) =>
          setPending((prev) => (prev ? { ...prev, condensedQuery: query } : prev)),
        onError: (message) => setError(message),
      });
      // Re-read from the server so the thread reflects what was actually persisted.
      setMessages(await listMessages(conversationId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setPending(null);
    }
  }

  return (
    <div className="flex h-[calc(100vh-10rem)] flex-col">
      <div className="flex-1 space-y-4 overflow-y-auto pb-4">
        {messages.length === 0 && !pending && (
          <p className="py-12 text-center text-sm text-neutral-500">
            Ask a question about your uploaded documents.
          </p>
        )}

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
            content={pending.content || "…"}
            citations={pending.citations}
            condensedQuery={pending.condensedQuery}
          />
        )}

        <div ref={bottomRef} />
      </div>

      {error && (
        <p className="mb-3 rounded-md bg-red-50 px-4 py-2 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">
          {error}
        </p>
      )}

      <div className="flex gap-2 border-t border-neutral-200 pt-4 dark:border-neutral-800">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          rows={2}
          placeholder="Ask a question…"
          className="flex-1 resize-none rounded-md border border-neutral-300 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-500 dark:border-neutral-700 dark:bg-neutral-900"
        />
        <button
          onClick={send}
          disabled={!draft.trim() || pending !== null}
          className="self-end rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-neutral-700 disabled:opacity-40 dark:bg-neutral-100 dark:text-neutral-900"
        >
          {pending ? "Thinking…" : "Send"}
        </button>
      </div>
    </div>
  );
}

function MessageBubble({
  role,
  content,
  citations,
  condensedQuery,
}: {
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  condensedQuery: string | null;
}) {
  const isUser = role === "user";
  return (
    <div className={isUser ? "flex justify-end" : "flex justify-start"}>
      <div
        className={`max-w-[85%] rounded-lg px-4 py-3 ${
          isUser
            ? "bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900"
            : "border border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-900"
        }`}
      >
        {isUser ? (
          <p className="whitespace-pre-wrap text-sm leading-relaxed">{content}</p>
        ) : (
          <div className="space-y-2 text-sm leading-relaxed [&_li]:ml-4 [&_li]:list-disc [&_strong]:font-semibold [&_ul]:space-y-1">
            <ReactMarkdown>{content}</ReactMarkdown>
          </div>
        )}
        {!isUser && condensedQuery && (
          <p className="mt-2 text-xs italic text-neutral-500">
            Retrieved using: “{condensedQuery}”
          </p>
        )}
        {!isUser && <CitationPanel citations={citations} />}
      </div>
    </div>
  );
}
