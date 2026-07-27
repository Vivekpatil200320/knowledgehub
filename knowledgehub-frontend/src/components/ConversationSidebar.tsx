"use client";

import { useMemo, useState } from "react";
import { MessageSquarePlus, Trash2 } from "lucide-react";
import type { Conversation } from "@/lib/api";
import {
  GROUP_ORDER,
  fullTimestamp,
  groupFor,
  isoTimestamp,
  timeLabel,
  type DateGroup,
} from "@/lib/format";

export function ConversationSidebar({
  conversations,
  selectedId,
  isDraft,
  onSelect,
  onNew,
  onDelete,
}: {
  conversations: Conversation[];
  selectedId: string | null;
  isDraft: boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}) {
  const [confirmingId, setConfirmingId] = useState<string | null>(null);

  const grouped = useMemo(() => {
    const buckets = new Map<DateGroup, Conversation[]>();
    for (const conversation of conversations) {
      const group = groupFor(conversation.last_message_at);
      const bucket = buckets.get(group);
      if (bucket) bucket.push(conversation);
      else buckets.set(group, [conversation]);
    }
    return GROUP_ORDER.filter((g) => buckets.has(g)).map((g) => ({
      group: g,
      items: buckets.get(g)!,
    }));
  }, [conversations]);

  return (
    <div className="flex h-full flex-col bg-surface-raised">
      <div className="shrink-0 border-b border-border p-3">
        <button
          type="button"
          onClick={onNew}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-3 py-2.5 text-sm font-medium text-accent-fg transition-colors duration-150 hover:bg-accent-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-surface-raised"
        >
          <MessageSquarePlus aria-hidden="true" className="size-4" />
          New chat
        </button>
      </div>

      <nav
        aria-label="Chat history"
        className="scrollbar-slim min-h-0 flex-1 overflow-y-auto px-2 py-3"
      >
        {isDraft && (
          <p className="mb-2 rounded-lg border border-dashed border-border-strong px-3 py-2 text-xs text-text-subtle">
            New chat — send a message to save it
          </p>
        )}

        {conversations.length === 0 && !isDraft ? (
          <p className="px-3 py-6 text-center text-xs text-text-subtle">
            No conversations yet. Start one above.
          </p>
        ) : (
          grouped.map(({ group, items }) => (
            <section key={group} className="mb-4 last:mb-0">
              <h2 className="px-3 pb-1.5 text-[11px] font-semibold uppercase tracking-wider text-text-subtle">
                {group}
              </h2>
              <ul className="space-y-0.5">
                {items.map((conversation) => {
                  const isActive = conversation.id === selectedId && !isDraft;
                  const isConfirming = confirmingId === conversation.id;
                  const title = conversation.title ?? "Untitled chat";

                  return (
                    <li key={conversation.id} className="group/row relative">
                      <button
                        type="button"
                        onClick={() => onSelect(conversation.id)}
                        aria-current={isActive ? "true" : undefined}
                        title={title}
                        className={`w-full rounded-lg py-2 pl-3 pr-9 text-left transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-surface-raised ${
                          isActive
                            ? "bg-accent-soft"
                            : "hover:bg-surface-sunken"
                        }`}
                      >
                        {/* Active state carries a bar as well as a tint, so it
                            doesn't rely on colour alone. */}
                        {isActive && (
                          <span
                            aria-hidden="true"
                            className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-accent"
                          />
                        )}
                        <span
                          className={`block truncate text-sm ${
                            isActive ? "font-medium text-text" : "text-text-muted"
                          }`}
                        >
                          {title}
                        </span>
                        <span className="mt-0.5 flex items-center gap-1.5 text-[11px] text-text-subtle">
                          <time
                            dateTime={isoTimestamp(conversation.last_message_at)}
                            title={fullTimestamp(conversation.last_message_at)}
                          >
                            {timeLabel(conversation.last_message_at)}
                          </time>
                          {conversation.message_count > 0 && (
                            <>
                              <span aria-hidden="true">·</span>
                              <span className="tabular-nums">
                                {conversation.message_count}{" "}
                                {conversation.message_count === 1 ? "message" : "messages"}
                              </span>
                            </>
                          )}
                        </span>
                      </button>

                      {/* Kept in the DOM (not hover-mounted) so it stays reachable
                          by keyboard; only its opacity is hover-dependent. */}
                      <button
                        type="button"
                        onClick={() =>
                          isConfirming
                            ? (onDelete(conversation.id), setConfirmingId(null))
                            : setConfirmingId(conversation.id)
                        }
                        onBlur={() => setConfirmingId(null)}
                        aria-label={
                          isConfirming
                            ? `Confirm delete “${title}”`
                            : `Delete “${title}”`
                        }
                        className={`absolute right-1 top-1.5 grid size-7 place-items-center rounded-md transition duration-150 focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover/row:opacity-100 ${
                          isConfirming
                            ? "bg-danger-soft text-danger opacity-100"
                            : "text-text-subtle opacity-0 hover:bg-surface-sunken hover:text-danger"
                        }`}
                      >
                        <Trash2 aria-hidden="true" className="size-3.5" />
                      </button>

                      {isConfirming && (
                        <span
                          role="status"
                          className="absolute -bottom-0.5 right-9 z-10 rounded bg-danger px-1.5 py-0.5 text-[10px] font-medium text-white"
                        >
                          Click again
                        </span>
                      )}
                    </li>
                  );
                })}
              </ul>
            </section>
          ))
        )}
      </nav>
    </div>
  );
}
