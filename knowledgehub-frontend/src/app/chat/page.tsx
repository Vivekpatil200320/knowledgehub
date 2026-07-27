"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { createConversation, listConversations, type Conversation } from "@/lib/api";

export default function ChatIndexPage() {
  const router = useRouter();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listConversations()
      .then(setConversations)
      .catch((err) => setError(err.message));
  }, []);

  async function startConversation() {
    try {
      const conversation = await createConversation();
      router.push(`/chat/${conversation.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start a conversation");
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Conversations</h1>
          <p className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
            Ask questions across your uploaded documents.
          </p>
        </div>
        <button
          onClick={startConversation}
          className="rounded-md bg-neutral-900 px-4 py-2 text-sm font-medium text-white transition hover:bg-neutral-700 dark:bg-neutral-100 dark:text-neutral-900 dark:hover:bg-neutral-300"
        >
          New conversation
        </button>
      </div>

      {error && (
        <p className="rounded-md bg-red-50 px-4 py-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">
          {error}
        </p>
      )}

      <div className="overflow-hidden rounded-lg border border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-900">
        {conversations.length === 0 ? (
          <p className="px-4 py-8 text-center text-sm text-neutral-500">
            No conversations yet.
          </p>
        ) : (
          <ul className="divide-y divide-neutral-200 dark:divide-neutral-800">
            {conversations.map((conversation) => (
              <li key={conversation.id}>
                <Link
                  href={`/chat/${conversation.id}`}
                  className="block px-4 py-3 transition hover:bg-neutral-50 dark:hover:bg-neutral-800"
                >
                  <p className="text-sm font-medium">
                    {conversation.title ?? "Untitled conversation"}
                  </p>
                  <p className="mt-0.5 text-xs text-neutral-500">
                    {new Date(conversation.created_at).toLocaleString()}
                  </p>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
