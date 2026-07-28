"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Menu, PanelRight } from "lucide-react";
import { ChatPanel, type PendingReply } from "@/components/ChatPanel";
import { ConversationSidebar } from "@/components/ConversationSidebar";
import { DocumentsPanel } from "@/components/DocumentsPanel";
import { Drawer } from "@/components/Drawer";
import {
  createConversation,
  deleteConversation,
  deleteDocument,
  listConversations,
  listDocuments,
  listMessages,
  streamMessage,
  uploadDocument,
  type ChatMessage,
  type Conversation,
  type DocumentInfo,
} from "@/lib/api";

function Workspace() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const selectedId = searchParams.get("c");

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState<PendingReply | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);

  const [chatError, setChatError] = useState<string | null>(null);
  const [docError, setDocError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const [historyOpen, setHistoryOpen] = useState(false);
  const [docsOpen, setDocsOpen] = useState(false);

  // Guards against a stream outliving the conversation it was sent for: switching
  // conversations mid-stream must not let that stream's tokens (or its final
  // re-fetched thread) render into whatever conversation is selected afterward.
  // The AbortController stops the network read; the generation counter is a second,
  // independent guard for the narrow race where a response finishes right as the
  // abort fires — belt and suspenders, since either one alone leaves that window open.
  const streamGeneration = useRef(0);
  const abortController = useRef<AbortController | null>(null);

  const refreshConversations = useCallback(async () => {
    try {
      setConversations(await listConversations());
    } catch (err) {
      setChatError(err instanceof Error ? err.message : "Could not load history");
    }
  }, []);

  const refreshDocuments = useCallback(async () => {
    try {
      setDocuments(await listDocuments());
    } catch (err) {
      setDocError(err instanceof Error ? err.message : "Could not load documents");
    }
  }, []);

  useEffect(() => {
    refreshConversations();
    refreshDocuments();
  }, [refreshConversations, refreshDocuments]);

  // Belt-and-suspenders: abort any in-flight stream if the page unmounts entirely,
  // not just on a same-page conversation switch.
  useEffect(() => {
    return () => abortController.current?.abort();
  }, []);

  // Ingestion runs in the background, so poll while anything is still in flight.
  useEffect(() => {
    const inFlight = documents.some(
      (d) => d.status === "pending" || d.status === "processing",
    );
    if (!inFlight) return;
    const timer = setInterval(refreshDocuments, 3000);
    return () => clearInterval(timer);
  }, [documents, refreshDocuments]);

  // The selected conversation lives in the URL, so refresh and back/forward work.
  useEffect(() => {
    if (!selectedId) {
      setMessages([]);
      return;
    }
    let cancelled = false;
    listMessages(selectedId)
      .then((loaded) => {
        if (!cancelled) setMessages(loaded);
      })
      .catch((err) => {
        if (!cancelled) setChatError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  function select(id: string | null) {
    // Any in-flight stream belongs to the conversation being left; abort it so its
    // tokens can't keep arriving and rendering into the one being switched to.
    abortController.current?.abort();
    setPending(null);
    router.replace(id ? `/?c=${id}` : "/", { scroll: false });
    setChatError(null);
    setHistoryOpen(false);
  }

  async function send(content: string) {
    setChatError(null);

    // Conversations are created lazily: clicking "New chat" alone must not leave
    // an empty untitled row behind in the sidebar.
    let conversationId = selectedId;
    if (!conversationId) {
      try {
        conversationId = (await createConversation()).id;
        router.replace(`/?c=${conversationId}`, { scroll: false });
      } catch (err) {
        setChatError(err instanceof Error ? err.message : "Could not start a chat");
        return;
      }
    }

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

    const myGeneration = ++streamGeneration.current;
    const isCurrent = () => streamGeneration.current === myGeneration;
    const controller = new AbortController();
    abortController.current = controller;

    try {
      await streamMessage(
        conversationId,
        content,
        {
          onToken: (token) =>
            isCurrent() &&
            setPending((prev) => (prev ? { ...prev, content: prev.content + token } : prev)),
          onCitations: (citations) =>
            isCurrent() && setPending((prev) => (prev ? { ...prev, citations } : prev)),
          onCondensedQuery: (query) =>
            isCurrent() &&
            setPending((prev) => (prev ? { ...prev, condensedQuery: query } : prev)),
          onError: (message) => isCurrent() && setChatError(message),
        },
        controller.signal,
      );
      // Re-read so the thread reflects what was actually persisted — but only if
      // this is still the conversation on screen; otherwise this would overwrite
      // whatever thread the user has since switched to.
      if (isCurrent()) setMessages(await listMessages(conversationId));
    } catch (err) {
      // An abort means the user navigated away — that's not a failure to report.
      if (err instanceof DOMException && err.name === "AbortError") return;
      if (isCurrent()) {
        setChatError(err instanceof Error ? err.message : "Something went wrong");
      }
    } finally {
      if (isCurrent()) {
        setPending(null);
        // The first message names the thread, so the sidebar needs re-reading.
        refreshConversations();
      }
    }
  }

  async function removeConversation(id: string) {
    try {
      await deleteConversation(id);
      if (id === selectedId) select(null);
      await refreshConversations();
    } catch (err) {
      setChatError(err instanceof Error ? err.message : "Could not delete chat");
    }
  }

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    setUploading(true);
    setDocError(null);
    try {
      for (const file of Array.from(files)) {
        await uploadDocument(file);
      }
      await refreshDocuments();
    } catch (err) {
      setDocError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function removeDocument(id: string) {
    try {
      await deleteDocument(id);
      await refreshDocuments();
    } catch (err) {
      setDocError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  const activeTitle =
    conversations.find((c) => c.id === selectedId)?.title ?? "New chat";
  const readyDocuments = documents
    .filter((d) => d.status === "ready")
    .map((d) => ({ filename: d.filename, title: d.title }));

  const history = (
    <ConversationSidebar
      conversations={conversations}
      selectedId={selectedId}
      isDraft={selectedId === null}
      onSelect={select}
      onNew={() => select(null)}
      onDelete={removeConversation}
    />
  );

  const docs = (
    <DocumentsPanel
      documents={documents}
      uploading={uploading}
      error={docError}
      onUpload={upload}
      onDelete={removeDocument}
    />
  );

  return (
    <div className="grid h-dvh grid-cols-1 grid-rows-[auto_1fr] lg:grid-cols-[280px_1fr] xl:grid-cols-[280px_1fr_320px]">
      <header className="col-span-full flex shrink-0 items-center gap-2 border-b border-border bg-surface-raised px-3 py-2">
        <button
          type="button"
          onClick={() => setHistoryOpen(true)}
          aria-label="Open chat history"
          className="grid size-9 place-items-center rounded-lg text-text-muted transition-colors duration-150 hover:bg-surface-sunken hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring lg:hidden"
        >
          <Menu aria-hidden="true" className="size-4" />
        </button>

        <span className="font-semibold tracking-tight">KnowledgeHub</span>

        <span
          className="ml-2 hidden min-w-0 flex-1 truncate text-sm text-text-subtle sm:block"
          title={activeTitle}
        >
          {activeTitle}
        </span>

        <button
          type="button"
          onClick={() => setDocsOpen(true)}
          aria-label="Open documents"
          className="ml-auto grid size-9 place-items-center rounded-lg text-text-muted transition-colors duration-150 hover:bg-surface-sunken hover:text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring xl:hidden"
        >
          <PanelRight aria-hidden="true" className="size-4" />
        </button>
      </header>

      <aside className="hidden min-h-0 border-r border-border lg:block">{history}</aside>

      <main className="min-h-0 min-w-0">
        <ChatPanel
          messages={messages}
          pending={pending}
          error={chatError}
          readyDocuments={readyDocuments}
          onSend={send}
        />
      </main>

      <aside className="hidden min-h-0 border-l border-border xl:block">{docs}</aside>

      <Drawer
        open={historyOpen}
        side="left"
        title="Chat history"
        onClose={() => setHistoryOpen(false)}
      >
        {history}
      </Drawer>
      <Drawer
        open={docsOpen}
        side="right"
        title="Documents"
        onClose={() => setDocsOpen(false)}
      >
        {docs}
      </Drawer>
    </div>
  );
}

export default function Page() {
  // useSearchParams needs a Suspense boundary for prerendering.
  return (
    <Suspense fallback={<div className="h-dvh bg-surface" />}>
      <Workspace />
    </Suspense>
  );
}
