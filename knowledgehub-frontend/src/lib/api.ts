export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type DocumentStatus = "pending" | "processing" | "ready" | "failed";

export interface DocumentInfo {
  id: string;
  filename: string;
  content_type: string;
  status: DocumentStatus;
  status_detail: string | null;
  chunk_count: number | null;
  created_at: string;
  /** The document's own opening line (name/heading), looked up once it's ready.
   *  Prefer this over `filename` when building a starter question — a résumé named
   *  "candidate-profile.pdf" never contains the phrase "candidate profile", so a
   *  filename-derived question about it can score below the refusal threshold. */
  title: string | null;
}

export interface Citation {
  document_id: string;
  filename: string;
  chunk_index: number;
  snippet: string;
  score: number;
}

export interface ChatMessage {
  id: string;
  conversation_id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[] | null;
  condensed_query: string | null;
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string | null;
  created_at: string;
  /** Newest message time, falling back to created_at for an unused thread. */
  last_message_at: string;
  message_count: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}/api${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed (${response.status})`);
  }
  return response.status === 204 ? (undefined as T) : response.json();
}

export const listDocuments = () => request<DocumentInfo[]>("/documents");

export const deleteDocument = (id: string) =>
  request<void>(`/documents/${id}`, { method: "DELETE" });

export function uploadDocument(file: File) {
  const form = new FormData();
  form.append("file", file);
  return request<DocumentInfo>("/documents", { method: "POST", body: form });
}

export const listConversations = () => request<Conversation[]>("/conversations");

export const createConversation = () =>
  request<Conversation>("/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });

export const deleteConversation = (id: string) =>
  request<void>(`/conversations/${id}`, { method: "DELETE" });

export const listMessages = (conversationId: string) =>
  request<ChatMessage[]>(`/conversations/${conversationId}/messages`);

export interface StreamHandlers {
  onToken: (token: string) => void;
  onCitations: (citations: Citation[]) => void;
  onCondensedQuery?: (query: string) => void;
  onError?: (message: string) => void;
}

/** Consumes the SSE chat stream, dispatching each event type to its handler. */
export async function streamMessage(
  conversationId: string,
  content: string,
  handlers: StreamHandlers,
): Promise<void> {
  const response = await fetch(
    `${API_URL}/api/conversations/${conversationId}/messages/stream`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    },
  );

  if (!response.ok || !response.body) {
    throw new Error(`Stream failed (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const line = frame.trim();
      if (!line.startsWith("data:")) continue;

      const event = JSON.parse(line.slice(5).trim());
      if (event.type === "token") handlers.onToken(event.data);
      else if (event.type === "citations") handlers.onCitations(event.data);
      else if (event.type === "condensed_query") handlers.onCondensedQuery?.(event.data);
      else if (event.type === "error") handlers.onError?.(event.data);
    }
  }
}
