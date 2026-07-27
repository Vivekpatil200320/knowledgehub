"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteDocument,
  listDocuments,
  uploadDocument,
  type DocumentInfo,
} from "@/lib/api";

const STATUS_STYLES: Record<string, string> = {
  ready: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  failed: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
  pending: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  processing: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
};

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      setDocuments(await listDocuments());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load documents");
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Ingestion is a background task, so poll while anything is still in flight.
  useEffect(() => {
    const inFlight = documents.some(
      (d) => d.status === "pending" || d.status === "processing",
    );
    if (!inFlight) return;
    const timer = setInterval(refresh, 3000);
    return () => clearInterval(timer);
  }, [documents, refresh]);

  async function handleFiles(files: FileList | null) {
    if (!files?.length) return;
    setUploading(true);
    setError(null);
    try {
      for (const file of Array.from(files)) {
        await uploadDocument(file);
      }
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  async function handleDelete(id: string) {
    try {
      await deleteDocument(id);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Documents</h1>
        <p className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
          Upload PDF, TXT or Markdown files. Processing runs in the background.
        </p>
      </div>

      <label
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          handleFiles(e.dataTransfer.files);
        }}
        className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-neutral-300 bg-white px-6 py-10 text-center transition hover:border-neutral-400 dark:border-neutral-700 dark:bg-neutral-900 dark:hover:border-neutral-600"
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.txt,.md"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <span className="text-sm font-medium">
          {uploading ? "Uploading…" : "Drop files here, or click to browse"}
        </span>
        <span className="mt-1 text-xs text-neutral-500">PDF, TXT, MD · max 20MB</span>
      </label>

      {error && (
        <p className="rounded-md bg-red-50 px-4 py-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-300">
          {error}
        </p>
      )}

      <div className="overflow-hidden rounded-lg border border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-900">
        {documents.length === 0 ? (
          <p className="px-4 py-8 text-center text-sm text-neutral-500">
            No documents yet.
          </p>
        ) : (
          <ul className="divide-y divide-neutral-200 dark:divide-neutral-800">
            {documents.map((doc) => (
              <li key={doc.id} className="flex items-center gap-4 px-4 py-3">
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">{doc.filename}</p>
                  <p className="mt-0.5 text-xs text-neutral-500">
                    {doc.chunk_count !== null
                      ? `${doc.chunk_count} chunks`
                      : "Awaiting processing"}
                    {doc.status_detail && ` · ${doc.status_detail}`}
                  </p>
                </div>
                <span
                  className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
                    STATUS_STYLES[doc.status] ?? "bg-neutral-100 text-neutral-700"
                  }`}
                >
                  {doc.status}
                </span>
                <button
                  onClick={() => handleDelete(doc.id)}
                  className="text-xs text-neutral-500 transition hover:text-red-600"
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
