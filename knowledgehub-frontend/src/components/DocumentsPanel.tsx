"use client";

import { useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Loader2,
  Trash2,
  Upload,
} from "lucide-react";
import type { DocumentInfo, DocumentStatus } from "@/lib/api";

/** Status carries an icon and a word, never colour alone. */
const STATUS: Record<
  DocumentStatus,
  { label: string; className: string; Icon: typeof CheckCircle2; spin?: boolean }
> = {
  ready: {
    label: "Ready",
    className: "bg-success-soft text-success",
    Icon: CheckCircle2,
  },
  processing: {
    label: "Processing",
    className: "bg-warning-soft text-warning",
    Icon: Loader2,
    spin: true,
  },
  pending: {
    label: "Queued",
    className: "bg-warning-soft text-warning",
    Icon: Loader2,
    spin: true,
  },
  failed: {
    label: "Failed",
    className: "bg-danger-soft text-danger",
    Icon: AlertCircle,
  },
};

export function DocumentsPanel({
  documents,
  uploading,
  error,
  onUpload,
  onDelete,
}: {
  documents: DocumentInfo[];
  uploading: boolean;
  error: string | null;
  onUpload: (files: FileList | null) => void;
  onDelete: (id: string) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const readyCount = documents.filter((d) => d.status === "ready").length;

  return (
    <div className="flex h-full flex-col bg-surface-raised">
      <div className="shrink-0 border-b border-border px-4 py-3">
        <h2 className="text-sm font-semibold">Documents</h2>
        <p className="mt-0.5 text-xs text-text-subtle">
          {documents.length === 0
            ? "Nothing ingested yet"
            : `${readyCount} of ${documents.length} ready to search`}
        </p>
      </div>

      <div className="shrink-0 px-3 pt-3">
        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            onUpload(e.dataTransfer.files);
          }}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-4 py-5 text-center transition-colors duration-150 focus-within:ring-2 focus-within:ring-ring ${
            dragging
              ? "border-accent bg-accent-soft"
              : "border-border-strong hover:border-accent hover:bg-surface-sunken"
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            multiple
            accept=".pdf,.txt,.md"
            aria-label="Upload documents (PDF, TXT or Markdown)"
            className="sr-only"
            disabled={uploading}
            onChange={(e) => onUpload(e.target.files)}
          />
          {uploading ? (
            <Loader2 aria-hidden="true" className="size-5 animate-spin text-accent" />
          ) : (
            <Upload aria-hidden="true" className="size-5 text-text-subtle" />
          )}
          <span className="mt-2 text-xs font-medium">
            {uploading ? "Uploading…" : "Drop files or click to browse"}
          </span>
          <span className="mt-0.5 text-[11px] text-text-subtle">
            PDF, TXT, MD · max 20MB
          </span>
        </label>
      </div>

      {error && (
        <p
          role="alert"
          className="mx-3 mt-3 flex items-start gap-2 rounded-lg bg-danger-soft px-3 py-2 text-xs text-danger"
        >
          <AlertCircle aria-hidden="true" className="mt-px size-3.5 shrink-0" />
          <span>{error}</span>
        </p>
      )}

      <div className="scrollbar-slim min-h-0 flex-1 overflow-y-auto px-3 py-3">
        {documents.length === 0 ? (
          <p className="px-1 py-6 text-center text-xs text-text-subtle">
            Upload a document to start asking questions about it.
          </p>
        ) : (
          <ul className="space-y-1.5">
            {documents.map((doc) => {
              const status = STATUS[doc.status] ?? STATUS.pending;
              const { Icon } = status;
              return (
                <li
                  key={doc.id}
                  className="group/doc rounded-lg border border-border bg-surface p-2.5"
                >
                  <div className="flex items-start gap-2">
                    <p
                      className="min-w-0 flex-1 truncate text-xs font-medium"
                      title={doc.filename}
                    >
                      {doc.filename}
                    </p>
                    <button
                      type="button"
                      onClick={() => onDelete(doc.id)}
                      aria-label={`Delete ${doc.filename}`}
                      className="grid size-6 shrink-0 place-items-center rounded text-text-subtle opacity-0 transition duration-150 hover:bg-danger-soft hover:text-danger focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover/doc:opacity-100"
                    >
                      <Trash2 aria-hidden="true" className="size-3.5" />
                    </button>
                  </div>

                  <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                    <span
                      className={`inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 text-[10px] font-medium ${status.className}`}
                    >
                      <Icon
                        aria-hidden="true"
                        className={`size-2.5 ${status.spin ? "animate-spin" : ""}`}
                      />
                      {status.label}
                    </span>
                    {doc.chunk_count !== null && (
                      <span className="text-[10px] tabular-nums text-text-subtle">
                        {doc.chunk_count} chunks
                      </span>
                    )}
                  </div>

                  {/* Failures state the reason inline — the recovery path is
                      re-uploading a supported file, not a generic retry. */}
                  {doc.status === "failed" && doc.status_detail && (
                    <p className="mt-1.5 rounded bg-danger-soft px-2 py-1 text-[10px] leading-relaxed text-danger">
                      {doc.status_detail}
                    </p>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
