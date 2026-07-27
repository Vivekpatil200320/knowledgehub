# Demo video script (5–8 min)

Setup before recording: `docker compose up --build`, wait for all three services,
then delete any existing documents so the upload is live on camera.

---

**0:00 — What it is (30s)**

"KnowledgeHub — upload documents, chat with them, every answer cites its source.
The interesting part isn't retrieval, it's making follow-up questions work, so
that's where most of this demo goes."

One screen, three panes: history on the left, conversation in the middle, the
documents you've ingested on the right. Nothing here is a separate page — you can
see what the assistant can actually read while you're asking it questions.

**0:30 — Upload (45s)**

Drag both files from `knowledgehub-backend/evals/corpus/` onto the drop zone in the
right-hand panel.

Call out: the row appears as `pending` **immediately** — upload returns before
processing finishes, because a large PDF would otherwise blow past the request
timeout. Watch it go `pending → processing → ready` with a chunk count.

Mention the two documents are deliberately similar: both have a Services section
and a Pricing section. That matters in 90 seconds.

**1:15 — Grounded answer with citations (60s)**

New chat → *"What services does Acme Cloud Platform offer?"*

As the answer lands, note the left panel: the thread just named itself from the
question and stamped the time. Worth a beat — clicking "New chat" alone doesn't
create anything; the conversation is only saved once you actually send something,
so the history never fills up with empty untitled rows.

Point out tokens streaming in, then expand a citation chip to show the exact source
chunk the answer came from — not a filename, the actual text.

**2:15 — The follow-up (the core of the demo, 2 min)**

Type just: *"What about pricing?"*

Before hitting send, say the quiet part out loud: on its own this question is
meaningless. Both documents have pricing sections. Embedding these three words
retrieves *a* pricing section — a coin flip on which.

Send it. Answer is Acme's pricing, citing only Acme.

Now point at **"Retrieved using: What is the pricing of Acme Cloud Platform?"** —
that's the condensation step. A cheap LLM call rewrote the follow-up into a
standalone query using conversation history, and *that* is what hit the vector
store. It's persisted on the message, so it's inspectable, not magic.

**4:15 — Memory that knows when to let go (1 min)**

Type: *"And what about Zenith? How much is it?"*

Condensed to "What is the pricing of Zenith?" — the new subject **replaces** Acme
rather than nesting under it.

Worth being honest here: the first version condensed this to *"the pricing of
Zenith, a service offered by Acme Cloud Platform"* — an invented relationship that
poisoned retrieval and produced a confidently wrong answer. That's why the eval
suite has cases in both directions: one where memory must be applied, one where it
must be displaced.

**5:15 — Refusal (45s)**

*"What is the capital of France?"*

Refused. Note that this isn't the model politely declining — nothing cleared the
similarity threshold, so the generation model was **never called**. A model that
isn't invoked can't hallucinate.

**6:00 — Evals (60s)**

Run `python evals/eval_pipeline.py` on camera. Ten cases, deterministic assertions,
no LLM judge. 100% retrieval precision, completeness, and contamination-free, 10/10.

Emphasise two things. First, it tracks *which document* got cited, not just whether
the answer looks right — a system with broken memory still returns a fluent pricing
answer to "what about pricing?", it just cites the wrong document. Second, the suite
was rebuilt after it passed clean through three real bugs: it now checks completeness
(every required fact present, catching terse answers) and forbidden terms (no
hallucination, no cross-document bleed), and `tests/test_eval_checks.py` proves those
checks reject the actual buggy answers from development — so the harness's teeth don't
depend on an LLM reproducing a failure on cue.

**7:00 — Close (30s)**

One `docker compose up`. Scope cuts are in the README with reasons — notably hybrid
search, skipped because a prior project of mine already evaluated it against
semantic-only on a small corpus and it didn't win.
