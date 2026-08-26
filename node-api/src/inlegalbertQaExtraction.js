// Experimental route for manually testing InLegalBERT against real
// documents, per explicit user request. Two things worth knowing before
// trusting this route:
//
// 1. law-ai/InLegalBERT itself is only a fill-mask model — it has no
//    question-answering head at all, so it can't be used for this
//    pipeline as-is. The only QA-capable checkpoint found on the Hub is a
//    community fine-tune, kinshuk-h/InLegalBERT-cbp-lkg-qa-triples-w-
//    context-finetuned, trained on Indian legal knowledge-graph QA
//    triples — a narrow, unvetted task, not general document reading.
// 2. No ONNX build of it exists anywhere, so it was converted locally with
//    optimum's ONNX exporter (see models/inlegalbert-qa/) and is loaded
//    from disk here, not downloaded from the Hub — this repo isn't
//    published anywhere else.
//
// Tested against a real document from this project (Town Villas caption)
// before wiring this route in: both the plaintiff and case-number
// questions came back with the wrong, oversized span AND a confidence
// score near zero (~1e-8, ~1e-7) — an order of magnitude worse than
// app/qaExtraction.js's distilbert-squad model on the same text. Kept as
// a separate, clearly-labeled route rather than replacing the main
// pipeline, so it can be compared side-by-side rather than trusted.

import path from "path";
import { fileURLToPath } from "url";
import { pipeline, env } from "@huggingface/transformers";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// `env` is a single shared object across the whole process — every
// pipeline() call anywhere in the app reads it, not just this one. Setting
// `env.allowRemoteModels = false` here used to disable remote model
// resolution GLOBALLY, which broke qaExtraction.js's and gemma3nExtraction.js's
// models too (they'd try their local path first under this same
// localModelPath, fail, and then — with allowRemoteModels now false for
// the whole process — throw instead of falling back to their normal
// cache/download path). Confirmed as a real bug from a real crash: a fresh
// Windows install failed at startup trying to load
// Xenova/distilbert-base-cased-distilled-squad because of this exact
// global flag, not anything Windows-specific. `env.localModelPath` alone
// is harmless to leave set — pipeline() always tries the local path first
// for every model regardless, and other models just fail that one local
// lookup silently and fall through to their normal resolution, as long as
// `allowRemoteModels` stays at its default (true).
env.localModelPath = path.join(__dirname, "..", "models");

let qaPipelinePromise = null;
function getQaPipeline() {
  if (!qaPipelinePromise) {
    // local_files_only scoped to just this call, not the whole process —
    // this model only exists locally (see file header), so there's nothing
    // to gain by letting it fall through to a remote fetch attempt, but
    // that's a per-call choice, not something that should affect any other
    // model's ability to resolve remotely.
    qaPipelinePromise = pipeline("question-answering", "inlegalbert-qa", {
      dtype: "fp32",
      local_files_only: true,
    });
  }
  return qaPipelinePromise;
}

export async function warmUp() {
  await getQaPipeline();
}

// Same chunking rationale as qaExtraction.js — transformers.js's QA
// pipeline doesn't stride over long context automatically.
const CONTEXT_CHUNK_CHARS = 1500;
const CONTEXT_CHUNK_OVERLAP = 200;

function chunkContext(text) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + CONTEXT_CHUNK_CHARS, text.length);
    chunks.push(text.slice(start, end));
    if (end >= text.length) break;
    start = end - CONTEXT_CHUNK_OVERLAP;
  }
  return chunks;
}

export async function answerQuestionsWithInLegalBert(text, questions) {
  if (!text.trim() || !questions.length) return [];

  const qa = await getQaPipeline();
  const answers = [];

  for (const rawQuestion of questions) {
    const question = rawQuestion.trim();
    if (!question) continue;

    let best = null;
    for (const chunk of chunkContext(text)) {
      const result = await qa(question, chunk);
      if (!best || result.score > best.score) best = result;
    }

    answers.push({
      question,
      answer: best.answer,
      confidence: best.score,
    });
  }
  return answers;
}
