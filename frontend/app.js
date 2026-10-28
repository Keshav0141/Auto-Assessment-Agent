"use strict";

/* Examiner's desk: upload -> check extracted text -> mark (streamed progress) -> marked script. */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const INPUT_KEYS = ["question_paper", "answer_sheet", "rubric", "model_answers"];
const IMAGE_RE = /\.(png|jpe?g|webp|bmp|tiff?)$/i;
const ACCEPT_RE = /\.(txt|md|pdf|png|jpe?g|webp|bmp|tiff?)$/i;
const MAX_IMAGE_SIDE = 2000;
const JPEG_QUALITY = 0.85;
const MAX_FILE_BYTES = 4 * 1024 * 1024;

const FLAG_LABELS = {
  prompt_injection: "Instructions aimed at the grader",
  blank: "Blank answer",
  off_topic: "Off topic",
  restates_question: "Repeats the question",
  gibberish: "Unreadable text",
  language_mismatch: "Not in the language the rubric asks for",
  illegible: "Illegible after OCR",
};
const LANG_CODES = {
  English: "en", Hindi: "hi", Hinglish: "hi-Latn", Tamil: "ta", Telugu: "te", Kannada: "kn",
  Malayalam: "ml", Bengali: "bn", Gujarati: "gu", Punjabi: "pa", Odia: "or", Urdu: "ur",
};

const state = {
  health: null,
  weights: { agreement: 1, variance: 1, semantic: 0, cross: 1 },
  result: null,
  thumbs: {},
};

/* ---------- helpers ---------- */

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function fmt(n) {
  if (n == null || Number.isNaN(Number(n))) return "–";
  const v = Math.round(Number(n) * 10) / 10;
  return Number.isInteger(v) ? String(v) : v.toFixed(1);
}
const pct = (v) => `${Math.round(Number(v) * 100)}%`;
const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

/* ---------- theme ---------- */

function effectiveTheme() {
  const set = document.documentElement.dataset.theme;
  if (set) return set;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function syncThemeButton() {
  const dark = effectiveTheme() === "dark";
  const btn = $("#theme-toggle");
  // Constant name; the on/off state is carried by aria-pressed (and the pressed style).
  btn.setAttribute("aria-pressed", String(dark));
  btn.title = dark ? "Dark desk is on" : "Dark desk is off";
}
function toggleTheme() {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (e) { /* storage unavailable */ }
  syncThemeButton();
}

/* ---------- health / status line ---------- */

async function loadHealth() {
  const line = $("#status-line");
  try {
    const res = await fetch("/api/health");
    const h = await res.json();
    state.health = h;
    if (h.weights) state.weights = h.weights;
    if (h.samples_per_question != null) $("#n_samples").value = Math.min(10, h.samples_per_question);
    if (h.confidence_threshold != null) $("#threshold").value = h.confidence_threshold;
    const parts = [];
    if (h.mock) {
      parts.push(`<span class="mock">Mock marking: no LLM key is set, so marks come from a keyword heuristic.</span>`);
    } else {
      parts.push(`<span>Marker <b>${esc(h.model)}</b></span>`);
    }
    parts.push(`<span>${esc(h.samples_per_question)} markings per answer</span>`);
    if (h.second_examiner_active) parts.push(`<span>Second examiner <b>${esc(h.judge2_model)}</b></span>`);
    else parts.push(`<span>No second examiner</span>`);
    if (h.second_examiner_active && h.adjudicate) parts.push(`<span>Moderator <b>${esc(h.referee_model)}</b></span>`);
    if (h.fallback_model && !h.mock) parts.push(`<span>Backup <b>${esc(h.fallback_model)}</b></span>`);
    line.innerHTML = parts.join("");
  } catch (e) {
    line.innerHTML = `<span class="mock">The server could not be reached.</span>`;
  }
}

async function loadSampleList() {
  try {
    const res = await fetch("/api/samples");
    if (!res.ok) return;
    const list = await res.json();
    const sel = $("#sample-select");
    sel.innerHTML = list.map((s) => `<option value="${esc(s.name)}">${esc(s.label)}</option>`).join("");
  } catch (e) { /* keep the static options */ }
}

async function loadSample() {
  const name = $("#sample-select").value;
  try {
    const res = await fetch(`/api/sample?name=${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error();
    const s = await res.json();
    for (const key of INPUT_KEYS) {
      $(`#${key}_text`).value = s[`${key}_text`] || "";
      clearFileStatus(key);
    }
    announce(`Loaded the ${s.label} sample.`);
  } catch (e) {
    showError("The sample could not be loaded. Is the server running?");
  }
}

/* ---------- uploads + extraction ---------- */

function isImageFile(f) {
  return (f.type && f.type.startsWith("image/")) || IMAGE_RE.test(f.name || "");
}

function loadImageEl(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("unreadable image")); };
    img.src = url;
  });
}

async function downscaleImage(file) {
  const img = await loadImageEl(file);
  const scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.max(1, Math.round(img.naturalWidth * scale));
  const h = Math.max(1, Math.round(img.naturalHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", JPEG_QUALITY));
  if (!blob) return file;
  const base = (file.name || "photo").replace(/\.[^.]+$/, "");
  return new File([blob], `${base}.jpg`, { type: "image/jpeg" });
}

function sheetOf(key) { return $(`.sheet[data-input="${key}"]`); }

function clearFileStatus(key) {
  if (state.thumbs[key]) { URL.revokeObjectURL(state.thumbs[key]); delete state.thumbs[key]; }
  $(".file-status", sheetOf(key)).innerHTML = "";
}

function setFileStatus(key, file, message, kind) {
  const box = $(".file-status", sheetOf(key));
  let thumb;
  if (file && isImageFile(file)) {
    if (state.thumbs[key]) URL.revokeObjectURL(state.thumbs[key]);
    state.thumbs[key] = URL.createObjectURL(file);
    thumb = `<img class="thumb" src="${state.thumbs[key]}" alt="Thumbnail of ${esc(file.name)}" />`;
  } else {
    const ext = ((file && file.name) || "").split(".").pop().toUpperCase().slice(0, 4) || "FILE";
    thumb = `<span class="thumb thumb--doc" aria-hidden="true">${esc(ext)}</span>`;
  }
  const cls = kind === "busy" ? "is-busy" : kind === "error" ? "is-error" : "";
  box.innerHTML = `${thumb}<div class="file-meta"><div class="file-name">${esc(file ? file.name : "")}</div>
    <div class="file-state ${cls}">${message}</div></div>`;
}

async function handleFile(key, file) {
  clearError();
  if (!ACCEPT_RE.test(file.name || "") && !isImageFile(file)) {
    setFileStatus(key, file, "This file type isn't supported. Use text, PDF or a photo (JPG/PNG/WebP).", "error");
    return;
  }
  let toSend = file;
  if (isImageFile(file)) {
    try { toSend = await downscaleImage(file); } catch (e) {
      setFileStatus(key, file, "This image could not be opened. Try another photo.", "error");
      return;
    }
  }
  if (toSend.size > MAX_FILE_BYTES) {
    setFileStatus(key, toSend, "This file is larger than 4 MB. Try fewer pages or a smaller photo.", "error");
    return;
  }
  setFileStatus(key, toSend, isImageFile(toSend) ? "Reading the photo" : "Reading the file", "busy");
  const fd = new FormData();
  fd.append(key, toSend, toSend.name);
  try {
    const res = await fetch("/api/extract", { method: "POST", body: fd });
    let data = {};
    try { data = await res.json(); } catch (e) { data = {}; }
    if (!res.ok) throw new Error(friendlyError(res.status, data.detail));
    const entry = (data.inputs || {})[key];
    if (!entry) throw new Error("Nothing came back for this file.");
    if (entry.error) {
      setFileStatus(key, toSend, esc(entry.error), "error");
      return;
    }
    const area = $(`#${key}_text`);
    area.value = entry.text || "";
    const msg = entry.warning
      ? `<span class="hl">${esc(entry.warning)}</span>`
      : "Text read from the file. You can edit it below before marking.";
    setFileStatus(key, toSend, msg);
    announce(`${toSend.name} was read. ${entry.warning || "Its text is in the box below."}`);
  } catch (e) {
    setFileStatus(key, toSend, esc(e.message || "The file could not be read."), "error");
  }
}

function setupSheet(key) {
  const sheet = sheetOf(key);
  const fileInput = $('input[data-role="file"]', sheet);
  const camInput = $('input[data-role="camera"]', sheet);
  $('[data-action="choose"]', sheet).addEventListener("click", () => fileInput.click());
  $('[data-action="camera"]', sheet).addEventListener("click", () => camInput.click());
  for (const inp of [fileInput, camInput]) {
    inp.addEventListener("change", () => {
      const f = inp.files && inp.files[0];
      inp.value = "";
      if (f) handleFile(key, f);
    });
  }
  const drop = $("[data-drop]", sheet);
  for (const ev of ["dragenter", "dragover"]) {
    sheet.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("is-over"); });
  }
  sheet.addEventListener("dragleave", (e) => {
    if (!sheet.contains(e.relatedTarget)) drop.classList.remove("is-over");
  });
  sheet.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("is-over");
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) handleFile(key, f);
  });
}

/* ---------- errors + announcements ---------- */

function friendlyError(status, detail) {
  if (status === 413) return "That file is too large. Please upload files up to 4 MB each.";
  if (status === 400) return detail || "The inputs could not be read. Check the text in each box and try again.";
  if (status === 502) return detail || "The marking model is unavailable right now. Please try again in a moment.";
  return detail || `Something went wrong (HTTP ${status}). Please try again.`;
}
function showError(msg) {
  const box = $("#error-box");
  box.innerHTML = `<strong>Couldn't mark the script.</strong> ${esc(msg)}`;
  box.hidden = false;
}
function clearError() { $("#error-box").hidden = true; }
function announce(msg) { $("#progress-live").textContent = msg; }

/* ---------- progress timeline (real stage events from the server) ---------- */

const CHECK = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7"/></svg>';
const DASH = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 12h10"/></svg>';
const CROSS = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 7l10 10M17 7 7 17"/></svg>';

const timeline = {
  t0: 0, timer: null, total: 0, samples: 5, second: false, moderation: false,
  graded: new Set(), seconded: new Set(), modStarted: new Set(), modDone: new Set(), done: new Set(),
  phase: "reading", failed: false, finished: false, lastAnnounced: "",

  start(samples) {
    Object.assign(this, {
      total: 0, samples, second: false, moderation: false, phase: "reading", failed: false, finished: false,
      graded: new Set(), seconded: new Set(), modStarted: new Set(), modDone: new Set(), done: new Set(),
    });
    $("#progress").hidden = false;
    $(".progress-title").textContent = "Marking in progress";
    this.t0 = Date.now();
    this.tick();
    clearInterval(this.timer);
    this.timer = setInterval(() => this.tick(), 1000);
    this.draw();
  },
  tick() {
    const s = Math.floor((Date.now() - this.t0) / 1000);
    $("#elapsed").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  },
  handle(ev) {
    if (ev.type === "start") {
      this.total = ev.questions.length;
      this.samples = ev.samples;
      this.second = !!ev.second_examiner;
      this.moderation = !!ev.moderation;
      this.phase = "grading";
    } else if (ev.type === "stage") {
      const q = ev.question;
      if (ev.stage === "graded") this.graded.add(q);
      if (ev.stage === "second_done") this.seconded.add(q);
      if (ev.stage === "moderation") this.modStarted.add(q);
      if (ev.stage === "moderation_done") this.modDone.add(q);
      if (ev.stage === "done") { this.done.add(q); this.graded.add(q); this.seconded.add(q); }
    }
    this.draw();
  },
  finish() {
    this.finished = true;
    clearInterval(this.timer);
    $(".progress-title").textContent = "Marking finished";
    this.draw();
  },
  fail() {
    this.failed = true;
    clearInterval(this.timer);
    $(".progress-title").textContent = "Marking stopped";
    this.draw();
  },
  steps() {
    const n = this.total;
    const of = (k) => (n ? `${k} of ${n} questions` : "");
    const allDone = n > 0 && this.done.size === n;
    const steps = [];
    steps.push({
      label: "Reading",
      st: this.phase === "reading" ? (this.failed ? "failed" : "active") : "done",
      note: this.phase === "reading" ? "Parsing the papers" : plural(n, "question found", "questions found"),
    });
    const gradedN = this.graded.size;
    steps.push({
      label: `Grading ×${this.samples}`,
      st: this.phase === "reading" ? "pending" : gradedN === n ? "done" : (this.failed ? "failed" : "active"),
      note: this.phase === "reading" ? "Waiting" : of(gradedN),
    });
    if (!this.second && this.phase !== "reading") {
      steps.push({ label: "Second examiner", st: "skipped", note: "Not configured" });
    } else {
      const k = this.seconded.size;
      steps.push({
        label: "Second examiner",
        st: this.phase === "reading" || (k === 0 && gradedN === 0) ? "pending" : k === n ? "done" : (this.failed ? "failed" : "active"),
        note: this.phase === "reading" ? "Waiting" : of(k),
      });
    }
    if (!this.moderation && this.phase !== "reading") {
      steps.push({ label: "Moderation", st: "skipped", note: "No second examiner to compare with" });
    } else {
      const started = this.modStarted.size;
      const open = started - this.modDone.size;
      let st = "pending";
      let note = "Only if the examiners disagree";
      if (allDone) { st = "done"; note = started ? `${plural(started, "answer", "answers")} moderated` : "None needed"; }
      else if (open > 0) { st = this.failed ? "failed" : "active"; note = `Moderating ${plural(open, "answer", "answers")}`; }
      else if (started) { st = "active"; note = `${plural(started, "answer", "answers")} moderated so far`; }
      steps.push({ label: "Moderation", st, note });
    }
    steps.push({
      label: "Done",
      st: this.finished ? "done" : this.failed ? "failed" : "pending",
      note: this.finished ? "Script marked" : this.failed ? "Stopped with an error" : "",
    });
    return steps;
  },
  draw() {
    const words = { done: "done", active: "in progress", pending: "waiting", skipped: "skipped", failed: "stopped" };
    const steps = this.steps();
    $("#timeline").innerHTML = steps.map((s) => `
      <li class="tl-step is-${s.st}">
        <span class="tl-dot">${s.st === "done" ? CHECK : s.st === "skipped" ? DASH : s.st === "failed" ? CROSS : ""}</span>
        <span class="tl-label">${esc(s.label)}</span>
        <span class="tl-state"><span class="sr-only">${words[s.st]}: </span>${esc(s.note)}</span>
      </li>`).join("");
    const current = steps.find((s) => s.st === "active") || steps[steps.length - 1];
    const msg = `${current.label}: ${current.note || words[current.st]}`;
    if (msg !== this.lastAnnounced) { this.lastAnnounced = msg; announce(msg); }
  },
};

/* ---------- grading request ---------- */

async function readStream(res, onEvent) {
  let result = null;
  let error = null;
  const handleLine = (line) => {
    if (!line.trim()) return;
    const ev = JSON.parse(line);
    if (ev.type === "result") result = ev.result;
    else if (ev.type === "error") error = ev;
    else onEvent(ev);
  };
  if (res.body && res.body.getReader) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        handleLine(buf.slice(0, i));
        buf = buf.slice(i + 1);
      }
    }
    handleLine(buf);
  } else {
    (await res.text()).split("\n").forEach(handleLine);
  }
  if (error) throw new Error(friendlyError(error.status, error.detail));
  if (!result) throw new Error("The server closed the connection before marking finished. Please try again.");
  return result;
}

async function grade() {
  clearError();
  const paper = $("#question_paper_text").value.trim();
  if (!paper) {
    showError("Add a question paper first: upload it or paste its text into the Question paper box.");
    $("#question_paper_text").focus();
    return;
  }
  const n = Math.max(1, Math.min(10, parseInt($("#n_samples").value, 10) || 5));
  let thr = parseFloat($("#threshold").value);
  if (!Number.isFinite(thr)) thr = 0.85;
  thr = Math.max(0, Math.min(1, thr));

  const fd = new FormData();
  for (const key of INPUT_KEYS) {
    const v = $(`#${key}_text`).value;
    if (v.trim()) fd.append(`${key}_text`, v);
  }
  fd.append("n_samples", String(n));
  fd.append("threshold", String(thr));
  fd.append("stream", "true");

  const btn = $("#grade-btn");
  btn.disabled = true;
  btn.setAttribute("aria-busy", "true");
  $("#results").hidden = true; // never show an older script while a new one is being marked
  timeline.start(n);
  try {
    const res = await fetch("/api/grade", { method: "POST", body: fd });
    if (!res.ok) {
      let data = {};
      try { data = await res.json(); } catch (e) { data = {}; }
      throw new Error(friendlyError(res.status, data.detail));
    }
    const result = await readStream(res, (ev) => timeline.handle(ev));
    timeline.finish();
    render(result);
    const heading = $("#results-title");
    heading.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
    heading.focus({ preventScroll: true });
  } catch (e) {
    timeline.fail();
    showError(e.message || "We couldn't reach the server. Check your connection and try again.");
  } finally {
    btn.disabled = false;
    btn.removeAttribute("aria-busy");
  }
}

/* ---------- rendering the marked script ---------- */

// Hand-drawn marks (red pen).
const TICK = '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M5 17c2.5 1.6 4.4 4 6.2 7.2C15 15.5 20 9.4 27.5 4.5"/></svg>';
const HALF = '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M5 18c2.3 1.4 4 3.5 5.6 6.2C14 17 18.5 11.6 25 7"/><path d="M11.5 12.5c4.2.6 8.6 1 13.5 1.4"/></svg>';
const XMARK = '<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M7 6.5c5.6 5.8 11.2 12.3 18 19"/><path d="M25.5 6c-6.4 6-12.4 12.6-18 19.5"/></svg>';
const CHEV = '<svg class="ico chev" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>';

function markSymbol(awarded, max) {
  if (max > 0 && awarded >= max - 1e-9) return { svg: TICK, word: "full marks", short: "full" };
  if (awarded <= 1e-9) return { svg: XMARK, word: "no marks", short: "none" };
  return { svg: HALF, word: "part marks", short: "part" };
}

function reviewReasons(r) {
  const c = r.confidence;
  const g = r.grading;
  const reasons = [];
  const flags = c.integrity_flags || [];
  const texts = c.integrity_reasons || [];
  if (flags.length) {
    flags.forEach((f, i) => reasons.push(`<b>${esc(FLAG_LABELS[f] || f)}.</b> ${esc(texts[i] || "")}`));
    reasons.push(`Confidence is capped at 50% while an integrity flag is open.`);
  } else if (c.combined_confidence < c.threshold) {
    reasons.push(`<b>Low confidence.</b> ${pct(c.combined_confidence)} is below the ${pct(c.threshold)} needed.`);
  }
  const pre = c.pre_adjudication_total != null ? c.pre_adjudication_total : g.total_awarded;
  if (c.judge2_total != null && c.grader_role !== "judge2" && Math.abs(c.judge2_total - pre) >= 0.25 * g.max_marks) {
    reasons.push(`<b>The examiners disagreed.</b> The marker gave ${fmt(pre)} and the second examiner ${fmt(c.judge2_total)} out of ${fmt(g.max_marks)}.`);
  }
  return reasons;
}

function barRow(label, value, weight) {
  if (value == null) {
    return `<div class="bar-row is-na"><dt>${esc(label)}</dt><dd class="bar-track" aria-hidden="true"></dd><dd class="bar-val">n/a</dd></div>`;
  }
  const w = Math.max(0, Math.min(100, Math.round(value * 100)));
  const unused = weight === 0 ? " (not counted)" : "";
  return `<div class="bar-row"><dt>${esc(label)}${unused}</dt>
    <dd class="bar-track" aria-hidden="true"><div class="bar-fill" style="width:${w}%"></div></dd>
    <dd class="bar-val">${w}%</dd></div>`;
}

function dotLine(r) {
  const g = r.grading;
  const c = r.confidence;
  const max = g.max_marks || 1;
  const W = 300, x0 = 14, x1 = W - 14, yAxis = 46;
  const x = (v) => x0 + (Math.max(0, Math.min(max, v)) / max) * (x1 - x0);
  const ticks = [];
  const step = max <= 10 ? 1 : Math.ceil(max / 10);
  for (let t = 0; t <= max + 1e-9; t += step) {
    ticks.push(`<line class="tick" x1="${x(t)}" y1="${yAxis - 4}" x2="${x(t)}" y2="${yAxis + 4}"/>`);
  }
  const labels = `<text class="tick-label" x="${x0}" y="${yAxis + 18}" text-anchor="middle">0</text>
    <text class="tick-label" x="${x1}" y="${yAxis + 18}" text-anchor="middle">${fmt(max)}</text>`;
  const seen = {};
  const dots = r.sampled_totals.map((t) => {
    const k = fmt(t);
    seen[k] = (seen[k] || 0) + 1;
    return `<circle class="sample" cx="${x(t)}" cy="${yAxis - 9 - (seen[k] - 1) * 9}" r="4"/>`;
  }).join("");
  const second = c.judge2_total != null && c.grader_role !== "judge2"
    ? `<path class="second" d="M${x(c.judge2_total)} ${yAxis + 2} l6 6 -6 6 -6 -6z" transform="translate(0,-2)"/>` : "";
  const final = `<circle class="final" cx="${x(g.total_awarded)}" cy="${yAxis}" r="8"/>`;
  const height = Math.max(70, 70 + (Math.max(0, ...Object.values(seen)) - 3) * 9);
  const label = `Sampled totals ${r.sampled_totals.map(fmt).join(", ") || "none"} out of ${fmt(max)}` +
    (second ? `; second examiner ${fmt(c.judge2_total)}` : "") + `; final mark ${fmt(g.total_awarded)}.`;
  return `<svg viewBox="0 ${70 - height} ${W} ${height}" role="img" aria-label="${esc(label)}">
      <line class="axis" x1="${x0}" y1="${yAxis}" x2="${x1}" y2="${yAxis}"/>${ticks.join("")}${labels}${final}${dots}${second}
    </svg>
    <p class="dot-legend"><span>● marker's ${plural(r.sampled_totals.length, "marking", "markings")}: ${r.sampled_totals.map(fmt).join(", ") || "none"}</span>${
      second ? `<span>◆ second examiner: ${fmt(c.judge2_total)}</span>` : ""}<span>◌ final mark: ${fmt(g.total_awarded)}</span></p>`;
}

function whyPanel(r, id) {
  const c = r.confidence;
  const w = state.weights;
  const bars = [
    barRow("Agreement", c.agreement_confidence, w.agreement),
    barRow("Cross-model", c.cross_model_confidence, w.cross),
    barRow("Score spread", c.score_variance, w.variance),
  ];
  if (w.semantic) bars.push(barRow("Feedback consistency", c.semantic_consistency, w.semantic));
  const empty = c.n_samples === 0;
  return `<div class="why" id="${id}" hidden>
      <div><h4>How sure the markers were</h4>
        ${empty ? `<p>No marker was called: a blank answer gets 0 without marking.</p>` : `<dl class="bars">${bars.join("")}</dl>`}
      </div>
      ${empty ? "" : `<div class="dotline"><h4>Every mark given, on a 0–${fmt(r.grading.max_marks)} line</h4>${dotLine(r)}</div>`}
    </div>`;
}

function chipsFor(c) {
  const chips = [];
  if (c.answer_language) chips.push(`<span class="chip" title="Detected answer language">${esc(c.answer_language)}</span>`);
  if (c.grader_role === "fallback") chips.push(`<span class="chip chip--warn" title="${esc(c.graded_by || "")}">Backup grader used</span>`);
  if (c.grader_role === "judge2") chips.push(`<span class="chip chip--warn">Marked by the second examiner only</span>`);
  if (c.grader_role === "mock") chips.push(`<span class="chip">Mock marking</span>`);
  return chips.join("");
}

function pageFor(r, idx) {
  const g = r.grading;
  const c = r.confidence;
  const qid = esc(g.question_id);
  const anchor = `q-${idx}`;
  const flagged = !!c.flagged;
  const langCode = LANG_CODES[(c.answer_language || "").split(" + ")[0]] || "";

  const criteria = g.criterion_scores.map((s) => {
    const m = markSymbol(s.awarded, s.max);
    return `<li>
        <span class="mark-sym" role="img" aria-label="${m.word}">${m.svg}<span class="mark-word" aria-hidden="true">${m.short}</span></span>
        <span class="crit-text">${esc(s.criterion)}${s.justification ? `<span class="crit-why">${esc(s.justification)}</span>` : ""}</span>
        <span class="crit-mark" aria-label="${fmt(s.awarded)} of ${fmt(s.max)} marks">${fmt(s.awarded)}/${fmt(s.max)}</span>
      </li>`;
  }).join("");

  const answer = r.student_answer_text && r.student_answer_text.trim()
    ? `<div class="answer"${langCode ? ` lang="${langCode}"` : ""}>${esc(r.student_answer_text)}</div>`
    : `<div class="answer answer--empty">No answer written.</div>`;

  const moderator = c.adjudicated
    ? `<div class="moderator">
        <p class="moderator-title">Moderator decision: <span class="pen">${fmt(c.pre_adjudication_total)} → ${fmt(g.total_awarded)}</span></p>
        <p>Moderated by ${esc(c.referee_model || "the moderator")}${c.referee_reason ? `: ${esc(c.referee_reason)}` : "."}</p>
      </div>`
    : "";

  const reasons = flagged ? reviewReasons(r) : [];
  const sticky = flagged
    ? `<aside class="sticky" aria-label="Why this needs a second look"><h4>Second look needed</h4>
        <ul>${(reasons.length ? reasons : ["Flagged for review."]).map((x) => `<li>${x}</li>`).join("")}</ul></aside>`
    : "";

  const stamp = flagged
    ? `<div class="stamp stamp--review">Second examiner needed<span class="stamp-sub">confidence ${pct(c.combined_confidence)}</span></div>`
    : `<div class="stamp stamp--ok">Confident<span class="stamp-sub">confidence ${pct(c.combined_confidence)}</span></div>`;

  return `<article class="page" id="${anchor}" aria-labelledby="${anchor}-title">
      <div class="page-margin" aria-hidden="true"><span class="qnum">${qid}</span></div>
      <div class="page-body">
        <header class="page-head">
          <div>
            <h3 id="${anchor}-title">Question ${qid}<span class="sr-only">: ${fmt(g.total_awarded)} of ${fmt(g.max_marks)} marks, ${flagged ? "second examiner needed" : "confident"}</span></h3>
            <div class="chips">${chipsFor(c)}</div>
          </div>
          ${stamp}
        </header>
        ${r.question_text ? `<p class="question-text">${esc(r.question_text)}</p>` : ""}
        <span class="block-label">Student's answer</span>
        ${answer}
        <ol class="criteria" aria-label="Marks per rubric criterion">${criteria}</ol>
        ${g.feedback ? `<p class="comment">${esc(g.feedback)}</p>` : ""}
        ${moderator}
        ${sticky}
        <button class="btn-link why-toggle" type="button" aria-expanded="false" aria-controls="${anchor}-why">Why this stamp? ${CHEV}</button>
        ${whyPanel(r, `${anchor}-why`)}
      </div>
      <div class="page-score" aria-hidden="true">
        <span class="score-mark"><span class="got">${fmt(g.total_awarded)}</span><span class="rule"></span><span class="of">${fmt(g.max_marks)}</span></span>
      </div>
    </article>`;
}

function coverFor(data) {
  const n = data.results.length;
  const review = data.results.filter((r) => r.confidence.flagged);
  const tally = data.results.map((r, i) => {
    const f = r.confidence.flagged;
    return `<li><a href="#q-${i}" class="${f ? "is-review" : ""}">
        <span class="t-q">Q${esc(r.grading.question_id)}</span>
        <span class="t-m">${fmt(r.grading.total_awarded)}/${fmt(r.grading.max_marks)}</span>
        <span class="t-s">${f ? "second look" : "confident"}</span></a></li>`;
  }).join("");
  const notes = (data.warnings || []).length
    ? `<div class="notes"><h4>Notes from the invigilator</h4><ul>${data.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>`
    : "";
  return `
    <div class="total" role="img" aria-label="Total ${fmt(data.total_awarded)} out of ${fmt(data.total_max)}">
      <svg viewBox="0 0 176 120" aria-hidden="true"><path class="total-ring"
        d="M98 9C52 4 12 22 9 55c-3 34 34 56 80 55 45-1 81-20 79-52C166 26 128 8 84 12"/></svg>
      <div><div class="total-num"><span class="got">${fmt(data.total_awarded)}</span><span class="of">/${fmt(data.total_max)}</span></div>
      <div class="total-label">total marks</div></div>
    </div>
    <div class="cover-body">
      <h3>${review.length ? `${plural(review.length, "answer needs", "answers need")} a second look` : "Every answer marked with confidence"}</h3>
      <dl class="cover-facts">
        <div><dt>Questions</dt><dd>${n}</dd></div>
        <div><dt>Overall confidence</dt><dd>${pct(data.overall_confidence)}</dd></div>
        <div><dt>Second look</dt><dd>${review.length ? review.map((r) => `Q${esc(r.grading.question_id)}`).join(", ") : "none"}</dd></div>
      </dl>
      <ol class="tally" aria-label="Marks per question">${tally}</ol>
    </div>
    <p class="how-to-read">Ticks, part-ticks and crosses are per rubric criterion, with the marks in the margin. A stamp says how sure the markers were. Yellow notes explain why an answer needs a human examiner.</p>
    ${notes}`;
}

function render(data) {
  state.result = data;
  $("#cover").innerHTML = coverFor(data);
  $("#pages").innerHTML = data.results.map(pageFor).join("");
  $("#results").hidden = false;
  for (const btn of $$(".why-toggle")) {
    btn.addEventListener("click", () => {
      const panel = document.getElementById(btn.getAttribute("aria-controls"));
      const open = btn.getAttribute("aria-expanded") === "true";
      btn.setAttribute("aria-expanded", String(!open));
      panel.hidden = open;
    });
  }
}

/* ---------- downloads ---------- */

function downloadJson() {
  if (!state.result) return;
  const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "marked-script.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function printScript() {
  const title = document.title;
  document.title = "marked-script";
  window.addEventListener("afterprint", () => { document.title = title; }, { once: true });
  window.print();
}

/* ---------- wiring ---------- */

function init() {
  syncThemeButton();
  $("#theme-toggle").addEventListener("click", toggleTheme);
  if (window.matchMedia) {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    if (mq.addEventListener) mq.addEventListener("change", syncThemeButton);
  }
  INPUT_KEYS.forEach(setupSheet);
  // A file dropped outside a sheet must not navigate away from the page.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());
  $("#sample-load").addEventListener("click", loadSample);
  $("#grade-btn").addEventListener("click", grade);
  $("#print-btn").addEventListener("click", printScript);
  $("#json-btn").addEventListener("click", downloadJson);
  loadHealth();
  loadSampleList().then(() => {
    if (!INPUT_KEYS.some((k) => $(`#${k}_text`).value.trim())) loadSample();
  });
}

init();
