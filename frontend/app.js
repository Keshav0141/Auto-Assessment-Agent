const $ = (id) => document.getElementById(id);
const inputIds = ["question_paper", "answer_sheet", "rubric", "model_answers"];
const textIds = ["question_paper_text", "answer_sheet_text", "rubric_text", "model_answers_text"];

let _timer = null;
let _t0 = 0;
let _nSamples = "5";
let _threshold = "0.6";

async function init() {
  try {
    const res = await fetch("/api/health");
    const h = await res.json();
    const pill = $("status-pill");
    pill.classList.remove("hidden");
    if (h.mock) {
      pill.className = "pill warn";
      pill.innerHTML = "<i class='ph ph-warning'></i> MOCK mode — no LLM API key set. Set OPENAI_API_KEY in .env for real grading.";
    } else {
      pill.className = "pill ok";
      pill.innerHTML = `<i class='ph ph-check-circle'></i> LLM ready — ${h.model} · ${h.samples_per_question} samples/question · threshold ${h.confidence_threshold}`;
    }
  } catch {
    const pill = $("status-pill");
    pill.classList.remove("hidden");
    pill.className = "pill warn";
    pill.innerHTML = "<i class='ph ph-warning'></i> Backend not reachable.";
  }
}

function confColor(c) {
  if (c >= 0.8) return "#00ff88"; // neon green
  if (c >= 0.6) return "#ffaa00"; // neon amber
  return "#ff3366"; // neon red
}

function esc(s) {
  const d = document.createElement("div");
  d.textContent = s == null ? "" : String(s);
  return d.innerHTML;
}

function gauge(pct, size, color) {
  const r = size / 2 - 8;
  const c = 2 * Math.PI * r;
  const dash = Math.max((pct / 100) * c, 0);
  const v = Math.round(pct);
  return `
    <svg class="gauge" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" role="img" aria-label="${v}% confidence">
      <defs>
        <filter id="glow-${pct}" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="4" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>
      <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="8"/>
      <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="${color}" stroke-width="8"
        stroke-linecap="round" stroke-dasharray="${dash.toFixed(1)} ${c.toFixed(1)}"
        transform="rotate(-90 ${size / 2} ${size / 2})" filter="url(#glow-${pct})"/>
      <text x="50%" y="52%" text-anchor="middle" dominant-baseline="middle" fill="${color}"
        font-family="Outfit,sans-serif" font-weight="800" font-size="${Math.round(size * 0.26)}" style="text-shadow: 0 0 10px ${color}">${v}</text>
    </svg>`;
}

function qCard(r, idx) {
  const g = r.grading;
  const c = r.confidence;
  const pct = Math.round(c.combined_confidence * 100);
  const color = confColor(c.combined_confidence);
  
  let qClass = "q";
  if (c.combined_confidence >= 0.8) {
      qClass += " ok";
  } else if (c.combined_confidence >= 0.6) {
      qClass += " warn";
  }
  
  let badge = '<span class="badge ok"><i class="ph ph-check"></i> OK</span>';
  let ariaExpanded = "false";
  
  if (c.flagged) {
    if (pct >= 40) {
      qClass += " medium-flagged flagged";
      badge = '<span class="badge flag medium"><i class="ph ph-eye"></i> REVIEW</span>';
    } else {
      qClass += " flagged open";
      ariaExpanded = "true";
      badge = '<span class="badge flag"><i class="ph ph-warning-circle"></i> REVIEW</span>';
    }
  }

  const rows = g.criterion_scores
    .map((s) => `<tr><td>${esc(s.criterion)}</td><td><strong>${s.awarded}/${s.max}</strong></td><td>${esc(s.justification)}</td></tr>`)
    .join("");

  const qText = r.question_text
    ? `<div class="q-block q-question"><div class="q-block-label"><i class="ph ph-file-text"></i> Question</div>${esc(r.question_text)}</div>`
    : "";
  const ansText = r.student_answer_text
    ? `<div class="q-block q-answer"><div class="q-block-label"><i class="ph ph-user"></i> Student answer</div>${esc(r.student_answer_text)}</div>`
    : "";

  return `
    <div class="${qClass}" style="transition-delay: ${idx * 100}ms">
      <button class="q-head" aria-expanded="${ariaExpanded}">
        <div class="q-head-info">
          <div class="q-title">Question ${esc(g.question_id)} <i class="ph ph-caret-down"></i></div>
          <div class="q-marks">${g.total_awarded} / ${g.max_marks} marks</div>
        </div>
        <div class="conf">
          <div style="text-align: right">
            ${badge}
          </div>
          ${gauge(pct, 64, color)}
        </div>
      </button>
      <div class="q-body">
        ${qText}
        ${ansText}
        <div class="q-feedback"><strong><i class="ph ph-chat-teardrop-text"></i> Feedback:</strong><br/> ${esc(g.feedback)}</div>
        <table>
          <thead><tr><th>Criterion</th><th>Awarded</th><th>Justification</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
        <div class="samples" style="color: #a0aec0; font-size: 0.9rem;">
          <b><i class="ph ph-chart-bar"></i> Confidence breakdown:</b> score-variance ${(c.score_variance * 100).toFixed(0)}% ·
          semantic-consistency ${(c.semantic_consistency * 100).toFixed(0)}%${c.eigv_confidence != null ? ` · eigv ${(c.eigv_confidence * 100).toFixed(0)}%` : ""} ·
          sampled totals: ${r.sampled_totals.map((t) => t.toFixed(1)).join(", ")}
        </div>
      </div>
    </div>`;
}

function animateValue(obj, start, end, duration) {
  let startTimestamp = null;
  const step = (timestamp) => {
    if (!startTimestamp) startTimestamp = timestamp;
    const progress = Math.min((timestamp - startTimestamp) / duration, 1);
    const easeProgress = progress * (2 - progress);
    obj.innerHTML = Math.floor(easeProgress * (end - start) + start);
    if (progress < 1) {
      window.requestAnimationFrame(step);
    } else {
      obj.innerHTML = end;
    }
  };
  window.requestAnimationFrame(step);
}

function fireConfetti() {
  const container = $("confetti-container");
  container.innerHTML = "";
  const colors = ['#00f0ff', '#ff006e', '#8a2be2', '#00ff88'];
  
  for (let i = 0; i < 60; i++) {
    const conf = document.createElement("div");
    conf.className = "confetti";
    conf.style.left = Math.random() * 100 + "vw";
    conf.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
    
    const duration = Math.random() * 2 + 2;
    const delay = Math.random() * 0.5;
    
    conf.animate([
      { transform: `translate3d(0, -10vh, 0) rotate(0)`, opacity: 1 },
      { transform: `translate3d(${Math.random()*200 - 100}px, 100vh, 0) rotate(${Math.random()*720}deg)`, opacity: 0 }
    ], {
      duration: duration * 1000,
      delay: delay * 1000,
      easing: 'cubic-bezier(.37,0,.63,1)',
      fill: 'forwards'
    });
    
    container.appendChild(conf);
  }
}

function render(data) {
  const resultsCard = $("results-card");
  resultsCard.classList.remove("hidden");
  resultsCard.classList.remove("fade-in");
  void resultsCard.offsetWidth;
  resultsCard.classList.add("fade-in");

  const flaggedCount = data.flagged.length;
  const flagBadge = flaggedCount
    ? '<span class="badge flag"><i class="ph ph-warning-circle"></i> HUMAN REVIEW NEEDED</span>'
    : '<span class="badge ok"><i class="ph ph-check-circle"></i> ALL HIGH CONFIDENCE</span>';

  const overallPct = Math.round(data.overall_confidence * 100);
  const overallColor = confColor(data.overall_confidence);

  const chart = data.results
    .map((r, i) => {
      const g = r.grading;
      const flaggedQ = data.flagged.includes(g.question_id);
      const barColor = flaggedQ
        ? "linear-gradient(90deg,#ff3366,#ffaa00)"
        : "linear-gradient(90deg,#00f0ff,#ff006e)";
      return `
        <div class="chart-row">
          <div class="chart-label">Q${esc(g.question_id)}</div>
          <div class="chart-track"><div class="chart-fill" id="chart-fill-${i}" style="background:${barColor}"></div></div>
          <div class="chart-val" style="width: 40px; text-align: right; font-weight: bold;">${g.total_awarded}/${g.max_marks}</div>
        </div>`;
    })
    .join("");

  $("summary").innerHTML = `
    <div class="sum-block" data-type="marks">
      <div class="big"><span id="total-awarded">0</span> / ${data.total_max}</div>
      <div class="lbl">Total marks</div>
    </div>
    <div class="sum-block" data-type="conf" style="transition-delay: 100ms">
      <div class="gauge-wrap">${gauge(overallPct, 120, overallColor)}</div>
      <div class="lbl">Overall confidence</div>
    </div>
    <div class="sum-block" data-type="flag" style="transition-delay: 200ms">
      <div class="big">${flaggedCount}</div>
      <div class="lbl">Flagged questions</div>
      <div class="badge-wrap" style="margin-top: 1rem;">${flagBadge}</div>
    </div>
    <div class="sum-block" data-type="chart" style="transition-delay: 300ms">
      <div class="lbl">Per-question scores</div>
      <div class="score-chart">${chart}</div>
    </div>`;

  $("legend").innerHTML = `
    <div style="background: rgba(0,240,255,0.05); border-left: 4px solid #00f0ff; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;"><strong>How to read this:</strong> each question is graded ${esc(_nSamples)} times and the confidence score reflects how
    consistently those samplings agree (score variance + semantic similarity). Gradings below the ${esc(_threshold)} threshold
    are flagged in red — click any question to review the question, student answer, criterion breakdown and feedback.</div>`;

  $("warnings").innerHTML = data.warnings.map((w) => `<p style="color: #ffaa00; margin-bottom: 1rem;">⚠ ${esc(w)}</p>`).join("");

  $("questions").innerHTML = data.results.map((r, i) => qCard(r, i)).join("");

  // Staggered animations
  setTimeout(() => {
    // Animate summary blocks
    document.querySelectorAll('.sum-block').forEach(el => el.classList.add('animate-in'));
    
    // Animate total marks counter
    animateValue($("total-awarded"), 0, data.total_awarded, 1500);
    
    // Animate chart bars
    data.results.forEach((r, i) => {
      setTimeout(() => {
        const fill = $("chart-fill-" + i);
        if (fill) {
          const g = r.grading;
          const frac = g.max_marks ? Math.round((g.total_awarded / g.max_marks) * 100) : 0;
          fill.style.width = frac + "%";
        }
      }, 500 + (i * 100)); // stagger chart bars
    });

    // Animate question cards
    document.querySelectorAll('.q').forEach(el => el.classList.add('animate-in'));
    
    // Subtle confetti if good overall score and confidence
    if (overallPct > 80 && (data.total_awarded / data.total_max) > 0.5) {
      fireConfetti();
    }
  }, 100);

  document.querySelectorAll(".q-head").forEach((h, i) => {
    h.addEventListener("click", () => {
      const q = document.querySelectorAll(".q")[i];
      const isExpanded = h.getAttribute("aria-expanded") === "true";
      h.setAttribute("aria-expanded", !isExpanded);
      q.classList.toggle("open");
    });
  });

  $("download-btn").onclick = () => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "grading_report.json";
    a.click();
  };
}

function startTimer() {
  _t0 = Date.now();
  tick();
  _timer = setInterval(tick, 1000);
}

function tick() {
  const s = Math.floor((Date.now() - _t0) / 1000);
  const m = Math.floor(s / 60);
  $("elapsed").textContent = `${m}:${String(s % 60).padStart(2, "0")}`;
}

function stopTimer() {
  if (_timer) {
    clearInterval(_timer);
    _timer = null;
  }
}

async function grade() {
  const fd = new FormData();
  inputIds.forEach((id) => {
    const f = $(id).files[0];
    if (f) fd.append(id, f);
  });
  textIds.forEach((id) => {
    const v = $(id).value.trim();
    if (v) fd.append(id, v);
  });
  _nSamples = $("n_samples").value || "5";
  _threshold = $("threshold").value || "0.6";
  fd.append("n_samples", _nSamples);
  fd.append("threshold", _threshold);

  $("loading-msg").textContent = `Grading ${_nSamples} samples per question…`;
  $("loading").classList.remove("hidden");
  $("grade-btn").disabled = true;
  $("results-card").classList.add("hidden");
  startTimer();

  const errB = $("error-banner");
  if (errB) errB.remove();

  try {
    const res = await fetch("/api/grade", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) {
      showError("Error: " + (data.detail || JSON.stringify(data)));
      return;
    }
    render(data);
    setTimeout(() => {
      $("results-card").scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);
  } catch (e) {
    showError("Request failed: " + e);
  } finally {
    stopTimer();
    $("loading").classList.add("hidden");
    $("grade-btn").disabled = false;
  }
}

function showError(msg) {
  const old = $("error-banner");
  if (old) old.remove();
  const banner = document.createElement("div");
  banner.id = "error-banner";
  banner.className = "error-banner";
  banner.innerHTML = `<span><i class="ph ph-warning"></i> ${esc(msg)}</span><button class="close" aria-label="Dismiss">&times;</button>`;
  banner.querySelector(".close").addEventListener("click", () => banner.remove());
  $("input-card").prepend(banner);
  setTimeout(() => banner.remove(), 15000);
}

$("grade-btn").addEventListener("click", grade);

const SAMPLE = {
  question_paper_text:
`Question 1 (5 marks)
State Newton's second law of motion and give its mathematical form. A 2 kg object experiences a net force of 10 N. What is its acceleration?

Question 2 (3 marks)
Define photosynthesis in your own words.

Question 3 (4 marks)
Solve the quadratic equation x^2 - 5x + 6 = 0 and verify your solutions.

Question 4 (6 marks)
A ball is thrown vertically upward with an initial velocity of 30 m/s. Find (a) the maximum height reached and (b) the time taken to reach it. (g = 10 m/s^2)`,
  answer_sheet_text:
`Question 1
Newton's second law says that force equals mass times acceleration, so F = ma. With m = 2 kg and F = 10 N, acceleration a = F/m = 10/2 = 5 m/s^2.

Question 2
Photosynthesis is when plants make their own food using sunlight and carbon dioxide, producing oxygen.

Question 3
x^2 - 5x + 6 = 0. This factors as (x - 2)(x - 3) = 0, so the solutions are x = 2 and x = 3.

Question 4
The ball goes up until its velocity reaches zero. Using h = u^2/2g gives h = 30^2/20 = 45 m. The time to reach the top is t = u/g = 30/10 = 3 s.`,
  rubric_text:
`Question 1
2 marks for stating the law correctly, 1 mark for the correct equation F=ma, 1 mark for correct substitution, 1 mark for correct final value and unit (5 m/s^2)

Question 2
1 mark for identifying it is a plant process, 1 mark for converting light to chemical energy, 1 mark for mentioning inputs/outputs (CO2+water -> glucose+oxygen)

Question 3
2 marks for correct factorization, 1 mark for correct solutions, 1 mark for verifying by substitution

Question 4
2 marks for correct equation choice, 2 marks for correct max height (45 m), 2 marks for correct time (3 s)`,
  model_answers_text:
`Question 1
Second law: the net force acting on an object equals the rate of change of its momentum. For constant mass, F = ma. a = F/m = 10/2 = 5 m/s^2, in the direction of the applied force.

Question 2
Plants use light energy to convert carbon dioxide and water into glucose and oxygen; this converts light energy into chemical energy stored in glucose.

Question 3
x^2 - 5x + 6 = (x - 2)(x - 3) = 0, so x = 2 or x = 3. Verify: 4 - 10 + 6 = 0 and 9 - 15 + 6 = 0.

Question 4
At maximum height the final velocity v = 0. (a) v^2 = u^2 - 2gh, so h = u^2 / (2g) = 900 / 20 = 45 m. (b) v = u - gt, so t = u/g = 30/10 = 3 s.`
};

$("sample-btn").addEventListener("click", () => {
  Object.entries(SAMPLE).forEach(([id, val]) => { $(id).value = val; });
  inputIds.forEach((id) => { $(id).value = ""; }); // clear any picked files
  $("results-card").classList.add("hidden");
});

// Auto-load sample data on first visit so "Grade" works immediately.
if (!textIds.some((id) => $(id).value.trim())) {
  Object.entries(SAMPLE).forEach(([id, val]) => { $(id).value = val; });
}

init();
