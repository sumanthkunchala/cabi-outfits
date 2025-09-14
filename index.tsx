// index.tsx  (ES module; plain JS so it runs even without a bundler)
const API_BASE = "http://127.0.0.1:8000";
const IMG_BASE = API_BASE + "/images";

const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const els = {
  brief:  $("#brief"),
  occ:    $("#occ"),
  bud:    $("#bud"),
  season: $("#season"),
  count:  $("#count"),
  go:     $("#go"),
  reset:  $("#reset"),
  recs:   $("#recsWrap"),
  summary:$("#summary"),
  histRad: $$('input[name="hist"]'),
  histWrap: $("#histWrap"),
  loadHist: $("#loadHist"),
  histBox:  $("#histBox"),
  pgClear:  $("#pg-clear"),
  pgExport: $("#pg-export"),
  pgCanvas: $("#pg-canvas"),
};

function sessionId() {
  const k = "cabi-session-id";
  if (!localStorage.getItem(k)) localStorage.setItem(k, cryptoRandom());
  return localStorage.getItem(k);
}
function cryptoRandom() {
  if (crypto?.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c=>{
    const r=Math.random()*16|0, v=c==='x'?r:(r&0x3|0x8); return v.toString(16);
  });
}
function escapeHtml(s) {
  return String(s||"").replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[m]));
}
function placeholderSvg(pid) {
  return `data:image/svg+xml,${encodeURIComponent(
    `<svg xmlns='http://www.w3.org/2000/svg' width='420' height='420'>
       <rect width='100%' height='100%' fill='#0f1420'/>
       <text x='50%' y='50%' fill='#9aa3b2' font-family='sans-serif' font-size='14' text-anchor='middle'>${pid}</text>
     </svg>`
  )}`;
}

function buildPrompt() {
  const parts = [];
  const brief = els.brief.value.trim();
  if (brief) parts.push(brief);

  const occ = els.occ.value.trim();
  if (occ) parts.push(`occasion: ${occ}`);

  const season = (els.season.value || "").trim();
  if (season && season.toLowerCase() !== "all") parts.push(`season: ${season}`);

  const acc = (document.querySelector('input[name="acc"]:checked')?.value || "yes");
  if (acc === "no") parts.push("no accessories");

  const bud = els.bud.value.trim();
  if (bud) parts.push(`budget max $${bud}`);

  return parts.join(" | ");
}

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ---- Rendering ----
function renderOutfits(outfits, opts) {
  els.recs.innerHTML = "";
  if (!outfits?.length) {
    els.summary.textContent = "No outfits found. Try tweaking your brief or increasing count.";
    return;
  }
  const frag = document.createDocumentFragment();

  outfits.forEach((o, idx) => {
    // Card
    const card = document.createElement("div");
    card.className = "rec-card glass"; // uses your existing "glass" look
    card.style.padding = "10px";

    // Header
    const head = document.createElement("div");
    head.className = "rec-head";
    head.innerHTML = `
      <div class="small">
        <b>#${idx + 1}</b> &nbsp; score ${o.score.toFixed(3)} • base ${escapeHtml(o.base)}
        ${o.palette?.length ? `• palette ${o.palette.map(escapeHtml).join(", ")}` : ""}
      </div>`;
    card.appendChild(head);

    // Items strip
    const strip = document.createElement("div");
    strip.className = "rec-strip";
    strip.style.display = "grid";
    strip.style.gridTemplateColumns = `repeat(${o.items.length}, minmax(160px, 1fr))`;
    strip.style.gap = "12px";
    strip.style.marginTop = "10px";

    o.items.forEach(it => {
      const cell = document.createElement("div");
      cell.className = "rec-item";
      cell.style.border = "1px solid #20243a";
      cell.style.borderRadius = "10px";
      cell.style.overflow = "hidden";
      cell.style.background = "#141724";

      const img = document.createElement("img");
      img.src = `${IMG_BASE}/${encodeURIComponent(it.product_id)}.jpg`;
      img.alt = it.name || it.product_id;
      img.loading = "lazy";
      img.style.width = "100%";
      img.style.height = "240px";
      img.style.objectFit = "cover";
      img.draggable = true;
      img.onerror = () => { img.src = placeholderSvg(it.product_id); };

      // drag → playground
      img.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("text/plain", JSON.stringify({
          pid: it.product_id,
          name: it.name || "",
          src: `${IMG_BASE}/${encodeURIComponent(it.product_id)}.jpg`
        }));
      });
      img.addEventListener("click", () => addToPlayground({
        pid: it.product_id,
        name: it.name || "",
        src: `${IMG_BASE}/${encodeURIComponent(it.product_id)}.jpg`
      }));

      const cap = document.createElement("div");
      cap.className = "small";
      cap.style.padding = "8px 10px";
      cap.innerHTML = `<b>${escapeHtml(it.name || "")}</b><br><span style="opacity:.7">${escapeHtml(it.category||"")}/${escapeHtml(it.subtype||"")}</span>`;

      cell.appendChild(img);
      cell.appendChild(cap);
      strip.appendChild(cell);
    });

    card.appendChild(strip);
    frag.appendChild(card);

    // Silent view feedback
    sendFeedback("view", o, opts.prompt).catch(()=>{});
  });

  els.recs.appendChild(frag);
  els.summary.textContent = `Generated ${outfits.length} outfit${outfits.length>1?"s":""} for: "${opts.prompt}"`;
}

// ---- Feedback ----
async function sendFeedback(action, outfit, prompt) {
  try {
    await fetchJSON(API_BASE + "/feedback", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        prompt,
        action,
        product_ids: outfit.items.map(it => String(it.product_id)),
        outfit_score: outfit.score,
        session_id: sessionId(),
        extra: { ui: "index.html" }
      })
    });
  } catch (e) {
    // non-fatal
    console.debug("feedback error:", e);
  }
}

// ---- Playground ----
function installPlayground() {
  const zone = els.pgCanvas;
  if (!zone) return;

  zone.addEventListener("dragover", (e) => e.preventDefault());
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    try {
      const data = JSON.parse(e.dataTransfer.getData("text/plain"));
      addToPlayground(data);
    } catch {}
  });

  els.pgClear?.addEventListener("click", () => { zone.innerHTML = ""; });

  els.pgExport?.addEventListener("click", async () => {
    try {
      const canvas = await html2canvas(zone, {backgroundColor:"#0e1118", useCORS:true});
      const a = document.createElement("a");
      a.download = "outfits.png";
      a.href = canvas.toDataURL("image/png");
      a.click();
    } catch (e) {
      alert("Export failed: " + e.message);
    }
  });
}

function addToPlayground({pid, name, src}) {
  const zone = els.pgCanvas;
  const wrap = document.createElement("div");
  wrap.className = "pg-item";
  wrap.style.position = "absolute";
  wrap.style.left = Math.round(20 + Math.random()*40) + "px";
  wrap.style.top  = Math.round(20 + Math.random()*40) + "px";
  wrap.style.transform = "translate(0,0) scale(1)";
  wrap.style.cursor = "move";
  wrap.title = (name || pid) + " — wheel to resize, double-click to remove";

  const img = document.createElement("img");
  img.src = src;
  img.alt = name || pid;
  img.style.maxWidth = "280px";
  img.style.userSelect = "none";
  img.draggable = false;
  img.onerror = () => { img.src = placeholderSvg(pid); };

  // drag within canvas
  let dragging = false, sx=0, sy=0, ox=0, oy=0, scale=1;
  wrap.addEventListener("mousedown", (e) => {
    dragging = true; sx = e.clientX; sy = e.clientY;
    const r = wrap.getBoundingClientRect(); ox = r.left; oy = r.top;
    e.preventDefault();
  });
  document.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - sx, dy = e.clientY - sy;
    wrap.style.left = (ox + dx - zone.getBoundingClientRect().left) + "px";
    wrap.style.top  = (oy + dy - zone.getBoundingClientRect().top) + "px";
  });
  document.addEventListener("mouseup", () => dragging=false);

  // wheel to resize
  wrap.addEventListener("wheel", (e) => {
    e.preventDefault();
    scale = Math.max(0.3, Math.min(2.5, scale + (e.deltaY < 0 ? 0.1 : -0.1)));
    wrap.style.transform = `translate(0,0) scale(${scale.toFixed(2)})`;
  }, {passive:false});

  // remove
  wrap.addEventListener("dblclick", () => wrap.remove());

  wrap.appendChild(img);
  zone.appendChild(wrap);
}

// ---- History toggle (UI stays the same; stubbed loader) ----
function installHistory() {
  if (!els.histRad?.length) return;
  const sync = () => {
    const val = document.querySelector('input[name="hist"]:checked')?.value || "no";
    els.histWrap.style.display = (val === "yes") ? "block" : "none";
  };
  els.histRad.forEach(r => r.addEventListener("change", sync));
  sync();

  els.loadHist?.addEventListener("click", () => {
    const id = $("#cust")?.value?.trim();
    els.histBox.textContent = id ? `Loaded mock history for ${id} (stub)` : "Enter a customer id or email first.";
  });
}

// ---- Generate Looks ----
async function onGenerate() {
  try {
    els.go.disabled = true;
    els.summary.textContent = "Generating...";
    els.recs.innerHTML = "";

    const prompt = buildPrompt();
    const k = Math.max(1, Math.min(10, parseInt(els.count.value || "4", 10)));

    const data = await fetchJSON(API_BASE + "/recommend/outfit", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ prompt, k })
    });

    renderOutfits(data.outfits || [], { prompt });
  } catch (e) {
    console.error(e);
    els.summary.textContent = "Error: " + e.message;
  } finally {
    els.go.disabled = false;
  }
}

function onReset() {
  els.brief.value = "";
  els.occ.value = "";
  els.bud.value = "";
  els.season.value = "All";
  els.count.value = "4";
  els.recs.innerHTML = "";
  els.summary.textContent = "Your generated outfits will appear here.";
}

// ---- Wire up ----
window.addEventListener("DOMContentLoaded", () => {
  installPlayground();
  installHistory();
  els.go?.addEventListener("click", onGenerate);
  els.reset?.addEventListener("click", onReset);

  // Auto-generate once so the page isn't empty
  onGenerate().catch(()=>{});
});
