import React, { useRef, useEffect, useState, useCallback } from "react";

/* ------------------------------------------------------------------ *
 * Photon accumulation: mono vs OSC on a 2x2 pixel block.
 *
 * Broadband and narrowband share one model:
 *   - a photon belongs to a band (R/G/B, or Ha/OIII/SII)
 *   - a filter in front of the sensor either passes that band or not
 *   - the pixel it lands on responds with some probability
 * Mono's filter is a slab over the whole sensor. The OSC has BOTH a
 * slab (its duoband, in narrowband) and a dye on every pixel.
 * ------------------------------------------------------------------ */

const C = {
  bg: "#070B14", panel: "#0D1422", edge: "#1B2740", edgeHi: "#2C3D5E",
  gold: "#D4A94A", text: "#C9D3E5", bright: "#E8EEF9", mid: "#A9B8D2",
  dim: "#66768F",
};

/* the colour filter array never changes: RGGB, whatever light arrives */
const DYE = [0, 1, 1, 2];
const DYE_COL = ["#FF4D5E", "#3FD98A", "#4D9BFF"];
const DYE_NAME = ["R", "G", "B"];

const BB_BANDS = [
  { name: "R", color: "#FF4D5E" },
  { name: "G", color: "#3FD98A" },
  { name: "B", color: "#4D9BFF" },
];
const BB_RESP = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; // resp[dye][band]

const NB_BANDS = [
  { name: "Hα", color: "#E8B33C" },   // gold
  { name: "OIII", color: "#2FD3C8" }, // teal
  { name: "SII", color: "#E048A8" },  // magenta
];
/* Ha 656 -> red only. OIII 500.7 straddles the blue/green crossover:
   all of green, most of blue. SII 672 -> red only, same as Ha. */
const NB_RESP = [[1, 0, 1], [0, 1, 0], [0, 0.6, 0]];

/* line photons are drawn in their own palette so they read apart from
   the continuum; the pixel they land on still follows true wavelength,
   so Ha and SII go to the red pixels however they are coloured here */
const bandCol = (M, i) => M.bands[i].color;

/* continuum and sky: ordinary broadband light, drawn in RGB */
const CONT_COL = ["#FF4D5E", "#3FD98A", "#4D9BFF"];
const CONT_FRAC = 0.7; // emission lines are a minority of what arrives

const BLOCK = 100;
const SLATE_H = 44;
const NOMINAL_BUDGET = 1100;
const SCALES = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000];

const SPEEDS = [
  { label: "Slow", rate: 0.6, flight: 5.2 },
  { label: "Normal", rate: 2.4, flight: 3.4 },
  { label: "Fast", rate: 12, flight: 1.5 },
  { label: "Sprint", rate: 45, flight: 0.7 },
];

/* ---------------- programmes ---------------- */

function buildLrgb(w) {
  /* all the luminance first, then R, G, B — one pass, no cycling */
  return [...Array.from({ length: Math.max(0, w) }, () => null), 0, 1, 2];
}
const rep = (v, n) => Array.from({ length: n }, () => v);

const bb = (label, w) => ({
  label, group: "bb", bands: BB_BANDS, resp: BB_RESP, w,
  monoSeq: w === null ? [null] : buildLrgb(w), // [null] == L for the whole run
  oscSeq: null,
  budget: w === null ? NOMINAL_BUDGET : buildLrgb(w).length * BLOCK,
});

const MODES = {
  lum: bb("L only", null),
  s8: bb("8:1:1:1", 8),
  s4: bb("4:1:1:1", 4),
  s1: bb("1:1:1:1", 1),
  s0: bb("1:1:1 no L", 0),
  /* one pass through the lines, in the order the palette is named.
     Mono changes filter at every line; the OSC changes only when it
     has to swap duoband, so the two schedules no longer line up. */
  hoo: {
    label: "HOO", group: "nb", bands: NB_BANDS.slice(0, 2), resp: NB_RESP,
    monoSeq: [0, 1],            // Ha, OIII
    oscSeq: [[0, 1]],           // one Ha+OIII duoband all night
    budget: 800, cont: CONT_FRAC,
  },
  sho: {
    label: "SHO", group: "nb", bands: NB_BANDS, resp: NB_RESP,
    monoSeq: [2, 0, 1],         // SII, Ha, OIII
    oscSeq: [[0, 1], [1, 2]],   // Ha+OIII, then SII+OIII
    budget: 1200, cont: CONT_FRAC,
  },
};
const ORDER_BB = ["lum", "s8", "s4", "s1", "s0"];
const ORDER_NB = ["hoo", "sho"];

const budgetOf = (m) => MODES[m].budget || NOMINAL_BUDGET;
/* a sequence divides the budget into equal blocks of its own */
const idxAt = (seq, pos, budget) =>
  seq && seq.length
    ? Math.min(seq.length - 1, Math.floor(pos / (budget / seq.length)))
    : 0;
const filterAt = (seq, pos, budget) =>
  seq && seq.length ? seq[idxAt(seq, pos, budget)] : null; // null == passes all
const monoFilterAt = (pos, m) => filterAt(MODES[m].monoSeq, pos, budgetOf(m));
const oscFilterAt = (pos, m) => filterAt(MODES[m].oscSeq, pos, budgetOf(m));

/* array-average response to a band, i.e. what fraction of the CFA sees it */
function arrayFrac(m, band) {
  const R = MODES[m].resp;
  return DYE.reduce((a, d) => a + R[d][band], 0) / 4;
}
/* analytic equilibrium fractions, used for the "converges to" captions */
function expected(m) {
  const M = MODES[m];
  const nb = M.bands.length;
  const budget = budgetOf(m);
  const S = 1200; // fine sampling across the run
  let mono = 0, osc = 0;
  const monoBand = new Array(nb).fill(0), oscBand = new Array(nb).fill(0);
  for (let t = 0; t < S; t++) {
    const pos = ((t + 0.5) * budget) / S;
    const mf = filterAt(M.monoSeq, pos, budget);
    const of = filterAt(M.oscSeq, pos, budget);
    for (let k = 0; k < nb; k++) {
      const mOk = mf === null || mf === k ? 1 : 0;
      const oOk = of === null || of.includes(k) ? arrayFrac(m, k) : 0;
      mono += mOk / (nb * S);
      osc += oOk / (nb * S);
      monoBand[k] += mOk / S;
      oscBand[k] += oOk / S;
    }
  }
  const keep = 1 - (M.cont || 0);
  return {
    mono: mono * keep, osc: osc * keep,
    ratio: osc > 0 ? mono / osc : 0, monoBand, oscBand,
  };
}

/* ---------------- slate ---------------- */

function slateGrid(budget, w, h) {
  const cols = Math.max(1, Math.ceil(Math.sqrt((budget * w) / h)));
  return { cols, rows: Math.max(1, Math.ceil(budget / cols)) };
}
const scaleFor = (n, cap) => SCALES.find((k) => n / k <= cap) || SCALES[SCALES.length - 1];
function pushQ(q, c) { q.push(c); if (q.length > 300) q.shift(); }
function syncSlate(slate, q, target, cap) {
  const t = Math.min(cap, Math.max(0, target));
  let steps = 8;
  while (slate.length < t && steps-- > 0)
    slate.push(q.length ? q[(Math.random() * q.length) | 0] : "#66768F");
  while (slate.length > t && steps-- > 0) slate.pop();
}

function freshSim(nb) {
  return {
    photons: [],
    mono: [0, 0, 0, 0], osc: [0, 0, 0, 0],
    monoAll: 0,                       // photons taken with no band filter (L)
    monoBand: new Array(nb).fill(0),
    oscBand: new Array(nb).fill(0),
    monoSlate: [], oscSlate: [], monoQ: [], oscQ: [],
    monoFlash: [0, 0, 0, 0], oscFlash: [0, 0, 0, 0],
    monoTot: 0, oscTot: 0, delivered: 0,
    monoLost: 0, oscLost: 0, oscSlabLost: 0, contDelivered: 0,
    pos: 0, acc: 0, done: false, lastF: "?", pulse: 0,
  };
}

/* ---------------- layout ---------------- */

function makeLayout(W, H, budget) {
  const cell = Math.round(Math.max(44, Math.min(80, W * 0.084)));
  const gap = Math.round(cell * 0.13);
  const gw = cell * 2 + gap, gh = cell * 2 + gap;
  const panelW = Math.round(gw + cell * 2.2);
  const panelGap = Math.round(cell * 0.55);

  const labelY = 22, badgeY = 42, panelTop = 58;
  const apexY = panelTop + 34;
  const gridTop = apexY + Math.round(cell * 1.75);
  const filterY = apexY + (gridTop - apexY) * 0.56;
  const slateY = gridTop + gh + 30;
  const slatePad = 14;
  const panelH = slateY + SLATE_H + 26 - panelTop;

  const monoPX = Math.round(W / 2 - panelGap / 2 - panelW);
  const oscPX = Math.round(W / 2 + panelGap / 2);

  const gridX = (px) => px + (panelW - gw) / 2;
  const center = (px, i) => ({
    x: gridX(px) + (i % 2) * (cell + gap) + cell / 2,
    y: gridTop + ((i / 2) | 0) * (cell + gap) + cell / 2,
  });
  const apex = (px) => ({ x: px + panelW / 2, y: apexY });

  const slateW = panelW - slatePad * 2;
  const sg = slateGrid(budget, slateW, SLATE_H);
  const slateSlot = (px, slot) => ({
    x: px + slatePad + (((slot % sg.cols) + 0.5) * slateW) / sg.cols,
    y: slateY + SLATE_H - ((Math.floor(slot / sg.cols) + 0.5) * SLATE_H) / sg.rows,
  });

  return {
    cell, gap, gw, gh, panelW, panelH, panelTop, labelY, badgeY,
    apexY, gridTop, filterY, slateY, monoPX, oscPX, gridX, center, apex,
    slatePad, slateW, slateCap: sg.cols * sg.rows, slateSlot,
    dotR: Math.max(0.85, Math.min(slateW / sg.cols, SLATE_H / sg.rows) * 0.34),
  };
}

/* ================================================================== */

export default function PhotonAccumulation() {
  const canvasRef = useRef(null);
  const modeRef = useRef("lum");
  const spdRef = useRef(SPEEDS[1]);
  const playRef = useRef(true);
  const sim = useRef(freshSim(3));

  const [mode, setMode] = useState("lum");
  const [speed, setSpeed] = useState(1);
  const [playing, setPlaying] = useState(true);
  const hubRef = useRef(false);
  const [ui, setUi] = useState({
    delivered: 0, monoTot: 0, oscTot: 0, monoAll: 0,
    monoBand: [0, 0, 0], oscBand: [0, 0, 0],
    monoLost: 0, oscLost: 0, oscSlabLost: 0, contDelivered: 0,
    pos: 0, done: false, filter: null, oscFilter: null,
  });

  useEffect(() => void (modeRef.current = mode), [mode]);
  useEffect(() => void (spdRef.current = SPEEDS[speed]), [speed]);
  useEffect(() => void (playRef.current = playing), [playing]);
  useEffect(() => {
    const m = typeof window !== "undefined" && window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)");
    if (m && m.matches) { setPlaying(false); playRef.current = false; }
  }, []);

  const reset = useCallback((m) => {
    const nb = MODES[m || modeRef.current].bands.length;
    sim.current = freshSim(nb);
    setUi({
      delivered: 0, monoTot: 0, oscTot: 0, monoAll: 0,
      monoBand: new Array(nb).fill(0), oscBand: new Array(nb).fill(0),
      monoLost: 0, oscLost: 0, oscSlabLost: 0, contDelivered: 0,
      pos: 0, done: false, filter: null, oscFilter: null,
    });
  }, []);

  const burst = useCallback((n) => {
    const s = sim.current, m = modeRef.current, M = MODES[m];
    const nb = M.bands.length;
    let left = M.monoSeq ? Math.min(n, budgetOf(m) - s.pos) : n;
    for (let i = 0; i < left; i++) {
      const px = (Math.random() * 4) | 0;
      const mf = monoFilterAt(s.pos, m), of = oscFilterAt(s.pos, m);
      s.delivered++; s.pos++;
      if (Math.random() < (M.cont || 0)) {
        /* continuum and sky: outside every narrowband passband, so it
           stops at the slab on both cameras */
        const c = CONT_COL[(Math.random() * 3) | 0];
        s.contDelivered++;
        pushQ(s.monoQ, c); s.monoLost++;
        pushQ(s.oscQ, c); s.oscLost++; s.oscSlabLost++;
        continue;
      }
      const band = (Math.random() * nb) | 0;
      const bc = bandCol(M, band);
      if (mf === null) { s.mono[px]++; s.monoAll++; s.monoTot++; }
      else if (mf === band) { s.mono[px]++; s.monoBand[band]++; s.monoTot++; }
      else { pushQ(s.monoQ, bc); s.monoLost++; }

      const passes = of === null || of.includes(band);
      if (passes && Math.random() < M.resp[DYE[px]][band]) {
        s.osc[px]++; s.oscBand[band]++; s.oscTot++;
      } else {
        pushQ(s.oscQ, bc); s.oscLost++;
        if (!passes) s.oscSlabLost++;
      }
    }
    if (M.monoSeq && s.pos >= budgetOf(m)) s.done = true;
  }, []);

  const H = 520;

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    let raf = 0, last = performance.now(), uiAcc = 0;

    const loop = (now) => {
      raf = requestAnimationFrame(loop);
      let dt = (now - last) / 1000; last = now;
      if (dt > 0.1) dt = 0.1;

      const W = cv.clientWidth || 900;
      const s = sim.current, m = modeRef.current, M = MODES[m];
      const nb = M.bands.length;
      const budget = budgetOf(m);
      const L = makeLayout(W, H, budget);
      const spd = spdRef.current;

      /* ---- spawn ---- */
      if (playRef.current && !s.done) {
        s.acc += dt * spd.rate;
        while (s.acc >= 1) {
          s.acc -= 1;
          if (M.monoSeq && s.pos >= budget) { s.done = true; break; }
          const cont = Math.random() < (M.cont || 0);
          const band = cont ? 0 : (Math.random() * nb) | 0;
          const px = (Math.random() * 4) | 0;
          const mf = monoFilterAt(s.pos, m), of = oscFilterAt(s.pos, m);
          const jx = (Math.random() - 0.5) * L.cell * 0.34;
          const jy = (Math.random() - 0.5) * L.cell * 0.34;
          const col = cont
            ? CONT_COL[(Math.random() * 3) | 0]
            : bandCol(M, band);
          const oPasses = !cont && (of === null || of.includes(band));
          const oPixel = oPasses && Math.random() < M.resp[DYE[px]][band];
          if (cont) s.contDelivered++;
          for (const side of [0, 1]) {
            const base = side === 0 ? L.monoPX : L.oscPX;
            const a = L.apex(base), c = L.center(base, px);
            const tx = c.x + jx, ty = c.y + jy;
            const ok = cont ? false
              : side === 0 ? mf === null || mf === band : oPixel;
            /* stopped at the slab, or carried on to the pixel */
            const atSlab = cont ? true
              : side === 0 ? mf !== null && mf !== band : !oPasses;
            s.photons.push({
              side, band, px, ok, atSlab, cont, col,
              ax: a.x, ay: a.y, tx, ty, x: a.x, y: a.y,
              tStop: atSlab ? (L.filterY - a.y) / (ty - a.y) : 1,
              t: 0, dur: spd.flight, phase: 0,
            });
          }
          s.delivered++; s.pos++;
        }
      }

      /* ---- integrate ---- */
      for (let i = s.photons.length - 1; i >= 0; i--) {
        const p = s.photons[i];
        if (p.phase === 0) {
          if (playRef.current) p.t += dt / p.dur;
          const stop = Math.min(1, p.tStop);
          if (p.t >= stop) {
            p.x = p.ax + (p.tx - p.ax) * stop;
            p.y = p.ay + (p.ty - p.ay) * stop;
            if (p.ok) {
              if (p.side === 0) {
                s.mono[p.px]++; s.monoTot++; s.monoFlash[p.px] = 1;
                if (monoFilterAt(Math.max(0, s.pos - 1), m) === null) s.monoAll++;
                else s.monoBand[p.band]++;
              } else {
                s.osc[p.px]++; s.oscBand[p.band]++; s.oscTot++; s.oscFlash[p.px] = 1;
              }
              s.photons.splice(i, 1);
              continue;
            }
            p.phase = 1;
            if (p.side === 0) { pushQ(s.monoQ, p.col); s.monoLost++; }
            else { pushQ(s.oscQ, p.col); s.oscLost++; if (p.atSlab) s.oscSlabLost++; }
            const pile = p.side === 0 ? s.monoSlate : s.oscSlate;
            const dest = L.slateSlot(p.side === 0 ? L.monoPX : L.oscPX,
              Math.min(L.slateCap - 1, pile.length));
            p.x0 = p.x; p.y0 = p.y; p.sx = dest.x; p.sy = dest.y; p.t2 = 0;
          } else {
            p.x = p.ax + (p.tx - p.ax) * p.t;
            p.y = p.ay + (p.ty - p.ay) * p.t;
          }
        } else {
          if (playRef.current) p.t2 += dt / 1.15;
          const e = Math.min(1, p.t2);
          p.x = p.x0 + (p.sx - p.x0) * e;
          p.y = p.y0 + (p.sy - p.y0) * (e * e);
          if (e >= 1) { s.photons.splice(i, 1); continue; }
        }
      }
      for (let i = 0; i < 4; i++) {
        s.monoFlash[i] = Math.max(0, s.monoFlash[i] - dt * 2.2);
        s.oscFlash[i] = Math.max(0, s.oscFlash[i] - dt * 2.2);
      }

      const scale = scaleFor(s.delivered, L.slateCap);
      syncSlate(s.monoSlate, s.monoQ, Math.round(s.monoLost / scale), L.slateCap);
      syncSlate(s.oscSlate, s.oscQ, Math.round(s.oscLost / scale), L.slateCap);

      const posC = Math.min(s.pos, budget - 1);
      const mf = monoFilterAt(posC, m), of = oscFilterAt(posC, m);
      const key = `${mf}|${of}`;
      if (key !== s.lastF) { s.lastF = key; s.pulse = 1; }
      s.pulse = Math.max(0, s.pulse - dt * 1.4);

      /* ---- draw ---- */
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) {
        cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      drawPanel(ctx, L, true, s, M, mf, scale);
      drawPanel(ctx, L, false, s, M, of, scale);

      const inbound = new Set();
      for (const p of s.photons) if (p.phase === 0) inbound.add(p.side * 4 + p.px);
      for (const k of inbound) {
        const c = L.center((k / 4 | 0) === 0 ? L.monoPX : L.oscPX, k % 4);
        ctx.strokeStyle = "rgba(212,169,74,0.30)";
        ctx.lineWidth = 1.4;
        roundRect(ctx, c.x - L.cell / 2, c.y - L.cell / 2, L.cell, L.cell, 7);
        ctx.stroke();
      }

      for (const p of s.photons) {
        const col = p.col || M.bands[p.band].color;
        if (p.phase === 0) {
          const back = Math.max(0, p.t - 0.16);
          const bx = p.ax + (p.tx - p.ax) * back, by = p.ay + (p.ty - p.ay) * back;
          const grd = ctx.createLinearGradient(bx, by, p.x, p.y);
          grd.addColorStop(0, "rgba(0,0,0,0)");
          grd.addColorStop(1, col);
          ctx.globalAlpha = 0.42; ctx.strokeStyle = grd;
          ctx.lineWidth = 2.4; ctx.lineCap = "round";
          ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(p.x, p.y); ctx.stroke();
          ctx.globalAlpha = 1;
          ctx.globalAlpha = p.cont ? 0.14 : 0.26;
          ctx.fillStyle = col;
          ctx.beginPath(); ctx.arc(p.x, p.y, p.cont ? 4.5 : 8, 0, 6.2832); ctx.fill();
          ctx.globalAlpha = 1;
          ctx.fillStyle = col;
          ctx.beginPath(); ctx.arc(p.x, p.y, p.cont ? 2.8 : 3.6, 0, 6.2832); ctx.fill();
        } else {
          ctx.globalAlpha = 0.5; ctx.fillStyle = col;
          ctx.beginPath(); ctx.arc(p.x, p.y, 2.5, 0, 6.2832); ctx.fill();
          ctx.globalAlpha = 1;
        }
      }

      uiAcc += dt;
      if (uiAcc > 0.08) {
        uiAcc = 0;
        setUi({
          delivered: s.delivered, monoTot: s.monoTot, oscTot: s.oscTot,
          monoAll: s.monoAll, monoBand: [...s.monoBand], oscBand: [...s.oscBand],
          monoLost: s.monoLost, oscLost: s.oscLost, oscSlabLost: s.oscSlabLost,
          contDelivered: s.contDelivered,
          pos: Math.min(s.pos, budget), done: s.done, filter: mf, oscFilter: of,
        });
      }
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  /* ---------------- panel painter ---------------- */
  function drawPanel(ctx, L, isMono, s, M, filt, scale) {
    const hub = hubRef.current;
    const px = isMono ? L.monoPX : L.oscPX;
    const counts = isMono ? s.mono : s.osc;
    const flash = isMono ? s.monoFlash : s.oscFlash;
    const lost = isMono ? s.monoLost : s.oscLost;
    const midX = px + L.panelW / 2;
    const ap = L.apex(px);
    const isNB = M.group === "nb";

    ctx.textAlign = "center";
    ctx.font = "600 11px ui-monospace, Menlo, monospace";
    ctx.fillStyle = isMono ? C.bright : "#8FA6C8";
    lsText(ctx, isMono ? "MONO" : "OSC · BAYER", midX, L.labelY, 2.4);

    const fBands = filt === null ? null : isMono ? [filt] : filt;
    const fLabel = filt === null
      ? isMono ? "L — passes everything" : "UV/IR cut — passes everything"
      : isMono
        ? `${M.bands[filt].name} — blocks the rest`
        : `${filt.map((b) => M.bands[b].name).join(" + ")} duoband`;
    ctx.font = "500 11px ui-monospace, Menlo, monospace";
    ctx.fillStyle = filt === null ? (isMono ? C.gold : C.dim)
      : isMono ? bandCol(M, filt) : C.mid;
    ctx.fillText(fLabel, midX, L.badgeY);

    roundRect(ctx, px, L.panelTop, L.panelW, L.panelH, 14);
    ctx.fillStyle = C.panel; ctx.fill();
    ctx.strokeStyle = C.edge; ctx.lineWidth = 1; ctx.stroke();

    for (let i = 0; i < 4; i++) {
      const c = L.center(px, i);
      ctx.strokeStyle = "rgba(120,150,200,0.115)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(ap.x, ap.y); ctx.lineTo(c.x, c.y); ctx.stroke();
    }

    ctx.fillStyle = "rgba(212,169,74,0.85)";
    ctx.beginPath(); ctx.arc(ap.x, ap.y, 3, 0, 6.2832); ctx.fill();
    ctx.strokeStyle = "rgba(212,169,74,0.28)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ap.x - 15, ap.y); ctx.lineTo(ap.x + 15, ap.y); ctx.stroke();
    ctx.font = "500 9px ui-monospace, Menlo, monospace";
    ctx.fillStyle = C.dim; ctx.textAlign = "center";
    ctx.fillText("aperture", ap.x, ap.y - 10);

    /* the slab: mono always has one; the OSC only in narrowband */
    if (isMono || isNB) {
      const gx = L.gridX(px);
      const sw = L.gw + 22, sxx = gx - 11, sy = L.filterY - 7;
      const cols = fBands ? fBands.map((b) => bandCol(M, b)) : ["#E8EEF9"];
      const pl = s.pulse;
      roundRect(ctx, sxx, sy, sw, 14, 4);
      if (cols.length === 1) {
        const t = hexRgb(cols[0]);
        ctx.fillStyle = `rgba(${t[0]},${t[1]},${t[2]},${(fBands ? 0.24 : 0.07) + pl * 0.3})`;
      } else {
        const g = ctx.createLinearGradient(sxx, 0, sxx + sw, 0);
        g.addColorStop(0, cols[0] + "55");
        g.addColorStop(0.5, "rgba(10,16,28,0.35)");
        g.addColorStop(1, cols[1] + "55");
        ctx.fillStyle = g;
      }
      ctx.fill();
      const edge = hexRgb(cols[0]);
      ctx.strokeStyle = `rgba(${edge[0]},${edge[1]},${edge[2]},${(fBands ? 0.6 : 0.3) + pl * 0.35})`;
      ctx.lineWidth = 1 + pl * 1.6;
      ctx.stroke();
      ctx.font = "700 9px ui-monospace, Menlo, monospace";
      ctx.fillStyle = fBands ? bandCol(M, fBands[0]) : C.mid;
      ctx.textAlign = "left";
      ctx.fillText(fBands ? fBands.map((b) => M.bands[b].name).join("+") : "L", sxx + sw + 6, L.filterY + 3.5);
      ctx.textAlign = "right";
      ctx.font = "500 8.5px ui-monospace, Menlo, monospace";
      ctx.fillStyle = C.dim;
      ctx.fillText("filter", sxx - 6, L.filterY + 3);
    }

    /* cells — the dye is always RGGB, whatever light is arriving */
    const numSize = Math.round(L.cell * 0.36);
    for (let i = 0; i < 4; i++) {
      const c = L.center(px, i), n = counts[i];
      const b = 1 - Math.exp(-n / 40);
      const base = isMono ? [232, 238, 249] : hexRgb(DYE_COL[DYE[i]]);
      const x = c.x - L.cell / 2, y = c.y - L.cell / 2;
      roundRect(ctx, x, y, L.cell, L.cell, 7);
      ctx.fillStyle = `rgba(${base[0]},${base[1]},${base[2]},${0.07 + b * 0.85})`;
      ctx.fill();
      ctx.strokeStyle = flash[i] > 0 ? C.gold : C.edgeHi;
      ctx.lineWidth = flash[i] > 0 ? 2 : 1;
      ctx.stroke();
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.font = `700 ${numSize}px ui-monospace, Menlo, monospace`;
      ctx.fillStyle = b > 0.45 ? "rgba(7,11,20,0.88)" : C.mid;
      ctx.fillText(String(n), c.x, c.y + 1);
      if (!isMono) {
        ctx.font = "700 9px ui-monospace, Menlo, monospace";
        ctx.fillStyle = b > 0.45 ? "rgba(7,11,20,0.55)" : DYE_COL[DYE[i]];
        ctx.textAlign = "left";
        ctx.fillText(DYE_NAME[DYE[i]], x + 6, y + 10);
      }
      ctx.textBaseline = "alphabetic";
    }

    /* slate */
    const frac = s.delivered ? lost / s.delivered : 0;
    const sx0 = px + L.slatePad;
    ctx.font = "500 9.5px ui-monospace, Menlo, monospace";
    ctx.textAlign = "left"; ctx.fillStyle = C.dim;
    ctx.fillText("DISCARDED", sx0, L.slateY - 8);
    ctx.textAlign = "right";
    ctx.fillStyle = frac > 0.4 ? C.gold : C.dim;
    ctx.fillText(`${lost.toLocaleString()}  ·  ${(frac * 100).toFixed(0)}%`, sx0 + L.slateW, L.slateY - 8);
    ctx.textAlign = "left";

    roundRect(ctx, sx0, L.slateY, L.slateW, SLATE_H, 6);
    ctx.fillStyle = "#0A101C"; ctx.fill();
    ctx.strokeStyle = C.edge; ctx.lineWidth = 1; ctx.stroke();

    const slate = isMono ? s.monoSlate : s.oscSlate;
    const slots = Math.min(L.slateCap, Math.round(s.delivered / scale));
    ctx.fillStyle = "rgba(108,126,156,0.26)";
    for (let k = slate.length; k < slots; k++) {
      const pos = L.slateSlot(px, k);
      ctx.beginPath(); ctx.arc(pos.x, pos.y, L.dotR * 0.62, 0, 6.2832); ctx.fill();
    }
    ctx.globalAlpha = 0.62;
    for (let k = 0; k < Math.min(slate.length, L.slateCap); k++) {
      const pos = L.slateSlot(px, k);
      ctx.fillStyle = slate[k];
      ctx.beginPath(); ctx.arc(pos.x, pos.y, L.dotR, 0, 6.2832); ctx.fill();
    }
    ctx.globalAlpha = 1;

    ctx.font = "500 8.5px ui-monospace, Menlo, monospace";
    ctx.fillStyle = "rgba(102,118,143,0.85)"; ctx.textAlign = "center";
    ctx.fillText(
      scale === 1 ? "one socket = one photon · full slate = whole run"
        : `one socket = ${scale.toLocaleString()} photons`,
      sx0 + L.slateW / 2, L.slateY + SLATE_H + 12
    );
    ctx.textAlign = "left";
  }

  /* ---------------- readouts ---------------- */
  const M = MODES[mode];
  const isNB = M.group === "nb";
  const budget = budgetOf(mode);
  const seq = M.monoSeq || [];
  const N = ui.delivered || 1;
  const ratio = ui.oscTot > 0 ? ui.monoTot / ui.oscTot : 0;
  const snr = ratio > 0 ? Math.sqrt(ratio) : 0;
  const exp = expected(mode);
  const monoSegs = seq.length
    ? seq.map((f) => (f === null
        ? { name: "L", cols: [C.bright] }
        : { name: M.bands[f].name, cols: [bandCol(M, f)] }))
    : [{ name: "L", cols: [C.bright] }];
  const oscSegs = M.oscSeq
    ? M.oscSeq.map((f) => ({
        name: f.map((b) => M.bands[b].name).join("+"),
        cols: f.map((b) => bandCol(M, b)),
      }))
    : [{ name: "UV/IR cut", cols: ["#5B6B85"] }];
  const monoIdx = Math.min(monoSegs.length - 1, Math.floor((ui.pos / budget) * monoSegs.length));
  const oscIdx = Math.min(oscSegs.length - 1, Math.floor((ui.pos / budget) * oscSegs.length));
  const barMax = Math.max(
    1, ui.monoAll || 0,
    ...M.bands.map((_, i) => ui.monoBand[i] || 0),
    ...M.bands.map((_, i) => ui.oscBand[i] || 0)
  );

  return (
    <div style={{ background: C.bg, color: C.text, minHeight: "100%", padding: "18px 16px 26px", fontFamily: "ui-sans-serif, -apple-system, 'Helvetica Neue', Arial, sans-serif" }}>
      <div style={{ maxWidth: 980, margin: "0 auto" }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <h1 style={{ fontSize: 19, fontWeight: 600, margin: 0 }}>Where the mono advantage comes from</h1>
          <span style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 10.5, color: C.gold, border: `1px solid ${C.edgeHi}`, borderRadius: 999, padding: "3px 9px" }}>v0.18</span>
        </div>
        <p style={{ color: C.dim, fontSize: 12.5, lineHeight: 1.55, margin: "6px 0 12px", maxWidth: 720 }}>
          One photon stream, delivered identically to both sensors. Mono's filter
          is a slab in front of the whole sensor. In narrowband the OSC has a slab
          too — its duoband — <em>and</em> a dye on every pixel, so it loses photons
          twice over. Bandwidth cancels: 3 nm and 7 nm give the same ratios.
        </p>
        {isNB && (
          <p style={{ color: C.dim, fontSize: 11.5, lineHeight: 1.55, margin: "0 0 12px", maxWidth: 720 }}>
            Most of what arrives is broadband continuum and sky glow, drawn in
            red, green and blue. The emission lines are the minority — gold for
            Hα, teal for OIII, magenta for SII. Every narrowband filter rejects
            the continuum at the slab on both cameras alike: that rejection is
            the point of the filter, and since it costs both equally it leaves
            every ratio untouched. The line colours are a display choice; which
            pixel a photon lands on still follows true wavelength, so Hα and SII
            both go to the red pixels.
          </p>
        )}

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 10 }}>
          <Tag>LRGB</Tag>
          <Seg options={ORDER_BB.map((k) => ({ k, label: MODES[k].label }))} value={mode}
            onChange={(k) => { setMode(k); reset(k); setPlaying(true); }} />
          <Tag>Narrowband</Tag>
          <Seg options={ORDER_NB.map((k) => ({ k, label: MODES[k].label }))} value={mode}
            onChange={(k) => { setMode(k); reset(k); setPlaying(true); }} />
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 12 }}>
          {ui.done
            ? <Btn onClick={() => { reset(); setPlaying(true); }} accent>Run again</Btn>
            : <Btn onClick={() => setPlaying((p) => !p)} accent>{playing ? "Pause" : "Play"}</Btn>}
          <Seg options={SPEEDS.map((s, i) => ({ k: i, label: s.label }))} value={speed} onChange={setSpeed} />
          <Btn onClick={() => burst(budget)}>Finish session</Btn>
          <Btn onClick={() => { reset(); setPlaying(true); }}>Reset</Btn>
        </div>

        <div style={{ position: "relative", border: `1px solid ${C.edge}`, borderRadius: 16, overflow: "hidden", background: "radial-gradient(120% 80% at 50% -10%, #101A2C 0%, #070B14 62%)" }}>
          <canvas ref={canvasRef} style={{ width: "100%", height: H, display: "block" }} />
        </div>

        {M.monoSeq && (
          <div style={{ marginTop: 10, border: `1px solid ${C.edge}`, borderRadius: 12, padding: "11px 14px", background: C.panel }}>
            <ScheduleBar label="MONO" segs={monoSegs} active={monoIdx} done={ui.done} />
            <div style={{ height: 7 }} />
            <ScheduleBar label="OSC" segs={oscSegs} active={oscIdx} done={ui.done} />
            <div style={{ marginTop: 8, fontFamily: "ui-monospace, Menlo, monospace", fontSize: 9.5, color: ui.done ? C.gold : C.dim, letterSpacing: "0.07em" }}>
              {ui.done
                ? `RUN COMPLETE · ${budget.toLocaleString()} PHOTONS`
                : `${ui.pos} / ${budget} PHOTONS · MONO ${monoSegs[monoIdx].name} · OSC ${oscSegs[oscIdx].name}`}
            </div>
          </div>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(148px, 1fr))", gap: 10, marginTop: 12 }}>
          <Stat label="Photons delivered" value={ui.delivered.toLocaleString()} />
          <Stat label="Mono electrons" value={ui.monoTot.toLocaleString()} sub={`${((ui.monoTot / N) * 100).toFixed(1)}% kept`} tone={C.bright} />
          <Stat label="OSC electrons" value={ui.oscTot.toLocaleString()} sub={`${((ui.oscTot / N) * 100).toFixed(1)}% kept`} tone="#8FA6C8" />
          <Stat label="Photon ratio = time advantage" value={ratio ? `${ratio.toFixed(2)}×` : "—"} sub={`converges to ${exp.ratio.toFixed(2)}×`} tone={C.gold} />
          <Stat label="SNR advantage" value={snr ? `${snr.toFixed(2)}×` : "—"} sub="= √(photon ratio)" tone={C.gold} />
          {isNB && (
            <Stat label="Continuum and sky rejected" value={ui.contDelivered.toLocaleString()}
              sub="thrown away by both" tone="#7F8FA8" />
          )}
        </div>

        {isNB && (
          <div style={{ marginTop: 10, border: `1px solid ${C.edge}`, borderRadius: 12, padding: "12px 14px", background: C.panel }}>
            <div style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 10, color: C.dim, letterSpacing: "0.09em", marginBottom: 10 }}>
              PER-LINE ADVANTAGE — WHERE THE REAL STORY IS
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 10 }}>
              {M.bands.map((b, i) => {
                const r = exp.oscBand[i] > 0 ? exp.monoBand[i] / exp.oscBand[i] : 0;
                const monoWins = r >= 1;
                return (
                  <div key={i} style={{ border: `1px solid ${C.edge}`, borderRadius: 10, padding: "9px 11px" }}>
                    <div style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 12, fontWeight: 700, color: bandCol(M, i), marginBottom: 4 }}>{b.name}</div>
                    <div style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 18, fontWeight: 600, color: monoWins ? C.bright : "#8FA6C8" }}>
                      {(monoWins ? r : 1 / r).toFixed(2)}×
                    </div>
                    <div style={{ fontSize: 13, fontWeight: 500, color: C.mid, marginTop: 4, letterSpacing: "0.01em" }}>
                      {monoWins ? "faster on mono" : "faster on the OSC"}
                    </div>
                    <div style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 10, color: C.dim, marginTop: 5 }}>
                      mono {(ui.monoBand[i] || 0).toLocaleString()} · osc {(ui.oscBand[i] || 0).toLocaleString()}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 10, marginTop: 10 }}>
          <Card title="MONO — WHAT IT COLLECTED">
            {!isNB && <Row label="L" color={C.bright} value={ui.monoAll} max={barMax} />}
            {M.bands.map((b, i) => <Row key={i} label={b.name} color={bandCol(M, i)} value={ui.monoBand[i]} max={barMax} />)}
            <Total label="Total electrons" value={ui.monoTot} />
            <Note>
              {isNB
                ? "Every pixel collects every line — but only one line at a time, because there is one filter for the whole sensor."
                : M.w === 0
                  ? "No luminance frames at all, so mono and the OSC end on exactly the same total. Mono partitions the spectrum in time, the OSC in space."
                  : M.w === null
                    ? "Nothing is filtered out, so every photon delivered is recorded. This is the ceiling the other splits trade against."
                    : "Separate populations: the L frames and the colour frames are different photons, so they add."}
            </Note>
          </Card>

          <Card title="OSC — WHAT IT COLLECTED">
            {M.bands.map((b, i) => <Row key={i} label={b.name} color={bandCol(M, i)} value={ui.oscBand[i]} max={barMax} />)}
            {!isNB && <Row label="L" color={C.dim} value={ui.oscTot} max={barMax} ghost />}
            <Total label="Total electrons" value={ui.oscTot} />
            <Note>
              {isNB
                ? <>Two losses, not one: {ui.oscSlabLost.toLocaleString()} photons stopped at the duoband, {(ui.oscLost - ui.oscSlabLost).toLocaleString()} at a pixel whose dye didn't match. OIII does best — at 500.7 nm it is seen by all the green pixels and most of the blue, about 65% of the array, against 25% for Hα and SII.</>
                : "Its luminance is the same electrons as the colour rows, not additional ones — twice as many green pixels, so synthetic L is green-weighted rather than flat."}
            </Note>
          </Card>
        </div>

        <p style={{ color: C.dim, fontSize: 11.5, lineHeight: 1.6, marginTop: 14, maxWidth: 740 }}>
          In SHO the OSC needs two duobands and so gives up the one thing it had —
          simultaneity — while keeping the pixel handicap. OIII is collected under
          both filters and ends up over-served; Hα and SII starve. And on a Bayer
          array Hα and SII are not merely slower but inseparable, since both land
          on the same red pixels: only the filter in front tells them apart.
        </p>
      </div>
    </div>
  );
}

/* ---------------- UI ---------------- */

function ScheduleBar({ label, segs, active, done }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 9.5, color: C.dim, letterSpacing: "0.1em", width: 34 }}>{label}</span>
      <div style={{ flex: 1, display: "flex", gap: 3, height: 8 }}>
        {segs.map((sg, i) => (
          <div key={i} title={sg.name} style={{
            flex: 1,
            borderRadius: 3,
            opacity: done ? 0.45 : i === active ? 1 : i < active ? 0.38 : 0.13,
            background: sg.cols.length > 1
              ? `linear-gradient(90deg, ${sg.cols[0]}, ${sg.cols[1]})`
              : sg.cols[0],
          }} />
        ))}
      </div>
      <span style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 9.5, color: done ? C.dim : C.mid, width: 74, textAlign: "right" }}>
        {segs[active].name}
      </span>
    </div>
  );
}

function Tag({ children }) {
  return <span style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 9.5, color: C.dim, letterSpacing: "0.1em", textTransform: "uppercase" }}>{children}</span>;
}
function Card({ title, children }) {
  return (
    <div style={{ border: `1px solid ${C.edge}`, borderRadius: 12, padding: "12px 14px", background: C.panel }}>
      <div style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 10, color: C.dim, letterSpacing: "0.09em", marginBottom: 10 }}>{title}</div>
      {children}
    </div>
  );
}
function Row({ label, color, value, max, ghost }) {
  const v = Number(value) || 0;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 6 }}>
      <span style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 11, fontWeight: 700, color, width: 28 }}>{label}</span>
      <div style={{ flex: 1, height: 8, background: "#141C2E", borderRadius: 4, overflow: "hidden" }}>
        <div style={{ width: `${Math.min(100, (v / max) * 100)}%`, height: "100%", background: ghost ? "transparent" : color, border: ghost ? `1px dashed ${C.edgeHi}` : "none", borderRadius: 4, transition: "width 0.2s linear" }} />
      </div>
      <span style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 12, color: ghost ? C.dim : C.text, width: 52, textAlign: "right" }}>{v.toLocaleString()}</span>
    </div>
  );
}
function Total({ label, value }) {
  const v = Number(value) || 0;
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", borderTop: `1px solid ${C.edge}`, marginTop: 9, paddingTop: 8 }}>
      <span style={{ fontSize: 11, color: C.dim }}>{label}</span>
      <span style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 16, fontWeight: 600, color: C.gold }}>{v.toLocaleString()}</span>
    </div>
  );
}
function Note({ children }) {
  return <div style={{ fontSize: 11, color: C.dim, lineHeight: 1.5, marginTop: 8 }}>{children}</div>;
}
function Btn({ children, onClick, accent }) {
  const [h, setH] = useState(false);
  return (
    <button onClick={onClick} onPointerEnter={() => setH(true)} onPointerLeave={() => setH(false)}
      style={{ background: accent ? (h ? "#E3BC5C" : C.gold) : h ? "#182236" : C.panel, color: accent ? "#0A0E17" : C.text, border: `1px solid ${accent ? C.gold : C.edgeHi}`, borderRadius: 8, padding: "7px 13px", fontSize: 12, fontWeight: 600, cursor: "pointer", fontFamily: "inherit" }}>
      {children}
    </button>
  );
}
function Seg({ options, value, onChange }) {
  return (
    <div style={{ display: "flex", border: `1px solid ${C.edgeHi}`, borderRadius: 8, overflow: "hidden" }}>
      {options.map((o) => {
        const on = o.k === value;
        return (
          <button key={String(o.k)} onClick={() => onChange(o.k)}
            style={{ background: on ? "#1D2942" : "transparent", color: on ? C.bright : C.dim, border: "none", padding: "7px 12px", fontSize: 12, fontWeight: on ? 600 : 500, cursor: "pointer", fontFamily: "inherit" }}>
            {o.label}
          </button>
        );
      })}
    </div>
  );
}
function Stat({ label, value, sub, tone }) {
  return (
    <div style={{ border: `1px solid ${C.edge}`, borderRadius: 12, padding: "10px 12px", background: C.panel }}>
      <div style={{ fontSize: 9.5, color: C.dim, letterSpacing: "0.07em", textTransform: "uppercase", fontFamily: "ui-monospace, Menlo, monospace", marginBottom: 5, lineHeight: 1.3 }}>{label}</div>
      <div style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 20, fontWeight: 600, color: tone || C.text, lineHeight: 1.1 }}>{value}</div>
      {sub && <div style={{ fontSize: 10.5, color: C.dim, marginTop: 3, fontFamily: "ui-monospace, Menlo, monospace" }}>{sub}</div>}
    </div>
  );
}

/* ---------------- helpers ---------------- */

function lsText(ctx, text, cx, y, ls) {
  const chars = Array.from(text);
  const widths = chars.map((c) => ctx.measureText(c).width);
  const total = widths.reduce((a, b) => a + b, 0) + ls * (chars.length - 1);
  const prev = ctx.textAlign;
  ctx.textAlign = "left";
  let x = cx - total / 2;
  for (let i = 0; i < chars.length; i++) { ctx.fillText(chars[i], x, y); x += widths[i] + ls; }
  ctx.textAlign = prev;
}
function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}
function hexRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
