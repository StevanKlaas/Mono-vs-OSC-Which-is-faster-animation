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

/* why each line fares as it does on a Bayer array */
const LINE_WHY = {
  "Hα": "Red pixels only, so three in four are blind to it — always a flat no, never a partial one. Mono loses it solely while another filter is on the wheel.",
  "OIII": "Green transmits it fully, red not at all, and blue about six times in ten — 500.7 nm sits where the blue dye is already closing. That gives 0.5 + 0.25 × 0.6 = 0.65 of the array, beating mono's 50% duty cycle. The only line a colour sensor wins.",
  "SII": "Red pixels only, exactly like Hα — and indistinguishable from it there. Only the filter in front can separate the two.",
};

/* Relative photon flux at the three lines for real targets. SII is the
   6717+6731 doublet. These are representative rather than measured: line
   ratios vary strongly within a single nebula, M42 most of all. */
const TARGETS = [
  { k: "equal", g: "Reference", label: "Equal mix", mix: [1, 1, 1] },

  { k: "ic1396", g: "Emission nebulae", label: "IC 1396 Elephant's Trunk", mix: [100, 10, 22] },
  { k: "sh2155", g: "Emission nebulae", label: "Sh2-155 Cave", mix: [100, 12, 22] },
  { k: "ngc7000", g: "Emission nebulae", label: "NGC 7000 North America", mix: [100, 12, 20] },
  { k: "ic1805", g: "Emission nebulae", label: "IC 1805 Heart", mix: [100, 15, 20] },
  { k: "sh2129", g: "Emission nebulae", label: "Sh2-129 + Ou4 field", mix: [100, 18, 12] },
  { k: "m16", g: "Emission nebulae", label: "M16 Eagle", mix: [100, 25, 18] },
  { k: "rosette", g: "Emission nebulae", label: "NGC 2237 Rosette", mix: [100, 30, 20] },
  { k: "m8", g: "Emission nebulae", label: "M8 Lagoon", mix: [100, 40, 15] },
  { k: "bubble", g: "Emission nebulae", label: "NGC 7635 Bubble", mix: [100, 45, 15] },
  { k: "crescent", g: "Emission nebulae", label: "NGC 6888 Crescent", mix: [100, 55, 12] },
  { k: "veil", g: "Emission nebulae", label: "Veil NGC 6992 (remnant)", mix: [100, 70, 55] },
  { k: "m42", g: "Emission nebulae", label: "M42 Orion core", mix: [100, 90, 12] },

  { k: "helix", g: "Planetary nebulae", label: "NGC 7293 Helix", mix: [100, 120, 25] },
  { k: "ngc6781", g: "Planetary nebulae", label: "NGC 6781", mix: [100, 150, 20] },
  { k: "m27", g: "Planetary nebulae", label: "M27 Dumbbell", mix: [100, 350, 8] },
  { k: "m57", g: "Planetary nebulae", label: "M57 Ring", mix: [100, 400, 10] },
  { k: "cateye", g: "Planetary nebulae", label: "NGC 6543 Cat's Eye", mix: [100, 450, 5] },
  { k: "saturn", g: "Planetary nebulae", label: "NGC 7009 Saturn", mix: [100, 500, 5] },
  { k: "ou4", g: "Planetary nebulae", label: "Ou4 Giant Squid alone", mix: [1, 100, 1] },
];
const TARGET = Object.fromEntries(TARGETS.map((t) => [t.k, t]));
const UNIFORM = [1, 1, 1];

/* weights over just the bands this scenario uses, renormalised */
function mixWeights(M, mix) {
  const nb = M.bands.length;
  const w = mix.slice(0, nb);
  const tot = w.reduce((a, b) => a + b, 0) || 1;
  return w.map((v) => v / tot);
}
function pickBand(w) {
  let r = Math.random();
  for (let i = 0; i < w.length; i++) { r -= w[i]; if (r <= 0) return i; }
  return w.length - 1;
}

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
    oscSeq: [[1, 2], [0, 1]],   // OIII+SII first, then Ha+OIII
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
function expected(m, mix) {
  const M = MODES[m];
  const nb = M.bands.length;
  const w = mixWeights(M, mix || UNIFORM);
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
      mono += (mOk * w[k]) / S;
      osc += (oOk * w[k]) / S;
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

function freshSim(nb, ol = 1) {
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
    deliveredBand: new Array(nb).fill(0),
    /* every mono capture, attributed to the colour of the photon rather
       than to the filter that happened to be mounted */
    monoAllBand: new Array(nb).fill(0),
    monoLostBand: new Array(nb).fill(0),
    oscSlabBand: new Array(nb).fill(0),
    /* band x mounted filter, so a loss can name the filter that caused it */
    monoLostBy: Array.from({ length: nb }, () => new Array(nb).fill(0)),
    oscSlabBy: Array.from({ length: nb }, () => new Array(ol).fill(0)),
    oscDyeBand: new Array(nb).fill(0),  // dye has no transmission there
    oscPartBand: new Array(nb).fill(0), // dye transmits, but only partly
    pos: 0, acc: 0, done: false, lastF: "?", pulse: 0,
  };
}

/* ---------------- layout ---------------- */

function makeLayout(W, H, budget) {
  const cell = Math.round(Math.max(52, Math.min(146, W * 0.142)));
  const gap = Math.round(cell * 0.09);
  const gw = cell * 2 + gap, gh = cell * 2 + gap;

  /* the slate is a tall column on the outer edge of each panel:
     left of mono, right of the OSC, so the two sensors mirror */
  const slateW = Math.round(Math.max(30, Math.min(56, cell * 0.38)));
  const slatePad = 14;
  const slateGap = Math.round(cell * 0.13);
  const panelW = Math.round(gw + slateW + slateGap + slatePad * 2);
  const panelGap = Math.round(cell * 0.26);

  const labelY = 22, badgeY = 42, panelTop = 58;
  const apexY = panelTop + 34;
  const gridTop = apexY + Math.round(cell * 0.86);
  const filterY = apexY + (gridTop - apexY) * 0.56;
  const panelH = gridTop + gh + 42 - panelTop;

  const monoPX = Math.round(W / 2 - panelGap / 2 - panelW);
  const oscPX = Math.round(W / 2 + panelGap / 2);

  const gridX = (px, isMono) =>
    isMono ? px + slatePad + slateW + slateGap : px + slatePad;
  const slateX = (px, isMono) =>
    isMono ? px + slatePad : px + panelW - slatePad - slateW;
  const center = (px, i, isMono) => ({
    x: gridX(px, isMono) + (i % 2) * (cell + gap) + cell / 2,
    y: gridTop + ((i / 2) | 0) * (cell + gap) + cell / 2,
  });
  const apex = (px, isMono) => ({ x: gridX(px, isMono) + gw / 2, y: apexY });

  const sg = slateGrid(budget, slateW, gh);
  const slateSlot = (px, isMono, slot) => ({
    x: slateX(px, isMono) + (((slot % sg.cols) + 0.5) * slateW) / sg.cols,
    y: gridTop + gh - ((Math.floor(slot / sg.cols) + 0.5) * gh) / sg.rows,
  });

  return {
    cell, gap, gw, gh, panelW, panelH, panelTop, labelY, badgeY,
    apexY, gridTop, filterY, monoPX, oscPX, gridX, slateX, center, apex,
    slatePad, slateW, slateGap, slateCap: sg.cols * sg.rows, slateSlot,
    dotR: Math.max(0.8, Math.min(slateW / sg.cols, gh / sg.rows) * 0.34),
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
  const [docs, setDocs] = useState(false);
  const [target, setTarget] = useState("equal");
  const tgtRef = useRef(UNIFORM);
  const [caveats, setCaveats] = useState(false);
  const hubRef = useRef(false);
  const [ui, setUi] = useState({
    delivered: 0, monoTot: 0, oscTot: 0, monoAll: 0,
    monoBand: [0, 0, 0], oscBand: [0, 0, 0],
    monoLost: 0, oscLost: 0, oscSlabLost: 0, contDelivered: 0,
    monoLostBand: [0, 0, 0], oscSlabBand: [0, 0, 0],
    oscDyeBand: [0, 0, 0], oscPartBand: [0, 0, 0],
    deliveredBand: [0, 0, 0], monoAllBand: [0, 0, 0], monoLostBy: [], oscSlabBy: [],
    pos: 0, done: false, filter: null, oscFilter: null,
  });

  useEffect(() => void (modeRef.current = mode), [mode]);
  useEffect(() => void (spdRef.current = SPEEDS[speed]), [speed]);
  useEffect(() => void (playRef.current = playing), [playing]);
  useEffect(() => { tgtRef.current = TARGET[target].mix; }, [target]);
  useEffect(() => {
    const m = typeof window !== "undefined" && window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)");
    if (m && m.matches) { setPlaying(false); playRef.current = false; }
  }, []);

  const reset = useCallback((m) => {
    const MM = MODES[m || modeRef.current];
    const nb = MM.bands.length, ol = MM.oscSeq ? MM.oscSeq.length : 1;
    sim.current = freshSim(nb, ol);
    setUi({
      delivered: 0, monoTot: 0, oscTot: 0, monoAll: 0,
      monoBand: new Array(nb).fill(0), oscBand: new Array(nb).fill(0),
      monoLost: 0, oscLost: 0, oscSlabLost: 0, contDelivered: 0,
      deliveredBand: new Array(nb).fill(0), monoAllBand: new Array(nb).fill(0),
      monoLostBand: new Array(nb).fill(0),
      oscSlabBand: new Array(nb).fill(0),
      oscDyeBand: new Array(nb).fill(0), oscPartBand: new Array(nb).fill(0),
      monoLostBy: Array.from({ length: nb }, () => new Array(nb).fill(0)),
      oscSlabBy: Array.from({ length: nb }, () => new Array(ol).fill(0)),
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
      const band = M.group === "nb" ? pickBand(mixWeights(M, tgtRef.current)) : (Math.random() * nb) | 0;
      const bc = bandCol(M, band);
      s.deliveredBand[band]++;
      const oi = idxAt(M.oscSeq, s.pos - 1, budgetOf(m));
      if (mf === null) { s.mono[px]++; s.monoAll++; s.monoTot++; s.monoAllBand[band]++; }
      else if (mf === band) { s.mono[px]++; s.monoBand[band]++; s.monoTot++; s.monoAllBand[band]++; }
      else {
        pushQ(s.monoQ, bc); s.monoLost++; s.monoLostBand[band]++;
        s.monoLostBy[band][mf]++;
      }

      const passes = of === null || of.includes(band);
      const resp = M.resp[DYE[px]][band];
      if (passes && Math.random() < resp) {
        s.osc[px]++; s.oscBand[band]++; s.oscTot++;
      } else {
        pushQ(s.oscQ, bc); s.oscLost++;
        if (!passes) { s.oscSlabLost++; s.oscSlabBand[band]++; s.oscSlabBy[band][oi]++; }
        else if (resp === 0) s.oscDyeBand[band]++;
        else s.oscPartBand[band]++;
      }
    }
    if (M.monoSeq && s.pos >= budgetOf(m)) s.done = true;
  }, []);

  const [canvasH, setCanvasH] = useState(640);
  const hRef = useRef(640);

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
      const L = makeLayout(W, 0, budget);
      const H = L.panelTop + L.panelH + 14;
      if (Math.abs(H - hRef.current) > 1) { hRef.current = H; setCanvasH(H); }
      const spd = spdRef.current;

      /* ---- spawn ---- */
      if (playRef.current && !s.done) {
        s.acc += dt * spd.rate;
        while (s.acc >= 1) {
          s.acc -= 1;
          if (M.monoSeq && s.pos >= budget) { s.done = true; break; }
          const cont = Math.random() < (M.cont || 0);
          const band = cont ? 0
            : M.group === "nb" ? pickBand(mixWeights(M, tgtRef.current))
            : (Math.random() * nb) | 0;
          const px = (Math.random() * 4) | 0;
          const mf = monoFilterAt(s.pos, m), of = oscFilterAt(s.pos, m);
          const jx = 0, jy = 0; // land dead centre, so the ray is the path
          const col = cont
            ? CONT_COL[(Math.random() * 3) | 0]
            : bandCol(M, band);
          const oPasses = !cont && (of === null || of.includes(band));
          const oResp = cont ? 0 : M.resp[DYE[px]][band];
          const oPixel = oPasses && Math.random() < oResp;
          if (cont) s.contDelivered++; else s.deliveredBand[band]++;
          for (const side of [0, 1]) {
            const base = side === 0 ? L.monoPX : L.oscPX;
            const a = L.apex(base, side === 0), c = L.center(base, px, side === 0);
            const tx = c.x + jx, ty = c.y + jy;
            const ok = cont ? false
              : side === 0 ? mf === null || mf === band : oPixel;
            /* stopped at the slab, or carried on to the pixel */
            const atSlab = cont ? true
              : side === 0 ? mf !== null && mf !== band : !oPasses;
            s.photons.push({
              side, band, px, ok, atSlab, cont, col, resp: oResp,
              mf, oi: idxAt(M.oscSeq, s.pos, budget),
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
                if (!p.cont) s.monoAllBand[p.band]++;
                if (p.mf === null) s.monoAll++;
                else s.monoBand[p.band]++;
              } else {
                s.osc[p.px]++; s.oscBand[p.band]++; s.oscTot++; s.oscFlash[p.px] = 1;
              }
              s.photons.splice(i, 1);
              continue;
            }
            p.phase = 1;
            if (p.side === 0) {
              pushQ(s.monoQ, p.col); s.monoLost++;
              if (!p.cont) {
                s.monoLostBand[p.band]++;
                if (p.mf !== null) s.monoLostBy[p.band][p.mf]++;
              }
            } else {
              pushQ(s.oscQ, p.col); s.oscLost++;
              if (p.atSlab) s.oscSlabLost++;
              if (!p.cont) {
                if (p.atSlab) { s.oscSlabBand[p.band]++; s.oscSlabBy[p.band][p.oi]++; }
                else if (p.resp === 0) s.oscDyeBand[p.band]++;
                else s.oscPartBand[p.band]++;
              }
            }
            const pile = p.side === 0 ? s.monoSlate : s.oscSlate;
            const dest = L.slateSlot(
              p.side === 0 ? L.monoPX : L.oscPX, p.side === 0,
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
        const isM = (k / 4 | 0) === 0;
        const c = L.center(isM ? L.monoPX : L.oscPX, k % 4, isM);
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
          ctx.lineWidth = 3.2; ctx.lineCap = "round";
          ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(p.x, p.y); ctx.stroke();
          ctx.globalAlpha = 1;
          /* narrowband line photons carry their letter until they land */
          const isLine = M.group === "nb" && !p.cont;
          ctx.globalAlpha = p.cont ? 0.14 : 0.26;
          ctx.fillStyle = col;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.cont ? 4.5 : isLine ? 11 : 8, 0, 6.2832);
          ctx.fill();
          ctx.globalAlpha = 1;
          ctx.fillStyle = col;
          ctx.beginPath();
          ctx.arc(p.x, p.y, p.cont ? 2.8 : isLine ? 6.6 : 3.6, 0, 6.2832);
          ctx.fill();
          if (isLine) {
            ctx.font = "700 10px ui-monospace, Menlo, monospace";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillStyle = "rgba(7,11,20,0.88)";
            ctx.fillText(M.bands[p.band].name[0], p.x, p.y + 0.5);
            ctx.textBaseline = "alphabetic";
            ctx.textAlign = "left";
          }
        } else {
          ctx.globalAlpha = 0.22; ctx.fillStyle = col;
          ctx.beginPath(); ctx.arc(p.x, p.y, 7, 0, 6.2832); ctx.fill();
          ctx.globalAlpha = 0.72;
          ctx.beginPath(); ctx.arc(p.x, p.y, 4.2, 0, 6.2832); ctx.fill();
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
          monoLostBand: [...s.monoLostBand],
          oscSlabBand: [...s.oscSlabBand],
          oscDyeBand: [...s.oscDyeBand], oscPartBand: [...s.oscPartBand],
          deliveredBand: [...s.deliveredBand], monoAllBand: [...s.monoAllBand],
          monoLostBy: s.monoLostBy.map((r) => [...r]),
          oscSlabBy: s.oscSlabBy.map((r) => [...r]),
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
    const ap = L.apex(px, isMono);
    const isNB = M.group === "nb";

    ctx.textAlign = "center";
    ctx.font = "600 12.5px ui-monospace, Menlo, monospace";
    ctx.fillStyle = isMono ? C.bright : "#8FA6C8";
    lsText(ctx, isMono ? "MONO" : "OSC · BAYER", midX, L.labelY, 2.4);

    const fBands = filt === null ? null : isMono ? [filt] : filt;
    const fLabel = filt === null
      ? isMono ? "L — passes everything" : "UV/IR cut — passes everything"
      : isMono
        ? `${M.bands[filt].name} — blocks the rest`
        : `${filt.map((b) => M.bands[b].name).join(" + ")} duoband`;
    ctx.font = "500 12.5px ui-monospace, Menlo, monospace";
    ctx.fillStyle = filt === null ? (isMono ? C.gold : C.dim)
      : isMono ? bandCol(M, filt) : C.mid;
    ctx.fillText(fLabel, midX, L.badgeY);

    roundRect(ctx, px, L.panelTop, L.panelW, L.panelH, 14);
    ctx.fillStyle = C.panel; ctx.fill();
    ctx.strokeStyle = C.edge; ctx.lineWidth = 1; ctx.stroke();

    for (let i = 0; i < 4; i++) {
      const c = L.center(px, i, isMono);
      ctx.strokeStyle = "rgba(130,162,214,0.22)"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(ap.x, ap.y); ctx.lineTo(c.x, c.y); ctx.stroke();
    }

    ctx.fillStyle = "rgba(212,169,74,0.85)";
    ctx.beginPath(); ctx.arc(ap.x, ap.y, 3, 0, 6.2832); ctx.fill();
    ctx.strokeStyle = "rgba(212,169,74,0.28)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ap.x - 15, ap.y); ctx.lineTo(ap.x + 15, ap.y); ctx.stroke();
    ctx.font = "500 10.5px ui-monospace, Menlo, monospace";
    ctx.fillStyle = C.dim; ctx.textAlign = "center";
    ctx.fillText("aperture", ap.x, ap.y - 11);

    /* the slab: mono always has one; the OSC only in narrowband */
    if (isMono || isNB) {
      const gx = L.gridX(px, isMono);
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
      ctx.font = "700 11px ui-monospace, Menlo, monospace";
      ctx.fillStyle = fBands ? bandCol(M, fBands[0]) : C.mid;
      ctx.textAlign = "left";
      ctx.fillText(fBands ? fBands.map((b) => M.bands[b].name).join("+") : "L", sxx + sw + 6, L.filterY + 3.5);
      ctx.textAlign = "right";
      ctx.font = "500 10px ui-monospace, Menlo, monospace";
      ctx.fillStyle = C.dim;
      ctx.fillText("filter", sxx - 6, L.filterY + 3);
    }

    /* cells — the dye is always RGGB, whatever light is arriving */
    const numSize = Math.round(L.cell * 0.32);
    for (let i = 0; i < 4; i++) {
      const c = L.center(px, i, isMono), n = counts[i];
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
        ctx.font = "700 11.5px ui-monospace, Menlo, monospace";
        ctx.fillStyle = b > 0.45 ? "rgba(7,11,20,0.55)" : DYE_COL[DYE[i]];
        ctx.textAlign = "left";
        ctx.fillText(DYE_NAME[DYE[i]], x + 7, y + 13);
      }
      ctx.textBaseline = "alphabetic";
    }

    /* slate: a tall column on the outer edge */
    const frac = s.delivered ? lost / s.delivered : 0;
    const sx0 = L.slateX(px, isMono);
    const scx = sx0 + L.slateW / 2;

    ctx.textAlign = "center";
    ctx.font = "600 10px ui-monospace, Menlo, monospace";
    ctx.fillStyle = C.dim;
    lsText(ctx, "DISCARDED", scx, L.gridTop - 25, 0.8);
    ctx.font = "600 12px ui-monospace, Menlo, monospace";
    ctx.fillStyle = frac > 0.4 ? C.gold : C.dim;
    ctx.fillText(`${lost.toLocaleString()} · ${(frac * 100).toFixed(0)}%`, scx, L.gridTop - 9);

    roundRect(ctx, sx0, L.gridTop, L.slateW, L.gh, 6);
    ctx.fillStyle = "#0A101C"; ctx.fill();
    ctx.strokeStyle = C.edge; ctx.lineWidth = 1; ctx.stroke();

    const slate = isMono ? s.monoSlate : s.oscSlate;
    const slots = Math.min(L.slateCap, Math.round(s.delivered / scale));
    ctx.fillStyle = "rgba(108,126,156,0.26)";
    for (let k = slate.length; k < slots; k++) {
      const pos = L.slateSlot(px, isMono, k);
      ctx.beginPath(); ctx.arc(pos.x, pos.y, L.dotR * 0.62, 0, 6.2832); ctx.fill();
    }
    ctx.globalAlpha = 0.62;
    for (let k = 0; k < Math.min(slate.length, L.slateCap); k++) {
      const pos = L.slateSlot(px, isMono, k);
      ctx.fillStyle = slate[k];
      ctx.beginPath(); ctx.arc(pos.x, pos.y, L.dotR, 0, 6.2832); ctx.fill();
    }
    ctx.globalAlpha = 1;

    ctx.font = "500 10.5px ui-monospace, Menlo, monospace";
    ctx.fillStyle = "rgba(112,129,156,0.9)"; ctx.textAlign = "center";
    ctx.fillText(
      scale === 1 ? "one socket = one photon · full slate = whole run"
        : `one socket = ${scale.toLocaleString()} photons`,
      px + L.panelW / 2, L.gridTop + L.gh + 26
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
  const tgt = TARGET[target];
  const exp = expected(mode, isNB ? tgt.mix : UNIFORM);
  const wts = mixWeights(M, isNB ? tgt.mix : UNIFORM);
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
  const dBand = ui.deliveredBand || [];
  const pctOf = (v, i) => (dBand[i] > 0 ? ((v || 0) / dBand[i]) * 100 : null);
  const delRows = [
    ...M.bands.map((b, i) => ({ label: b.name, color: bandCol(M, i), value: dBand[i] || 0 })),
    ...(isNB ? [{ label: "sky", color: C.dim, value: ui.contDelivered || 0 }] : []),
  ];
  const mAll = ui.monoAllBand || [];
  const monoRows = M.bands.map((b, i) => ({
    label: b.name, color: bandCol(M, i),
    value: mAll[i] || 0, pct: pctOf(mAll[i], i),
  }));
  const oscRows = M.bands.map((b, i) => ({
    label: b.name, color: bandCol(M, i),
    value: ui.oscBand[i] || 0, pct: pctOf(ui.oscBand[i], i),
  }));

  /* Bars run on one fixed scale shared by both sensors, anchored to where
     the largest row will finish. Nothing is renormalised as counts grow, so
     bar length is an absolute count and the two cards are directly
     comparable by eye. Stochastic rows get 4 sigma of headroom, because a
     count of 78 routinely lands at 84 and must not clip. */
  const lineTotal = budget * (1 - (M.cont || 0));
  const lBlocks = seq.filter((f) => f === null).length;
  const headroom = (e) => e + 4 * Math.sqrt(Math.max(0, e));
  const barMax = niceCeil(Math.max(
    1,
    /* mono's L tally is deterministic: every photon in an L block is kept */
    isNB ? 0 : (budget * lBlocks) / Math.max(1, seq.length),
    isNB ? 0 : headroom(exp.osc * budget),
    ...M.bands.map((_, i) =>
      headroom(Math.max(exp.monoBand[i], exp.oscBand[i]) * lineTotal * wts[i]))
  ));

  return (
    <div style={{ background: C.bg, color: C.text, minHeight: "100%", padding: "18px 16px 26px", fontFamily: "ui-sans-serif, -apple-system, 'Helvetica Neue', Arial, sans-serif" }}>
      <div style={{ maxWidth: 980, margin: "0 auto" }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <h1 style={{ fontSize: 19, fontWeight: 600, margin: 0 }}>Where the mono advantage comes from</h1>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <button onClick={() => setDocs(true)}
              style={{ background: C.gold, color: "#0A0E17", border: `1px solid ${C.gold}`, borderRadius: 8, padding: "6px 12px", fontSize: 12, fontWeight: 700, cursor: "pointer", fontFamily: "inherit" }}>
              How it works
            </button>
            <span style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 10.5, color: C.gold, border: `1px solid ${C.edgeHi}`, borderRadius: 999, padding: "3px 9px" }}>v0.37</span>
          </div>
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

        {isNB && (
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center", marginBottom: 10 }}>
            <Tag>Target</Tag>
            <select
              value={target}
              onChange={(e) => { setTarget(e.target.value); reset(); setPlaying(true); }}
              style={{ background: C.panel, color: C.bright, border: `1px solid ${C.edgeHi}`, borderRadius: 8, padding: "7px 10px", fontSize: 12, fontFamily: "inherit", fontWeight: 600 }}
            >
              {["Reference", "Emission nebulae", "Planetary nebulae"].map((g) => (
                <optgroup key={g} label={g}>
                  {TARGETS.filter((t) => t.g === g).map((t) => (
                    <option key={t.k} value={t.k}>{t.label}</option>
                  ))}
                </optgroup>
              ))}
            </select>
            <span style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 11.5, color: C.dim }}>
              Hα {tgt.mix[0]} · OIII {tgt.mix[1]}
              {M.bands.length > 2 ? ` · SII ${tgt.mix[2]}` : ""}
              {"  →  "}
              <span style={{ color: exp.ratio >= 1 ? C.bright : "#8FA6C8", fontWeight: 700 }}>
                {(exp.ratio >= 1 ? exp.ratio : 1 / exp.ratio).toFixed(2)}× {exp.ratio >= 1 ? "mono" : "OSC"}
              </span>
            </span>
          </div>
        )}

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginBottom: 12 }}>
          {ui.done
            ? <Btn onClick={() => { reset(); setPlaying(true); }} accent>Run again</Btn>
            : <Btn onClick={() => setPlaying((p) => !p)} accent>{playing ? "Pause" : "Play"}</Btn>}
          <Seg options={SPEEDS.map((s, i) => ({ k: i, label: s.label }))} value={speed} onChange={setSpeed} />
          <Btn onClick={() => burst(budget)}>Finish session</Btn>
          <Btn onClick={() => { reset(); setPlaying(true); }}>Reset</Btn>
        </div>

        <div style={{ position: "relative", border: `1px solid ${C.edge}`, borderRadius: 16, overflow: "hidden", background: "radial-gradient(120% 80% at 50% -10%, #101A2C 0%, #070B14 62%)" }}>
          <canvas ref={canvasRef} style={{ width: "100%", height: canvasH, display: "block" }} />
        </div>

        {M.monoSeq && (
          <div style={{ marginTop: 10, border: `1px solid ${C.edge}`, borderRadius: 12, padding: "11px 14px", background: C.panel }}>
            <div style={{ position: "relative" }}>
              <ScheduleBar label="MONO" segs={monoSegs} active={monoIdx} done={ui.done} />
              <div style={{ height: 9 }} />
              <ScheduleBar label="OSC" segs={oscSegs} active={oscIdx} done={ui.done} />
              <div style={{
                position: "absolute", top: -4, bottom: -4, width: 2,
                left: `calc(44px + (100% - 128px) * ${Math.min(1, ui.pos / budget)})`,
                background: C.gold, borderRadius: 1, pointerEvents: "none",
                boxShadow: "0 0 6px rgba(212,169,74,0.55)",
              }}>
                <div style={{ position: "absolute", top: -4, left: -2.5, width: 7, height: 7, borderRadius: 4, background: C.gold }} />
              </div>
            </div>
            <div style={{ marginTop: 11, fontFamily: "ui-monospace, Menlo, monospace", fontSize: 11, color: ui.done ? C.gold : C.dim, letterSpacing: "0.07em" }}>
              {ui.done
                ? `RUN COMPLETE · ${budget.toLocaleString()} PHOTONS`
                : `${ui.pos} / ${budget} PHOTONS · MONO ${monoSegs[monoIdx].name} · OSC ${oscSegs[oscIdx].name}`}
            </div>
          </div>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(186px, 1fr))", gap: 10, marginTop: 12 }}>
          <Stat label="Photons delivered" value={ui.delivered.toLocaleString()} rows={delRows} />
          <Stat label="Mono electrons" value={ui.monoTot.toLocaleString()} sub={`${((ui.monoTot / N) * 100).toFixed(1)}% kept`} tone={C.bright} rows={monoRows} />
          <Stat label="OSC electrons" value={ui.oscTot.toLocaleString()} sub={`${((ui.oscTot / N) * 100).toFixed(1)}% kept`} tone="#8FA6C8" rows={oscRows} />
          <Stat label="Photon ratio = time advantage" value={ratio ? `${ratio.toFixed(2)}×` : "—"} sub={`converges to ${exp.ratio.toFixed(2)}×`} tone={C.gold} />
          <Stat label="SNR advantage" value={snr ? `${snr.toFixed(2)}×` : "—"} sub="= √(photon ratio)" tone={C.gold} />
          {isNB && (
            <Stat label="Continuum and sky rejected" value={ui.contDelivered.toLocaleString()}
              sub="thrown away by both" tone="#7F8FA8" />
          )}
        </div>

        {isNB && (
          <div style={{ marginTop: 10, border: `1px solid ${C.edge}`, borderRadius: 12, padding: "12px 14px", background: C.panel }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, marginBottom: 10, flexWrap: "wrap" }}>
              <div style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 10.5, color: C.dim, letterSpacing: "0.09em" }}>
                PER-LINE ADVANTAGE — WHERE THE REAL STORY IS
              </div>
              <button onClick={() => setCaveats(true)}
                style={{ display: "flex", alignItems: "center", gap: 6, background: "rgba(212,169,74,0.10)", border: "1px solid rgba(212,169,74,0.5)", color: C.gold, borderRadius: 8, padding: "5px 10px", fontSize: 11.5, fontWeight: 700, cursor: "pointer", fontFamily: "inherit" }}>
                <span style={{ fontSize: 13 }}>⚠</span> Read the reservations
              </button>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(238px, 1fr))", gap: 10 }}>
              {M.bands.map((b, i) => {
                const mc = ui.monoBand[i] || 0, oc = ui.oscBand[i] || 0;
                /* measured, like every other readout; the analytic value is
                   shown underneath as the figure it settles on */
                const live = mc > 0 && oc > 0 ? mc / oc : 0;
                const oLost = (ui.oscSlabBand[i] || 0) + (ui.oscDyeBand[i] || 0) + (ui.oscPartBand[i] || 0);
                const monoWins = live >= 1;
                const tgt = exp.oscBand[i] > 0 ? exp.monoBand[i] / exp.oscBand[i] : 0;
                const tgtMono = tgt >= 1;
                return (
                  <div key={i} style={{ border: `1px solid ${C.edge}`, borderRadius: 10, padding: "9px 11px" }}>
                    <div style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 12, fontWeight: 700, color: bandCol(M, i), marginBottom: 4 }}>{b.name}</div>
                    <div style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 18, fontWeight: 600, color: live === 0 ? C.dim : monoWins ? C.bright : "#8FA6C8" }}>
                      {live === 0 ? "—" : `${(monoWins ? live : 1 / live).toFixed(2)}×`}
                    </div>
                    <div style={{ fontSize: 13, fontWeight: 500, color: C.mid, marginTop: 4, letterSpacing: "0.01em" }}>
                      {live === 0 ? "collecting" : monoWins ? "faster on mono" : "faster on the OSC"}
                    </div>
                    <div style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 11.5, color: C.dim, marginTop: 6 }}>
                      converges to {(tgtMono ? tgt : 1 / tgt).toFixed(2)}× {tgtMono ? "mono" : "OSC"}
                    </div>
                    <div style={{ borderTop: `1px solid ${C.edge}`, marginTop: 8, paddingTop: 7, fontSize: 11.5, lineHeight: 1.6 }}>
                      <Fate n={ui.deliveredBand[i] || 0} label={`${b.name} photons delivered`} tone={C.bright} />
                      <div style={{ borderTop: `1px solid ${C.edge}`, margin: "6px 0 7px" }} />
                      <Fate n={mc} label="recorded by mono" tone={C.mid} />
                      {(ui.monoLostBand[i] || 0) > 0 && (
                        <>
                          <Fate n={ui.monoLostBand[i] || 0} label="discarded" tone={C.mid} />
                          {M.bands.map((ob, f) => {
                            const n = (ui.monoLostBy[i] && ui.monoLostBy[i][f]) || 0;
                            return n > 0
                              ? <Fate key={f} n={n} indent label={`${ob.name} filter on the wheel`} />
                              : null;
                          })}
                        </>
                      )}
                      <div style={{ height: 7 }} />
                      <Fate n={oc} label="recorded by the OSC" tone={C.mid} />
                      {oLost > 0 && (
                        <>
                          <Fate n={oLost} label="discarded" tone={C.mid} />
                          {(M.oscSeq || []).map((fset, d) => {
                            const n = (ui.oscSlabBy[i] && ui.oscSlabBy[i][d]) || 0;
                            return n > 0
                              ? <Fate key={d} n={n} indent label={`${fset.map((x) => M.bands[x].name).join("+")} duoband mounted`} />
                              : null;
                          })}
                          {(ui.oscDyeBand[i] || 0) > 0 && (
                            <Fate n={ui.oscDyeBand[i] || 0} indent label="pixel blind to it" />
                          )}
                          {(ui.oscPartBand[i] || 0) > 0 && (
                            <Fate n={ui.oscPartBand[i] || 0} indent label="blue dye absorbed it" />
                          )}
                        </>
                      )}
                    </div>
                    <div style={{ fontSize: 11.5, color: C.dim, lineHeight: 1.5, marginTop: 7 }}>
                      {LINE_WHY[b.name] || ""}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 10, marginTop: 10 }}>
          <Card title="MONO — WHAT IT COLLECTED" sub={`full bar = ${Math.round(barMax).toLocaleString()} electrons`}>
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

          <Card title="OSC — WHAT IT COLLECTED" sub={`full bar = ${Math.round(barMax).toLocaleString()} electrons`}>
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
      {docs && <Docs onClose={() => setDocs(false)} />}
      {caveats && <Caveats onClose={() => setCaveats(false)} />}
    </div>
  );
}

/* ---------------- documentation ---------------- */

const D = { ink: "#1B2230", soft: "#5A6577", rule: "#E3E7EE", chip: "#F4F6F9" };

function Docs({ onClose }) {
  useEffect(() => {
    const k = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);

  const rows = ["lum", "s8", "s4", "s1", "s0", "hoo", "sho"].map((k) => {
    const M = MODES[k], e = expected(k, UNIFORM);
    return { k, label: M.label, nb: M.group === "nb", e, M };
  });

  return (
    <div onClick={onClose}
      style={{ position: "fixed", inset: 0, background: "rgba(4,7,13,0.72)", zIndex: 50, display: "flex", justifyContent: "center", alignItems: "flex-start", padding: "24px 14px", overflowY: "auto", WebkitOverflowScrolling: "touch" }}>
      <div onClick={(e) => e.stopPropagation()}
        style={{ background: "#FFFFFF", color: D.ink, maxWidth: 760, width: "100%", borderRadius: 16, padding: "22px 24px 30px", fontFamily: "ui-sans-serif, -apple-system, 'Helvetica Neue', Arial, sans-serif", lineHeight: 1.62, fontSize: 14.5, boxShadow: "0 24px 60px rgba(0,0,0,0.45)" }}>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, marginBottom: 6 }}>
          <h2 style={{ margin: 0, fontSize: 21, fontWeight: 700, letterSpacing: "-0.01em" }}>How this simulator works</h2>
          <button onClick={onClose}
            style={{ background: D.chip, border: `1px solid ${D.rule}`, borderRadius: 8, padding: "5px 11px", fontSize: 13, fontWeight: 600, cursor: "pointer", color: D.ink, fontFamily: "inherit", flexShrink: 0 }}>
            Close
          </button>
        </div>
        <p style={{ color: D.soft, marginTop: 0, fontSize: 13.5 }}>
          A photon-by-photon account of why a mono camera and a colour camera end
          a night with different numbers of electrons.
        </p>

        <H>The setup</H>
        <P>
          Each panel is a patch of sensor just four pixels across — two by two.
          That is the smallest piece containing a whole Bayer tile, so it is all
          you need to see the effect. On the left is a mono sensor, on the right
          an OSC with the usual RGGB colour filter array.
        </P>
        <P>
          The important part is that <B>both panels receive the same light</B>.
          Every photon is created once, then handed to both cameras at the same
          instant, on the same trajectory, aimed at the same pixel. Any difference
          at the end is the camera, not luck.
        </P>

        <H>How a photon is made</H>
        <P>
          Dice are rolled for each photon: which band it belongs to, which of the
          four pixels it is heading for, and — where the response is partial —
          whether that pixel actually converts it.
        </P>
        <UL items={[
          <>In the LRGB scenarios every photon is broadband, equally likely to be red, green or blue.</>,
          <>In the narrowband scenarios <B>70% of arriving photons are continuum and sky glow</B>, drawn small and plain in red, green and blue. The other 30% are emission line photons, drawn larger with a glow and carrying their letter — H, O or S — until they land.</>,
          <>The target pixel is chosen uniformly, so over a long run each of the four receives a quarter of the light.</>,
        ]} />
        <P>
          The speed control changes only how fast photons arrive. Slow is for
          watching one at a time; Sprint reaches the converged numbers quickly.
          <B> Finish session</B> jumps straight to the end of the run.
        </P>

        <H>The two places a photon can die</H>
        <P>This is the heart of it, and the two cameras do not have the same stages.</P>
        <P>
          <B>The slab</B> is a filter in front of the whole sensor. Mono always
          has one — L, or R, G, B, or a narrowband filter. In narrowband the OSC
          has a slab too, its duoband. A photon whose band the slab does not pass
          stops there and never reaches the sensor. This is where all the
          continuum dies in narrowband, on both cameras equally.
        </P>
        <P>
          <B>The dye</B> is bonded to each individual pixel, and only the OSC has
          one. A photon that clears the slab must still land on a pixel whose dye
          responds to it. Mono has no dye at all, so anything past the slab is
          recorded wherever it falls. That single difference is the whole mono
          advantage.
        </P>

        <H>What each dye responds to</H>
        <P>
          The colour filter array is the same physical object whatever light
          arrives, so it stays RGGB in narrowband too. Response is a probability,
          not a yes or no:
        </P>
        <Table
          head={["Dye", "Hα 656 nm", "OIII 500.7 nm", "SII 672 nm"]}
          body={[
            ["R", "1.0", "0", "1.0"],
            ["G", "0", "1.0", "0"],
            ["B", "0", "0.6", "0"],
            ["array average", "0.25", "0.65", "0.25"],
          ]}
        />
        <P>
          That 0.6 on blue is why an O is sometimes accepted on the blue pixel and
          sometimes not. At 500.7 nm the line sits where the blue dye is already
          closing, so it transmits the photon about six times in ten and absorbs
          it as heat the rest. Averaged over the tile, OIII reaches 65% of the
          array against 25% for Hα and SII — exactly why the OSC beats mono on
          OIII while losing badly on the other two lines.
        </P>
        <P>
          It is worth being precise about which layer does this. A photon reaching
          an OSC pixel passes two gates in series: the <B>dye</B> transmits some
          fraction of what arrives, and of what gets through, the <B>silicon</B>
          converts some fraction into a collected electron. The numbers in the
          table are dye transmission only. Silicon quantum efficiency is left out
          deliberately, because it is common to every pixel and to the mono sensor
          too, so it cancels out of every ratio here. Dye transmission does not
          cancel: it exists only on the OSC, and every photon it absorbs is one
          mono would have kept.
        </P>
        <P>
          So an OIII photon can fail on the OSC in three distinct ways, and the
          per-line panel counts them separately: <B>stopped by the duoband</B>, if
          that filter is not the one currently mounted; <B>pixel blind to it</B>,
          meaning a red pixel with no transmission at 500.7 nm; or <B>absorbed by
          the blue dye</B>, having reached a pixel that does transmit it, but only
          partly. Hα and SII never show the third kind — the dyes are either open
          or shut for them, so it is always a flat yes or no. Mono has only one
          way to lose a photon at all: the wrong filter was on the wheel.
        </P>
        <P>
          On a Bayer array Hα and SII behave identically: both are red light, both
          land only on red pixels. They are not merely slower on an OSC, they are
          <B> inseparable</B>. Only the filter in front can tell them apart, which
          is why SHO on a colour camera needs two duobands.
        </P>

        <H>Filter schedules</H>
        <P>
          Every run has a fixed budget of photons, and each sensor divides that
          budget into blocks of its own. The two bars under the canvas are drawn
          separately because the schedules genuinely differ.
        </P>
        <UL items={[
          <>LRGB runs a single pass: all the luminance first, then R, then G, then B. The OSC never changes filter at all — one unbroken UV/IR cut.</>,
          <>HOO: mono mounts Hα then OIII, half the run each. The OSC keeps one Hα+OIII duoband all night and so collects both lines throughout.</>,
          <>SHO: mono mounts SII, then Hα, then OIII, a third each. The OSC swaps once — OIII+SII for the first half, Hα+OIII for the second — changing filter at a different moment from mono, and collecting OIII under both.</>,
        ]} />

        <H>The slate</H>
        <P>
          Beneath each sensor is a slate with exactly one socket for every photon
          in the run's budget. A photon the sensor never recorded falls there and
          stays, lit in its own colour; a photon that was recorded leaves its
          socket empty. A completely full slate would mean nothing at all was
          captured, so the lit fraction is literally the discard rate.
        </P>

        <H>Where the numbers come from</H>
        <P>
          Under a bright sky the noise is dominated by the sky itself, and signal
          to noise goes as the square root of the photons collected. Two
          consequences, both used in the readouts:
        </P>
        <UL items={[
          <><B>Photon ratio = time advantage.</B> Reaching the same SNR takes the same number of photons, so a camera collecting twice as many gets there in half the time.</>,
          <><B>SNR advantage = √(photon ratio).</B> Twice the photons is 1.41× the signal to noise, not twice.</>,
        ]} />
        <P>
          The per-line panel counts real photons, exactly like the headline
          figures, so it wanders early in a run while the statistics are thin and
          settles as the counts build. The analytic value it is heading for is
          printed underneath each one.
        </P>

        <H>What every scenario converges to</H>
        <Table
          head={["Scenario", "Overall", "Per line"]}
          body={rows.map((r) => [
            r.label,
            r.e.ratio.toFixed(2) + "×",
            r.nb
              ? r.M.bands.map((b, i) => {
                  const q = r.e.oscBand[i] > 0 ? r.e.monoBand[i] / r.e.oscBand[i] : 0;
                  return b.name + " " + (q >= 1 ? q.toFixed(2) + "× mono" : (1 / q).toFixed(2) + "× OSC");
                }).join(",  ")
              : "—",
          ])}
        />
        <P>
          Both narrowband overall figures read 1.11×, which badly understates the
          case. That is an artefact of averaging over an equal mix of lines:
          mono's large wins on Hα and SII are diluted by its loss on OIII. The
          per-line numbers are the ones that matter — and the warning button
          above the per-line panel sets out the rest of the reservations, all of
          which lean the same way.
        </P>

        <H>What is deliberately left out</H>
        <P>
          The model is pure photon bookkeeping. There is no read noise, no dark
          current, no atmosphere and no optical throughput, because none of them
          change the comparison — they apply to both cameras alike.
        </P>
        <P>
          Three real effects <B>are</B> missing, and all three favour the OSC
          here, so the true mono advantage is a little larger than shown: a real
          colour filter array passes only about 85 to 90% of light even inside its
          own passband; demosaicing costs mid-frequency detail, worst on Hα which
          arrives on a grid twice as coarse; and the OSC's synthetic luminance is
          weighted (R+2G+B)/4 rather than flat, which under-serves red targets.
        </P>
        <P>
          One thing that genuinely does not matter: <B>filter bandwidth</B>. A
          3 nm filter collects less sky than a 7 nm one, but by the same factor on
          both cameras, so it cancels exactly out of every ratio here. Narrower
          filters shorten the night for everyone; they do not change who wins.
        </P>

        <div style={{ borderTop: `1px solid ${D.rule}`, marginTop: 22, paddingTop: 14, color: D.soft, fontSize: 12.5 }}>
          Everything above is geometry and counting. Nothing depends on the sensor
          model, the telescope or the sky brightness — those set how long the
          night has to be, not who collects more of it.
        </div>
      </div>
    </div>
  );
}

function Caveats({ onClose }) {
  useEffect(() => {
    const k = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);

  return (
    <div onClick={onClose}
      style={{ position: "fixed", inset: 0, background: "rgba(4,7,13,0.72)", zIndex: 50, display: "flex", justifyContent: "center", alignItems: "flex-start", padding: "24px 14px", overflowY: "auto", WebkitOverflowScrolling: "touch" }}>
      <div onClick={(e) => e.stopPropagation()}
        style={{ background: "#FFFFFF", color: D.ink, maxWidth: 720, width: "100%", borderRadius: 16, padding: "22px 24px 30px", fontFamily: "ui-sans-serif, -apple-system, 'Helvetica Neue', Arial, sans-serif", lineHeight: 1.62, fontSize: 14.5, boxShadow: "0 24px 60px rgba(0,0,0,0.45)" }}>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, marginBottom: 6 }}>
          <h2 style={{ margin: 0, fontSize: 21, fontWeight: 700 }}>Reservations on the narrowband numbers</h2>
          <button onClick={onClose}
            style={{ background: D.chip, border: `1px solid ${D.rule}`, borderRadius: 8, padding: "5px 11px", fontSize: 13, fontWeight: 600, cursor: "pointer", color: D.ink, fontFamily: "inherit", flexShrink: 0 }}>
            Close
          </button>
        </div>

        <div style={{ background: "#FDF6E3", border: "1px solid #E8D5A0", borderRadius: 10, padding: "12px 14px", margin: "10px 0 16px", fontSize: 14 }}>
          <B>This model is deliberately generous to the OSC.</B> Every assumption
          below leans the same way, so the real mono advantage in narrowband is
          larger than what you see here — in some cases considerably. The overall
          1.11× figure in particular is the most pessimistic-for-mono number in
          the whole tool and should not be quoted on its own.
        </div>

        <H>The line mix is equal, and no real target is</H>
        <P>
          By default the run delivers Hα, OIII and SII in equal numbers. That is a
          modelling convenience, not astronomy — so the target selector above the
          canvas lets you swap in the real line ratios of about twenty popular
          objects, and the overall figure moves a long way when you do.
        </P>
        <P>
          Emission nebulae are Hα-dominant and push the advantage toward mono:
          IC 1396 reaches about 2.0× on SHO. Planetary nebulae are the opposite,
          with OIII several times brighter than Hα, and they flip the overall
          verdict to the colour camera — roughly 1.6× the OSC on M57 or M27, and
          1.9× on Ou4 alone, which is essentially a pure OIII object. Per line
          nothing changes at all: mono still wins Hα and SII 2.67× and still loses
          OIII 1.95×. The mix only reweights how those combine, which is exactly
          why the overall number was the wrong one to quote.
        </P>
        <P>
          Mono does not care about the mix — it captures a third of whatever
          arrives. The OSC does care, because its worst channel is Hα. Rerun SHO
          with a 70/20/10 mix and the OSC falls to 0.23 against mono's 0.333, so
          the overall ratio moves from <B>1.11× to about 1.45×</B>. Weight it
          further toward Hα and it keeps climbing.
        </P>

        <H>Mono is made to spend its time evenly, which nobody does</H>
        <P>
          Here mono gives a third of the run to each line. In practice you spend
          more time on the weak lines — 20/30/50 or whatever the target needs. The
          OSC cannot do this at all: its duobands are fixed packages, so asking
          for more SII forces more OIII along with it.
        </P>
        <P>
          That freedom to allocate time unevenly is the entire architectural
          advantage of mono, and this simulation is configured not to use it.
        </P>

        <H>Four real effects sit outside the model</H>
        <UL items={[
          <><B>CFA transmission in-band.</B> A real Bayer dye passes only about 85 to 90% even at the peak of its own passband; here it is a clean 100%. This inconsistency is worth naming: the model charges the OSC for partial transmission at the OIII crossover, where it looks most rigorous, yet gives it a free pass everywhere else — so the OSC is flattered precisely where the physics appears most careful.</>,
          <><B>Demosaicing.</B> It costs mid-frequency detail, and worst on Hα, which arrives on a red grid twice as coarse as the mono one. Not modelled at all.</>,
          <><B>Dye leakage.</B> Real Bayer channels overlap heavily — green has non-zero response at 656 nm, and there is no wavelength where one dye is cleanly open and the others cleanly shut. Separating them needs a correction matrix that amplifies noise. Every response here outside the OIII crossover is an idealised 0 or 1, which is a larger simplification in the LRGB scenarios than in the narrowband ones.</>,
          <><B>Read noise.</B> Each OSC pixel sees only its half of the duoband, so it reaches the sky-limited threshold at longer subs than mono does. No noise of any kind is modelled.</>,
        ]} />

        <H>SII is a capability gap, not a time penalty</H>
        <P>
          On a Bayer array Hα and SII both land only on red pixels and cannot be
          told apart there. The 2.67× shown for SII understates the situation: it
          is not that the OSC is slower, it is that without a second duoband it
          cannot isolate SII at all. No time ratio captures that.
        </P>

        <H>What the model gets right</H>
        <P>
          The geometry is exact. Array fractions of 0.25 for Hα and SII and 0.65
          for OIII follow directly from the RGGB tile and the position of each
          line. So does the duty-cycle arithmetic. Those parts you can trust.
        </P>
        <P>
          And one thing genuinely does not matter: <B>filter bandwidth</B>. A 3 nm
          filter collects less sky than a 7 nm one, but by the same factor on both
          cameras, so it cancels out of every ratio here.
        </P>

        <H>The short version</H>
        <P>
          Mono's advantage is understated for SHO and roughly fair for HOO. The
          genuinely counter-intuitive result — and the one worth repeating — is
          not that mono loses, but that <B>OIII is the single line where a colour
          sensor beats mono outright</B>, because 500.7 nm recruits green and most
          of blue for about 65% of the array against mono's 50% duty cycle.
        </P>
      </div>
    </div>
  );
}

function H({ children }) {
  return <h3 style={{ fontSize: 15.5, fontWeight: 700, margin: "22px 0 6px" }}>{children}</h3>;
}
function P({ children }) {
  return <p style={{ margin: "0 0 10px" }}>{children}</p>;
}
function B({ children }) {
  return <strong style={{ fontWeight: 700 }}>{children}</strong>;
}
function UL({ items }) {
  return (
    <ul style={{ margin: "0 0 10px", paddingLeft: 20 }}>
      {items.map((it, i) => <li key={i} style={{ marginBottom: 5 }}>{it}</li>)}
    </ul>
  );
}
function Table({ head, body }) {
  return (
    <div style={{ overflowX: "auto", margin: "4px 0 12px" }}>
      <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 13.2 }}>
        <thead>
          <tr>
            {head.map((h, i) => (
              <th key={i} style={{ textAlign: i === 0 ? "left" : "center", padding: "6px 10px", borderBottom: `2px solid ${D.rule}`, color: D.soft, fontWeight: 600, whiteSpace: "nowrap" }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((r, i) => (
            <tr key={i}>
              {r.map((c, j) => (
                <td key={j} style={{ textAlign: j === 0 ? "left" : "center", padding: "6px 10px", borderBottom: `1px solid ${D.rule}`, fontFamily: j === 0 ? "inherit" : "ui-monospace, Menlo, monospace", fontWeight: j === 0 ? 600 : 400 }}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------------- UI ---------------- */

function ScheduleBar({ label, segs, active, done }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <span style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 10.5, color: C.dim, letterSpacing: "0.1em", width: 34 }}>{label}</span>
      <div style={{ flex: 1, display: "flex", gap: 3, height: 10 }}>
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
      <span style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 11, color: done ? C.dim : C.mid, width: 74, textAlign: "right" }}>
        {segs[active].name}
      </span>
    </div>
  );
}

function Fate({ n, label, tone, indent }) {
  return (
    <div style={{ display: "flex", gap: 8, alignItems: "baseline", paddingLeft: indent ? 16 : 0 }}>
      <span style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 11.5, color: tone || C.dim, minWidth: 42, textAlign: "right", fontWeight: tone ? 600 : 400 }}>
        {(n || 0).toLocaleString()}
      </span>
      <span style={{ color: tone || C.dim }}>{label}</span>
    </div>
  );
}

function Tag({ children }) {
  return <span style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 9.5, color: C.dim, letterSpacing: "0.1em", textTransform: "uppercase" }}>{children}</span>;
}
function Card({ title, sub, children }) {
  return (
    <div style={{ border: `1px solid ${C.edge}`, borderRadius: 12, padding: "12px 14px", background: C.panel }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 10, flexWrap: "wrap", marginBottom: 10 }}>
        <span style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 10.5, color: C.dim, letterSpacing: "0.09em" }}>{title}</span>
        {sub && <span style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 10, color: "rgba(102,118,143,0.8)" }}>{sub}</span>}
      </div>
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
function Stat({ label, value, sub, tone, rows }) {
  return (
    <div style={{ border: `1px solid ${C.edge}`, borderRadius: 12, padding: "10px 12px", background: C.panel }}>
      <div style={{ fontSize: 10.5, color: C.dim, letterSpacing: "0.07em", textTransform: "uppercase", fontFamily: "ui-monospace, Menlo, monospace", marginBottom: 5, lineHeight: 1.3 }}>{label}</div>
      <div style={{ fontFamily: "ui-monospace, Menlo, monospace", fontSize: 21, fontWeight: 600, color: tone || C.text, lineHeight: 1.1 }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: C.mid, marginTop: 4, fontFamily: "ui-monospace, Menlo, monospace" }}>{sub}</div>}
      {rows && rows.length > 0 && (
        <div style={{ marginTop: 9, borderTop: `1px solid ${C.edge}`, paddingTop: 8, display: "flex", flexDirection: "column", gap: 4 }}>
          {rows.map((r, i) => (
            <div key={i} style={{ display: "flex", alignItems: "baseline", gap: 7, fontFamily: "ui-monospace, Menlo, monospace", fontSize: 11.5 }}>
              <span style={{ color: r.color, fontWeight: 700, minWidth: 34 }}>{r.label}</span>
              <span style={{ color: C.text }}>{(r.value || 0).toLocaleString()}</span>
              {r.pct != null && <span style={{ color: C.dim }}>{r.pct.toFixed(0)}% kept</span>}
            </div>
          ))}
        </div>
      )}
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
/* round up to a readable step, finely enough not to waste bar width */
const NICE = [1, 1.1, 1.2, 1.5, 1.8, 2, 2.5, 3, 4, 5, 6, 8, 10];
function niceCeil(v) {
  if (!isFinite(v) || v <= 0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log10(v)));
  const f = v / mag;
  return Math.round((NICE.find((k) => f <= k + 1e-9) || 10) * mag);
}

function hexRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
