// === Config ===
const THRESHOLD = 11; // <- put the value you found on validation
const INDEX_URL = chrome.runtime.getURL("assets/phash_index_train.json");

// === Helpers ===
const popcount64 = (x) => {
  // x is BigInt
  let c = 0n;
  while (x) { x &= (x - 1n); c++; }
  return Number(c);
};

// Convert 2D array to DCT (float) — small, naïve DCT-II for pHash
function dct2(matrix) {
  const N = matrix.length; // square
  const out = Array.from({ length: N }, () => new Array(N).fill(0));
  const alpha = (k) => (k === 0 ? Math.SQRT1_2 : 1); // normalization
  for (let u = 0; u < N; u++) {
    for (let v = 0; v < N; v++) {
      let sum = 0;
      for (let x = 0; x < N; x++) {
        for (let y = 0; y < N; y++) {
          sum += matrix[x][y] *
                 Math.cos(((2*x+1)*u*Math.PI)/(2*N)) *
                 Math.cos(((2*y+1)*v*Math.PI)/(2*N));
        }
      }
      out[u][v] = 0.25 * alpha(u) * alpha(v) * sum;
    }
  }
  return out;
}

// Compute 64-bit pHash (8x8 from top-left of DCT after resize to 32x32 grayscale)
async function computePHashFromDataURL(dataURL) {
  const img = new Image();
  img.src = dataURL;
  await img.decode();

  const CANVAS_N = 32; // 32x32
  const canvas = new OffscreenCanvas(CANVAS_N, CANVAS_N);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  // Draw and grayscale
  ctx.drawImage(img, 0, 0, CANVAS_N, CANVAS_N);
  const { data } = ctx.getImageData(0, 0, CANVAS_N, CANVAS_N);
  const gray = [];
  for (let y = 0; y < CANVAS_N; y++) {
    const row = [];
    for (let x = 0; x < CANVAS_N; x++) {
      const i = (y * CANVAS_N + x) * 4;
      const r = data[i], g = data[i+1], b = data[i+2];
      row.push(0.299*r + 0.587*g + 0.114*b);
    }
    gray.push(row);
  }

  const dct = dct2(gray);

  // take 8x8 block from top-left; ignore DC at [0][0] when computing median
  const block = [];
  for (let u = 0; u < 8; u++) {
    for (let v = 0; v < 8; v++) {
      block.push(dct[u][v]);
    }
  }
  const withoutDC = block.slice(1);
  const sorted = withoutDC.slice().sort((a,b)=>a-b);
  const med = sorted[Math.floor(sorted.length/2)];

  // build 64-bit hash (bit=1 if coeff > median)
  let h = 0n;
  for (let i = 0; i < 64; i++) {
    const bit = (block[i] > med) ? 1n : 0n;
    h = (h << 1n) | bit;
  }
  return h;
}

// Nearest neighbor over index entries with {hash,label}
function nearestNeighbor(index, h) {
  let bestD = 1e9, bestLabel = null;
  for (const item of index) {
    // item.hash can be number; cast to BigInt
    const ih = BigInt(item.hash);
    const d = popcount64(ih ^ h);
    if (d < bestD) {
      bestD = d;
      bestLabel = item.label; // assume 1=phish, 0=legit OR "phishing"/"legitimate" strings
      if (bestD <= 2) break;
    }
  }
  return { bestD, bestLabel };
}

// Normalize label to "phishing"/"legitimate"
function normLabel(lbl) {
  if (typeof lbl === "number") return lbl === 1 ? "phishing" : "legitimate";
  const s = String(lbl).toLowerCase();
  return (s.includes("phish")) ? "phishing" : "legitimate";
}

// === Main click flow ===
async function onScan() {
  const resultEl = document.getElementById("result");
  const detailsEl = document.getElementById("details");
  resultEl.textContent = "Scanning…";
  resultEl.className = "muted";
  detailsEl.textContent = "";

  try {
    // 1) Load index (once per popup open)
    const index = await fetch(INDEX_URL).then(r => r.json());

    // 2) Capture current tab
    const dataURL = await chrome.tabs.captureVisibleTab(undefined, { format: "png" });

    // 3) Compute page pHash
    const h = await computePHashFromDataURL(dataURL);

    // 4) NN match
    const { bestD, bestLabel } = nearestNeighbor(index, h);
    const nearest = normLabel(bestLabel);

    // 5) Decision
    const isPhish = (nearest === "phishing") && (bestD <= THRESHOLD);

    resultEl.innerHTML = isPhish
      ? `<span class="badge phish">⚠️ Phishing suspected</span>`
      : `<span class="badge safe">✅ Looks safe</span>`;

    detailsEl.textContent = `Nearest: ${nearest} | distance=${bestD} | threshold=${THRESHOLD}`;

  } catch (e) {
    resultEl.textContent = "Error while scanning.";
    detailsEl.textContent = String(e);
  }
}

document.getElementById("scanBtn").addEventListener("click", onScan);
