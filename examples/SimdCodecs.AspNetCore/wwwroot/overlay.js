// Standalone in-place OCR overlay. Usage:
//   OcrOverlay.draw(ctx, image, lines, { showOriginal: false, showCharacters: false })
// lines: [{ text, box: [[x,y],...], characters: [{ text, box }] }, ...]
// ctx must be a 2d context (willReadFrequently recommended).
// showOriginal draws the image alone. showCharacters strokes per-character
// quads on the image. Neither keeps the line cover.
(function (global) {
  const FONT_STACK = '"Microsoft YaHei UI", "Segoe UI", sans-serif';

  function parseBox(box) {
    if (!box || box.length !== 4) return null;
    const pts = box.map((p) => [Number(p[0]), Number(p[1])]);
    if (pts.some((p) => !Number.isFinite(p[0]) || !Number.isFinite(p[1]))) return null;
    return pts;
  }

  function dist(a, b) {
    return Math.hypot(b[0] - a[0], b[1] - a[1]);
  }

  // Character quads already follow the recognition direction: point 0 to point 1
  // is the writing edge. Do not swap edges when the box is taller than it is wide.
  function characterGeometry(pts) {
    const along = pts[1];
    const len = dist(pts[0], along);
    const ht = dist(pts[0], pts[3]);
    return {
      cx: (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4,
      cy: (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4,
      len,
      ht,
      angle: Math.atan2(along[1] - pts[0][1], along[0] - pts[0][0]),
    };
  }

  function quadGeometry(pts) {
    let origin = pts[0];
    let along = pts[1];
    let across = pts[3];
    let len = dist(origin, along);
    let ht = dist(origin, across);
    if (ht > len) {
      along = pts[3];
      across = pts[1];
      const swap = len;
      len = ht;
      ht = swap;
    }
    const angle = Math.atan2(along[1] - origin[1], along[0] - origin[0]);
    return {
      pts,
      cx: (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4,
      cy: (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4,
      len,
      ht,
      angle,
    };
  }

  function cross(origin, a, b) {
    return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]);
  }

  function pointInQuad(point, quad) {
    let sign = 0;
    for (let i = 0; i < 4; i++) {
      const value = cross(quad[i], quad[(i + 1) % 4], point);
      if (value === 0) continue;
      const next = Math.sign(value);
      if (sign === 0) sign = next;
      else if (next !== sign) return false;
    }
    return true;
  }

  function luma(r, g, b) {
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }

  function colorDist(a, b) {
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  }

  function median(values) {
    if (!values.length) return 0;
    const sorted = values.slice().sort((a, b) => a - b);
    return sorted[(sorted.length / 2) | 0];
  }

  function cssRgb(color) {
    return `rgb(${color.r | 0},${color.g | 0},${color.b | 0})`;
  }

  function contrastInk(color) {
    return luma(color.r, color.g, color.b) > 140
      ? { r: 17, g: 17, b: 17 }
      : { r: 245, g: 245, b: 247 };
  }

  function quadBounds(ctx, pts, pad) {
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const minX = Math.max(0, Math.floor(Math.min(...xs) - pad));
    const minY = Math.max(0, Math.floor(Math.min(...ys) - pad));
    const maxX = Math.min(ctx.canvas.width, Math.ceil(Math.max(...xs) + pad));
    const maxY = Math.min(ctx.canvas.height, Math.ceil(Math.max(...ys) + pad));
    return { minX, minY, w: Math.max(0, maxX - minX), h: Math.max(0, maxY - minY) };
  }

  function collectInterior(ctx, pts) {
    const bounds = quadBounds(ctx, pts, 1);
    if (bounds.w < 1 || bounds.h < 1) return [];
    const img = ctx.getImageData(bounds.minX, bounds.minY, bounds.w, bounds.h);
    const data = img.data;
    const pixels = [];
    for (let y = 0; y < bounds.h; y++) {
      for (let x = 0; x < bounds.w; x++) {
        if (!pointInQuad([bounds.minX + x + 0.5, bounds.minY + y + 0.5], pts)) continue;
        const i = (y * bounds.w + x) * 4;
        pixels.push([data[i], data[i + 1], data[i + 2]]);
      }
    }
    return pixels;
  }

  function modeBackground(pixels) {
    if (!pixels.length) return { r: 255, g: 255, b: 255 };
    const buckets = new Map();
    for (const [r, g, b] of pixels) {
      const key = ((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4);
      let rec = buckets.get(key);
      if (!rec) {
        rec = { n: 0, r: 0, g: 0, b: 0 };
        buckets.set(key, rec);
      }
      rec.n++;
      rec.r += r;
      rec.g += g;
      rec.b += b;
    }
    let best = null;
    let second = null;
    for (const rec of buckets.values()) {
      if (!best || rec.n > best.n) {
        second = best;
        best = rec;
      } else if (!second || rec.n > second.n) {
        second = rec;
      }
    }
    const toColor = (rec) => ({ r: rec.r / rec.n, g: rec.g / rec.n, b: rec.b / rec.n });
    const bg = toColor(best);
    // Dense black text can win the mode. Prefer the runner-up if it is clearly a paper/highlight fill.
    if (second && best.n < pixels.length * 0.55 && luma(bg.r, bg.g, bg.b) < 80
      && luma(second.r / second.n, second.g / second.n, second.b / second.n) > 140) {
      return toColor(second);
    }
    return bg;
  }

  function pickInkColor(pixels, background) {
    const cores = [];
    for (const pixel of pixels) {
      const gap = colorDist(pixel, [background.r, background.g, background.b]);
      if (gap >= 16) cores.push({ r: pixel[0], g: pixel[1], b: pixel[2], d: gap });
    }
    if (!cores.length) return null;
    const cutoff = median(cores.map((c) => c.d));
    const strong = cores.filter((c) => c.d >= cutoff);
    const use = strong.length ? strong : cores;
    return {
      r: median(use.map((c) => c.r)),
      g: median(use.map((c) => c.g)),
      b: median(use.map((c) => c.b)),
    };
  }

  function traceQuad(ctx, pts) {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < 4; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
  }

  function fillQuad(ctx, pts, color) {
    traceQuad(ctx, pts);
    ctx.fillStyle = cssRgb(color);
    ctx.fill();
  }

  function strokeQuad(ctx, pts, color, width) {
    traceQuad(ctx, pts);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  }

  // One size for the whole line, from the interior boxes. End boxes are narrower
  // because outer blanks stay outside them; fitting each box would shrink the ends.
  function sharedCharacterSize(ctx, glyphs) {
    const body = glyphs.filter((g) => !g.blank);
    const interior = body.length >= 3 ? body.slice(1, -1) : body;
    const use = interior.length ? interior : body;
    const maxW = median(use.map((g) => g.geom.len)) || median(glyphs.map((g) => g.geom.len));
    const maxH = median(use.map((g) => g.geom.ht)) || median(glyphs.map((g) => g.geom.ht));
    const cjk = use.find((g) => /[\u3400-\u9fff]/.test(g.text));
    const sample = cjk ? [...cjk.text][0] : (use.find((g) => g.text.length === 1)?.text || "字");
    return fitFontSize(ctx, sample, Math.max(maxW, 1) * 0.92, Math.max(maxH, 1) * 0.9);
  }

  function fitFontSize(ctx, text, maxW, maxH) {
    let lo = 6;
    let hi = Math.max(8, maxH * 0.82);
    let best = 6;
    while (hi - lo > 0.5) {
      const mid = (lo + hi) / 2;
      ctx.font = `bold ${mid}px ${FONT_STACK}`;
      const metrics = ctx.measureText(text);
      const width = metrics.width;
      const height = (metrics.actualBoundingBoxAscent ?? mid * 0.8)
        + (metrics.actualBoundingBoxDescent ?? mid * 0.2);
      if (width <= maxW && height <= maxH) {
        best = mid;
        lo = mid;
      } else {
        hi = mid;
      }
    }
    return best;
  }

  function drawLine(ctx, line) {
    const pts = parseBox(line.box);
    if (!pts) return;
    const geom = quadGeometry(pts);
    if (geom.len < 2 || geom.ht < 2) return;

    const pixels = collectInterior(ctx, pts);
    const background = modeBackground(pixels);
    const ink = pickInkColor(pixels, background);
    fillQuad(ctx, pts, background);

    const text = line.text ?? "";
    if (!text) return;
    const size = fitFontSize(ctx, text, geom.len * 0.94, geom.ht * 0.9);
    ctx.save();
    ctx.translate(geom.cx, geom.cy);
    ctx.rotate(geom.angle);
    ctx.font = `bold ${size}px ${FONT_STACK}`;
    ctx.fillStyle = cssRgb(ink ?? contrastInk(background));
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, 0, 0);
    ctx.restore();
  }

  function drawCharacters(ctx, lines) {
    // Sample while the canvas is still the photo. The line quad is filled
    // first so original strokes outside a narrow character box are covered.
    // A character whose own background differs (a highlight) is filled again.
    const jobs = [];
    for (const line of lines) {
      const characters = line.characters;
      if (!characters?.length) continue;
      const linePts = parseBox(line.box);
      const lineBackground = linePts ? modeBackground(collectInterior(ctx, linePts)) : null;
      const glyphs = [];
      for (const character of characters) {
        const pts = parseBox(character.box);
        if (!pts) continue;
        const geom = characterGeometry(pts);
        if (geom.len < 1 || geom.ht < 1) continue;
        const text = character.text ?? "";
        const pixels = collectInterior(ctx, pts);
        const background = modeBackground(pixels);
        const ink = pickInkColor(pixels, background) ?? contrastInk(background);
        glyphs.push({ pts, geom, text, blank: !text.trim(), background, ink });
      }
      jobs.push({ linePts, lineBackground, glyphs, size: sharedCharacterSize(ctx, glyphs) });
    }
    for (const job of jobs) {
      if (job.linePts && job.lineBackground) fillQuad(ctx, job.linePts, job.lineBackground);
      for (const glyph of job.glyphs) {
        if (job.lineBackground && colorDist(
          [glyph.background.r, glyph.background.g, glyph.background.b],
          [job.lineBackground.r, job.lineBackground.g, job.lineBackground.b]) >= 24) {
          fillQuad(ctx, glyph.pts, glyph.background);
        }
        if (glyph.blank) continue;
        ctx.save();
        ctx.translate(glyph.geom.cx, glyph.geom.cy);
        ctx.rotate(glyph.geom.angle);
        ctx.font = `bold ${job.size}px ${FONT_STACK}`;
        ctx.fillStyle = cssRgb(glyph.ink);
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(glyph.text, 0, 0);
        ctx.restore();
      }
    }
    for (const job of jobs) {
      for (const glyph of job.glyphs) strokeQuad(ctx, glyph.pts, "#e10600", 1);
    }
  }

  function draw(ctx, image, lines, options) {
    if (!ctx || !image) return;
    const showOriginal = options?.showOriginal === true;
    const showCharacters = options?.showCharacters === true;
    ctx.canvas.width = image.naturalWidth || image.width;
    ctx.canvas.height = image.naturalHeight || image.height;
    ctx.drawImage(image, 0, 0);
    if (showOriginal || !lines?.length) return;
    if (showCharacters) {
      drawCharacters(ctx, lines);
      return;
    }
    for (const line of lines) drawLine(ctx, line);
  }

  global.OcrOverlay = { draw };
})(typeof window !== "undefined" ? window : globalThis);
