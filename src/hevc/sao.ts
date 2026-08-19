/** HEVC Sample Adaptive Offset filter (spec 8.7.3).
 *  Reads from the reconstructed planes and writes filtered samples into a
 *  copy, so intra prediction neighbors stay untouched (they already ran).
 */
import { DecodedFrame, CHROMA_420, CHROMA_422 } from '../frame.ts';
import type { Spt } from './pps.ts';

interface SaoParams {
  typeL: number; offL: Int32Array; eoL: number; bandL: number;
  typeC: number; offC: Int32Array; eoC: number; bandC: number;
  offC2: Int32Array; bandC2: number;
}

export function applySao(
  frame: DecodedFrame, params: SaoParams[],
  tileColBd: number[], tileRowBd: number[], ctbCols: number, ctbRows: number,
  sps: Spt, lumaOn: boolean, chromaOn: boolean, loopFilterAcrossTiles: boolean,
  pcmMap: Uint8Array | null = null, pcmGridW = 0,
  sliceIdMap: Int32Array | null = null, loopFilterAcrossSlices = true,
): void {
  const bdL = sps.bitDepthLuma;
  const bdC = sps.bitDepthChroma;
  const ctbLog2 = sps.log2CtbSize;
  const ctbSize = 1 << ctbLog2;
  const chromaFormat = sps.separateColourPlane ? 0 : sps.chromaFormatIdc;
  const hS = chromaFormat === 1 || chromaFormat === 2 ? 1 : 0;
  const vS = chromaFormat === 1 ? 1 : 0;

  const src = frame.planes.map(p => p.data);
  const dst = frame.planes.map(p => new Uint16Array(p.data));

  for (let ctbY = 0; ctbY < ctbRows; ctbY++) {
    for (let ctbX = 0; ctbX < ctbCols; ctbX++) {
      const ctbAddr = ctbY * ctbCols + ctbX;
      const p = params[ctbAddr]!;
      const x0 = ctbX * ctbSize, y0 = ctbY * ctbSize;

      // availability of neighbors for filtering across CTB edges
      const sid = sliceIdMap?.[((y0 >> 2) * pcmGridW) + (x0 >> 2)];
      const sameSlice = (x: number, y: number) => !sliceIdMap || sliceIdMap[(y >> 2) * pcmGridW + (x >> 2)] === sid;
      const availL = ctbX > 0 && (loopFilterAcrossTiles || !isTileBoundary(ctbX, tileColBd)) &&
        (loopFilterAcrossSlices || sameSlice(x0 - 1, y0));
      const availR = ctbX < ctbCols - 1 && (loopFilterAcrossTiles || !isTileBoundary(ctbX + 1, tileColBd)) &&
        (loopFilterAcrossSlices || sameSlice(x0 + ctbSize, y0));
      const availU = ctbY > 0 && (loopFilterAcrossTiles || !isTileBoundary(ctbY, tileRowBd)) &&
        (loopFilterAcrossSlices || sameSlice(x0, y0 - 1));
      const availD = ctbY < ctbRows - 1 && (loopFilterAcrossTiles || !isTileBoundary(ctbY + 1, tileRowBd)) &&
        (loopFilterAcrossSlices || sameSlice(x0, y0 + ctbSize));

      if (lumaOn && p.typeL !== 0) {
        const skipPcm = pcmMap ? (x: number, y: number) => !!pcmMap[(y >> 2) * pcmGridW + (x >> 2)] : undefined;
        filterPlane(frame.luma.width, frame.luma.height, src[0]!, dst[0]!, frame.luma.stride,
          x0, y0, Math.min(ctbSize, frame.luma.width - x0), Math.min(ctbSize, frame.luma.height - y0),
          p.typeL === 1 ? { kind: 'eo', cls: p.eoL, off: p.offL } : { kind: 'bo', band: p.bandL, off: p.offL },
          bdL, { l: availL, r: availR, u: availU, d: availD }, skipPcm);
      }
      if (chromaOn && chromaFormat !== 0 && p.typeC !== 0) {
        for (let c = 1; c <= 2; c++) {
          const pl = frame.planes[c]!;
          const off = c === 1 ? p.offC : p.offC2;
          const skipPcm = pcmMap ? (x: number, y: number) => !!pcmMap[((y << vS) >> 2) * pcmGridW + ((x << hS) >> 2)] : undefined;
          filterPlane(pl.width, pl.height, src[c]!, dst[c]!, pl.stride,
            x0 >> hS, y0 >> vS, Math.min(ctbSize >> hS, pl.width - (x0 >> hS)), Math.min(ctbSize >> vS, pl.height - (y0 >> vS)),
            p.typeC === 1 ? { kind: 'eo', cls: p.eoC, off } : { kind: 'bo', band: c === 1 ? p.bandC : p.bandC2, off },
            bdC, { l: availL, r: availR, u: availU, d: availD }, skipPcm);
        }
      }
    }
  }
  // swap filtered data back into the frame
  for (let i = 0; i < frame.planes.length; i++) frame.planes[i]!.data = dst[i]!;
}

function isTileBoundary(position: number, bd: number[]): boolean {
  for (let i = 1; i < bd.length - 1; i++) if (bd[i] === position) return true;
  return false;
}

interface Avail { l: boolean; r: boolean; u: boolean; d: boolean }

const EO_OFFSETS = [
  // edge offset classes 0..3: [dx1,dy1, dx2,dy2]
  [1, 0, -1, 0],   // 0: horizontal
  [0, 1, 0, -1],   // 1: vertical
  [1, 1, -1, -1],  // 2: 135 deg
  [1, -1, -1, 1],  // 3: 45 deg
];

function filterPlane(
  w: number, h: number, src: Uint16Array, dst: Uint16Array, stride: number,
  x0: number, y0: number, bw: number, bh: number,
  type: { kind: 'eo'; cls: number; off: Int32Array } | { kind: 'bo'; band: number; off: Int32Array },
  bitDepth: number, avail: Avail, skip?: (x: number, y: number) => boolean,
): void {
  for (let y = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++) {
      const px = x0 + x, py = y0 + y;
      const idx = py * stride + px;
      const c = src[idx]!;
      if (skip?.(px, py)) { dst[idx] = c; continue; }
      if (type.kind === 'bo') {
        const band = (c >> (bitDepth - 5)) - type.band;
        let bandIdx = band;
        if (bandIdx < 0) bandIdx += 32;
        if (bandIdx >= 32) bandIdx -= 32;
        let out = c;
        if (bandIdx < 4) {
          out = c + type.off[bandIdx]!;
        }
        dst[idx] = out < 0 ? 0 : out >= (1 << bitDepth) ? (1 << bitDepth) - 1 : out;
      } else {
        const [dx1, dy1, dx2, dy2] = EO_OFFSETS[type.cls]!;
        // edge availability
        const n1x = px + dx1, n1y = py + dy1, n2x = px + dx2, n2y = py + dy2;
        let a1: number | null = null, a2: number | null = null;
        if (n1x >= 0 && n1x < w && n1y >= 0 && n1y < h &&
          !(n1x < x0 && !avail.l) && !(n1x >= x0 + bw && !avail.r) &&
          !(n1y < y0 && !avail.u) && !(n1y >= y0 + bh && !avail.d)) {
          a1 = src[n1y * stride + n1x]!;
        }
        if (n2x >= 0 && n2x < w && n2y >= 0 && n2y < h &&
          !(n2x < x0 && !avail.l) && !(n2x >= x0 + bw && !avail.r) &&
          !(n2y < y0 && !avail.u) && !(n2y >= y0 + bh && !avail.d)) {
          a2 = src[n2y * stride + n2x]!;
        }
        if (a1 === null || a2 === null) {
          dst[idx] = c;
          continue;
        }
        const sgn = (x: number) => x > 0 ? 1 : x < 0 ? -1 : 0;
        const edgeIdx = sgn(c - a1) + sgn(c - a2); // -2..2
        // mapping: -2 -> off[0], -1 -> off[1], 0 -> none, +1 -> off[2], +2 -> off[3]
        let off: number;
        if (edgeIdx <= -2) off = type.off[0]!;
        else if (edgeIdx === -1) off = type.off[1]!;
        else if (edgeIdx === 0) off = 0;
        else if (edgeIdx === 1) off = type.off[2]!;
        else off = type.off[3]!;
        const out = c + off;
        dst[idx] = out < 0 ? 0 : out >= (1 << bitDepth) ? (1 << bitDepth) - 1 : out;
      }
    }
  }
}
