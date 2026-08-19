/** HEVC Sample Adaptive Offset filter (spec 8.7.3).
 *  Reads from the reconstructed planes and writes filtered samples into a
 *  copy, so intra prediction neighbors stay untouched (they already ran).
 */
import { DecodedFrame, CHROMA_420, CHROMA_422 } from '../frame.ts';
import type { SampleArray } from '../frame.ts';
import type { Spt } from './pps.ts';

interface SaoParams {
  typeL: number; offL: Int32Array; eoL: number; bandL: number;
  typeC: number; offC: Int32Array; eoC: number; bandC: number;
  offC2: Int32Array; bandC2: number;
}

function copySamples(data: SampleArray): SampleArray {
  return data instanceof Uint8Array ? new Uint8Array(data) : new Uint16Array(data);
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
  const filterLuma = lumaOn && params.some(value => value.typeL !== 0);
  const filterChroma = chromaOn && chromaFormat !== 0 && params.some(value => value.typeC !== 0);
  const stableLuma = filterLuma && params.some(value => value.typeL === 1);
  const stableChroma = filterChroma && params.some(value => value.typeC === 1);
  // Preserve a stable source only for planes that SAO can actually modify.
  // BO only reads its current sample and is safe in-place; EO planes retain an
  // immutable source because they inspect neighboring samples.
  const dst = frame.planes.map((p, index) =>
    index === 0 ? (stableLuma ? copySamples(p.data) : p.data) :
      (stableChroma ? copySamples(p.data) : p.data));

  for (let ctbY = 0; ctbY < ctbRows; ctbY++) {
    for (let ctbX = 0; ctbX < ctbCols; ctbX++) {
      const ctbAddr = ctbY * ctbCols + ctbX;
      const p = params[ctbAddr]!;
      const x0 = ctbX * ctbSize, y0 = ctbY * ctbSize;

      // availability of neighbors for filtering across CTB edges
      const sid = sliceIdMap?.[((y0 >> 2) * pcmGridW) + (x0 >> 2)];
      const availL = ctbX > 0 && (loopFilterAcrossTiles || !isTileBoundary(ctbX, tileColBd)) &&
        (loopFilterAcrossSlices || !sliceIdMap || sliceIdMap[(y0 >> 2) * pcmGridW + ((x0 - 1) >> 2)] === sid);
      const availR = ctbX < ctbCols - 1 && (loopFilterAcrossTiles || !isTileBoundary(ctbX + 1, tileColBd)) &&
        (loopFilterAcrossSlices || !sliceIdMap || sliceIdMap[(y0 >> 2) * pcmGridW + ((x0 + ctbSize) >> 2)] === sid);
      const availU = ctbY > 0 && (loopFilterAcrossTiles || !isTileBoundary(ctbY, tileRowBd)) &&
        (loopFilterAcrossSlices || !sliceIdMap || sliceIdMap[((y0 - 1) >> 2) * pcmGridW + (x0 >> 2)] === sid);
      const availD = ctbY < ctbRows - 1 && (loopFilterAcrossTiles || !isTileBoundary(ctbY + 1, tileRowBd)) &&
        (loopFilterAcrossSlices || !sliceIdMap || sliceIdMap[((y0 + ctbSize) >> 2) * pcmGridW + (x0 >> 2)] === sid);

      if (lumaOn && p.typeL !== 0) {
        filterPlane(frame.luma.width, frame.luma.height, src[0]!, dst[0]!, frame.luma.stride,
          x0, y0, Math.min(ctbSize, frame.luma.width - x0), Math.min(ctbSize, frame.luma.height - y0),
          p.typeL, p.typeL === 1 ? p.eoL : p.bandL, p.offL, bdL,
          availL, availR, availU, availD, pcmMap, pcmGridW, 0, 0);
      }
      if (chromaOn && chromaFormat !== 0 && p.typeC !== 0) {
        for (let c = 1; c <= 2; c++) {
          const pl = frame.planes[c]!;
          const off = c === 1 ? p.offC : p.offC2;
          filterPlane(pl.width, pl.height, src[c]!, dst[c]!, pl.stride,
            x0 >> hS, y0 >> vS, Math.min(ctbSize >> hS, pl.width - (x0 >> hS)), Math.min(ctbSize >> vS, pl.height - (y0 >> vS)),
            p.typeC, p.typeC === 1 ? p.eoC : c === 1 ? p.bandC : p.bandC2, off, bdC,
            availL, availR, availU, availD, pcmMap, pcmGridW, hS, vS);
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

const EO_OFFSETS = [
  // edge offset classes 0..3: [dx1,dy1, dx2,dy2]
  [1, 0, -1, 0],   // 0: horizontal
  [0, 1, 0, -1],   // 1: vertical
  [1, 1, -1, -1],  // 2: 135 deg
  [1, -1, -1, 1],  // 3: 45 deg
];

function filterPlane(
  w: number, h: number, src: SampleArray, dst: SampleArray, stride: number,
  x0: number, y0: number, bw: number, bh: number,
  type: number, classOrBand: number, offsets: Int32Array, bitDepth: number,
  availL: boolean, availR: boolean, availU: boolean, availD: boolean,
  pcmMap: Uint8Array | null, pcmGridW: number, pcmShiftX: number, pcmShiftY: number,
): void {
  const maximum = (1 << bitDepth) - 1;
  if (type === 2) {
    const shift = bitDepth - 5, startBand = classOrBand;
    for (let y = 0; y < bh; y++) {
      const row = (y0 + y) * stride + x0;
      for (let x = 0; x < bw; x++) {
        const px = x0 + x, py = y0 + y, index = row + x;
        const center = src[index]!;
        if (pcmMap && pcmMap[((py << pcmShiftY) >> 2) * pcmGridW + ((px << pcmShiftX) >> 2)]) {
          dst[index] = center;
          continue;
        }
        let band = (center >> shift) - startBand;
        if (band < 0) band += 32;
        if (band >= 32) band -= 32;
        const output = band < 4 ? center + offsets[band]! : center;
        dst[index] = output < 0 ? 0 : output > maximum ? maximum : output;
      }
    }
    return;
  }

  const [dx1, dy1, dx2, dy2] = EO_OFFSETS[classOrBand]!;
  for (let y = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++) {
      const px = x0 + x, py = y0 + y;
      const idx = py * stride + px;
      const c = src[idx]!;
      if (pcmMap && pcmMap[((py << pcmShiftY) >> 2) * pcmGridW + ((px << pcmShiftX) >> 2)]) {
        dst[idx] = c;
        continue;
      }
      // edge availability
      const n1x = px + dx1, n1y = py + dy1, n2x = px + dx2, n2y = py + dy2;
      let a1: number | null = null, a2: number | null = null;
      if (n1x >= 0 && n1x < w && n1y >= 0 && n1y < h &&
        !(n1x < x0 && !availL) && !(n1x >= x0 + bw && !availR) &&
        !(n1y < y0 && !availU) && !(n1y >= y0 + bh && !availD)) {
        a1 = src[n1y * stride + n1x]!;
      }
      if (n2x >= 0 && n2x < w && n2y >= 0 && n2y < h &&
        !(n2x < x0 && !availL) && !(n2x >= x0 + bw && !availR) &&
        !(n2y < y0 && !availU) && !(n2y >= y0 + bh && !availD)) {
        a2 = src[n2y * stride + n2x]!;
      }
      if (a1 === null || a2 === null) { dst[idx] = c; continue; }
      const first = c - a1, second = c - a2;
      const edge = (first > 0 ? 1 : first < 0 ? -1 : 0) +
        (second > 0 ? 1 : second < 0 ? -1 : 0);
      const offset = edge <= -2 ? offsets[0]! : edge === -1 ? offsets[1]! :
        edge === 0 ? 0 : edge === 1 ? offsets[2]! : offsets[3]!;
      const output = c + offset;
      dst[idx] = output < 0 ? 0 : output > maximum ? maximum : output;
    }
  }
}
