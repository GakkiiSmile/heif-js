/** HEVC intra prediction (spec 8.4.4): border collection, substitution,
 *  filtering, and the 35 prediction modes. */
import { INTRA_ANGLE, INV_ANGLE } from './tables.ts';
import { Plane } from '../frame.ts';
import { debugWrite } from '../debug.ts';

export interface IntraScratch {
  border: Int32Array;
  available: Uint8Array;
  filtered: Int32Array;
  reference: Int32Array;
}

export function createIntraScratch(): IntraScratch {
  return {
    border: new Int32Array(2 * 64 + 1),
    available: new Uint8Array(2 * 64 + 1),
    filtered: new Int32Array(2 * 64 + 1),
    reference: new Int32Array(4 * 32 + 1),
  };
}

export interface IntraCtx {
  planes: Plane[];
  chromaArrayType: number;
  strongIntraSmoothing: boolean;
  intraSmoothingDisabled: boolean;
  bitDepth: number;
  /** Z-scan address per 4x4 block (for neighbor availability); null = all decoded (single pass) */
  minTbAddrZs: Int32Array | null;
  picWidthInTbs: number;
  log2MinTb: number;
  sliceIdMap?: Int32Array;
  currentSliceId?: number;
  tileIdMap?: Int32Array;
  currentTileId?: number;
  scratch?: IntraScratch;
  tuDebug?: boolean;
}

const clipBd = (v: number, bd: number) => v < 0 ? 0 : v >= (1 << bd) ? (1 << bd) - 1 : v;

/**
 * Compute intra prediction into dst.
 * xB/yB are component-sample coordinates of the top-left of the block.
 */
export function intraPredict(
  ctx: IntraCtx, cIdx: number, xB: number, yB: number,
  predMode: number, nT: number, disableBoundaryFilter: boolean,
): void {
  const bd = ctx.bitDepth;
  const plane = ctx.planes[cIdx]!;
  const log2MinTb = ctx.log2MinTb;

  // border[0] = top-left; border[-1-y] = left; border[1+x] = top / top-right
  const scratch = ctx.scratch ?? createIntraScratch();
  const border = scratch.border;
  const off = 64;
  const avail = scratch.available;
  avail.fill(0, off - 2 * nT, off + 2 * nT + 1);
  const stride = plane.stride;
  const W = plane.width, H = plane.height;

  // availability in 4x4 grid coordinates of this component
  const SubW = cIdx === 0 || ctx.chromaArrayType === 3 ? 1 : 2;
  const SubH = cIdx === 0 || ctx.chromaArrayType === 2 || ctx.chromaArrayType === 3 ? 1 : 2;
  const subWShift = SubW - 1, subHShift = SubH - 1;
  const minTbAddrZs = ctx.minTbAddrZs;
  const picWidthInTbs = ctx.picWidthInTbs;
  const currTbX = (xB << subWShift) >> log2MinTb;
  const currTbY = (yB << subHShift) >> log2MinTb;
  const currZ = minTbAddrZs
    ? minTbAddrZs[currTbX + currTbY * picWidthInTbs]!
    : -1;
  const sliceIdMap = ctx.sliceIdMap, tileIdMap = ctx.tileIdMap;
  const currentSliceId = ctx.currentSliceId, currentTileId = ctx.currentTileId;
  const data = plane.data;

  let nAvail = 0;
  let firstValue = 0;

  // left column: border[-1-y] for y = nT-1 .. 0 collected bottom-up (order matters for firstValue)
  const leftX = xB - 1;
  if (leftX >= 0) {
    const tbX = (leftX << subWShift) >> log2MinTb;
    let previousGi = -1, usable = false;
    for (let y = 2 * nT - 1; y >= 0; y--) {
      const sampleY = yB + y;
      if (sampleY >= H) continue;
      const gi = tbX + (((sampleY << subHShift) >> log2MinTb) * picWidthInTbs);
      if (gi !== previousGi) {
        previousGi = gi;
        usable = (!sliceIdMap || sliceIdMap[gi] === currentSliceId) &&
          (!tileIdMap || tileIdMap[gi] === currentTileId) &&
          (!minTbAddrZs || minTbAddrZs[gi]! <= currZ);
      }
      if (!usable) continue;
      const value = data[leftX + sampleY * stride]!;
      if (nAvail === 0) firstValue = value;
      avail[off - 1 - y] = 1;
      border[off - 1 - y] = value;
      nAvail++;
    }
  }
  const topY = yB - 1;
  if (leftX >= 0 && topY >= 0) {
    const gi = ((leftX << subWShift) >> log2MinTb) +
      (((topY << subHShift) >> log2MinTb) * picWidthInTbs);
    if ((!sliceIdMap || sliceIdMap[gi] === currentSliceId) &&
        (!tileIdMap || tileIdMap[gi] === currentTileId) &&
        (!minTbAddrZs || minTbAddrZs[gi]! <= currZ)) {
      const value = data[leftX + topY * stride]!;
      if (nAvail === 0) firstValue = value;
      avail[off] = 1;
      border[off] = value;
      nAvail++;
    }
  }
  if (topY >= 0) {
    const tbY = ((topY << subHShift) >> log2MinTb) * picWidthInTbs;
    let previousGi = -1, usable = false;
    for (let x = 0; x < 2 * nT; x++) {
      const sampleX = xB + x;
      if (sampleX >= W) break;
      const gi = ((sampleX << subWShift) >> log2MinTb) + tbY;
      if (gi !== previousGi) {
        previousGi = gi;
        usable = (!sliceIdMap || sliceIdMap[gi] === currentSliceId) &&
          (!tileIdMap || tileIdMap[gi] === currentTileId) &&
          (!minTbAddrZs || minTbAddrZs[gi]! <= currZ);
      }
      if (!usable) continue;
      const value = data[sampleX + topY * stride]!;
      if (nAvail === 0) firstValue = value;
      avail[off + 1 + x] = 1;
      border[off + 1 + x] = value;
      nAvail++;
    }
  }

  // ---- reference sample substitution (8.4.4.2.2)
  if (ctx.tuDebug && xB === 0 && yB === 0) {
    debugWrite(`PREDDBG nAvail=${nAvail} first=${firstValue} bM1=${border[off - 1]} b0=${border[off]} predMode=${predMode} nT=${nT}\n`);
  }
  if (nAvail !== 4 * nT + 1) {
    if (nAvail === 0) {
      for (let i = -2 * nT; i <= 2 * nT; i++) border[off + i] = 1 << (bd - 1);
    } else {
      if (!avail[off - 2 * nT]) border[off - 2 * nT] = firstValue;
      for (let i = -2 * nT + 1; i <= 2 * nT; i++) {
        if (!avail[off + i]) border[off + i] = border[off + i - 1];
      }
    }
  }

  if (ctx.tuDebug && xB === 0 && yB === 0) {
    debugWrite(`PREDDBG2 bd=${bd} bM1=${border[off - 1]} b0=${border[off]} b33=${border[off + 33]}\n`);
  }

  // ---- reference sample filtering (8.4.4.2.3)
  if (!ctx.intraSmoothingDisabled && (cIdx === 0 || ctx.chromaArrayType === 3)) {
    let filterFlag = 0;
    if (predMode !== 1 && nT !== 4) { // not DC and not 4x4
      const minDistVerHor = Math.min(Math.abs(predMode - 26), Math.abs(predMode - 10));
      if (nT === 8) filterFlag = minDistVerHor > 7 ? 1 : 0;
      else if (nT === 16) filterFlag = minDistVerHor > 1 ? 1 : 0;
      else if (nT === 32) filterFlag = minDistVerHor > 0 ? 1 : 0;
    }
    if (filterFlag) {
      const biInt = ctx.strongIntraSmoothing && cIdx === 0 && nT === 32 &&
        Math.abs(border[off]! + border[off + 64]! - 2 * border[off + 32]!) < (1 << (bd - 5)) &&
        Math.abs(border[off]! + border[off - 64]! - 2 * border[off - 32]!) < (1 << (bd - 5)) ? 1 : 0;
      const pF = scratch.filtered;
      if (biInt) {
        const center = border[off]!, negative = border[off - 64]!, positive = border[off + 64]!;
        pF[off - 2 * nT] = border[off - 2 * nT]!;
        pF[off + 2 * nT] = border[off + 2 * nT]!;
        pF[off] = center;
        for (let i = 1; i <= 63; i++) {
          pF[off - i] = center + ((i * (negative - center) + 32) >> 6);
          pF[off + i] = center + ((i * (positive - center) + 32) >> 6);
        }
      } else {
        pF[off - 2 * nT] = border[off - 2 * nT]!;
        pF[off + 2 * nT] = border[off + 2 * nT]!;
        for (let i = -(2 * nT - 1); i <= 2 * nT - 1; i++) {
          pF[off + i] = (border[off + i + 1]! + 2 * border[off + i]! + border[off + i - 1]! + 2) >> 2;
        }
      }
      for (let i = -2 * nT; i <= 2 * nT; i++) border[off + i] = pF[off + i]!;
    }
  }

  const dst = plane.data;
  const dstride = stride;

  if (predMode === 0) {
    // planar
    const log2nT = Math.log2(nT);
    for (let y = 0; y < nT; y++) {
      for (let x = 0; x < nT; x++) {
        dst[xB + x + (yB + y) * dstride] =
          ((nT - 1 - x) * border[off - 1 - y]! + (x + 1) * border[off + 1 + nT]! +
            (nT - 1 - y) * border[off + 1 + x]! + (y + 1) * border[off - 1 - nT]! + nT) >> (log2nT + 1);
      }
    }
    return;
  }
  if (predMode === 1) {
    // DC
    let sum = 0;
    for (let i = 0; i < nT; i++) { sum += border[off + 1 + i]! + border[off - 1 - i]!; }
    const dc = (sum + nT) >> (Math.log2(nT) + 1);
    for (let y = 0; y < nT; y++) for (let x = 0; x < nT; x++) dst[xB + x + (yB + y) * dstride] = dc;
    // boundary filter (cIdx 0, nT < 32)
    if (cIdx === 0 && nT < 32) {
      dst[xB + yB * dstride] = (border[off - 1]! + 2 * dc + border[off + 1]! + 2) >> 2;
      for (let x = 1; x < nT; x++) {
        dst[xB + x + yB * dstride] = (border[off + 1 + x]! + 3 * dc + 2) >> 2;
      }
      for (let y = 1; y < nT; y++) {
        dst[xB + (yB + y) * dstride] = (border[off - 1 - y]! + 3 * dc + 2) >> 2;
      }
    }
    return;
  }

  // angular
  const angle = INTRA_ANGLE[predMode]!;
  const ref = scratch.reference;
  const roff = 64;
  // Preserve the zero-initialized semantics of the former per-call buffer for
  // any reference positions not populated by a particular angular mode.
  ref.fill(0, roff - 2 * nT, roff + 2 * nT + 1);
  if (predMode >= 18) {
    for (let x = 0; x <= nT; x++) ref[roff + x] = border[off + x]!;
    if (angle < 0) {
      const inv = INV_ANGLE[predMode - 11]!;
      if ((nT * angle) >> 5 < -1) {
        for (let x = (nT * angle) >> 5; x <= -1; x++) {
          ref[roff + x] = border[off - ((x * inv + 128) >> 8)]!;
        }
      }
    } else {
      for (let x = nT + 1; x <= 2 * nT; x++) ref[roff + x] = border[off + x]!;
    }
    for (let y = 0; y < nT; y++) {
      const base = (y + 1) * angle;
      const iIdx = base >> 5, iFact = base & 31;
      for (let x = 0; x < nT; x++) {
        dst[xB + x + (yB + y) * dstride] = iFact !== 0
          ? ((32 - iFact) * ref[roff + x + iIdx + 1]! + iFact * ref[roff + x + iIdx + 2]! + 16) >> 5
          : ref[roff + x + iIdx + 1]!;
      }
    }
    if (predMode === 26 && cIdx === 0 && nT < 32 && !disableBoundaryFilter) {
      for (let y = 0; y < nT; y++) {
        dst[xB + (yB + y) * dstride] = clipBd(
          border[off + 1]! + ((border[off - 1 - y]! - border[off]!) >> 1), bd,
        );
      }
    }
  } else {
    for (let x = 0; x <= nT; x++) ref[roff + x] = border[off - x]!;
    if (angle < 0) {
      const inv = INV_ANGLE[predMode - 11]!;
      if ((nT * angle) >> 5 < -1) {
        for (let x = (nT * angle) >> 5; x <= -1; x++) {
          ref[roff + x] = border[off + ((x * inv + 128) >> 8)]!;
        }
      }
    } else {
      for (let x = nT + 1; x <= 2 * nT; x++) ref[roff + x] = border[off - x]!;
    }
    for (let y = 0; y < nT; y++) {
      for (let x = 0; x < nT; x++) {
        const base = (x + 1) * angle;
        const iIdx = base >> 5, iFact = base & 31;
        dst[xB + x + (yB + y) * dstride] = iFact !== 0
          ? ((32 - iFact) * ref[roff + y + iIdx + 1]! + iFact * ref[roff + y + iIdx + 2]! + 16) >> 5
          : ref[roff + y + iIdx + 1]!;
      }
    }
    if (predMode === 10 && cIdx === 0 && nT < 32 && !disableBoundaryFilter) {
      for (let x = 0; x < nT; x++) {
        dst[xB + x + yB * dstride] = clipBd(
          border[off - 1]! + ((border[off + 1 + x]! - border[off]!) >> 1), bd,
        );
      }
    }
  }
}

/** MPM candidate derivation (spec 8.4.3). Unavailable neighbors use DC (1). */
export function fillIntraPredModeCandidates(
  candA: number | null, candB: number | null, out: number[] = [0, 0, 0],
): number[] {
  const a = candA ?? 1;
  const b = candB ?? 1;
  if (a === b) {
    if (a < 2) {
      out[0] = 0; out[1] = 1; out[2] = 26;
      return out; // planar, DC, angular 26
    }
    // same angular mode m: [m, m-1, m+1] with wrap in 2..33
    out[0] = a;
    out[1] = 2 + ((a - 2 - 1 + 32) % 32);
    out[2] = 2 + ((a - 2 + 1) % 32);
    return out;
  }
  let third: number;
  if (a !== 0 && b !== 0) third = 0;
  else if (a !== 1 && b !== 1) third = 1;
  else third = 26;
  out[0] = a; out[1] = b; out[2] = third;
  return out;
}

export function intraPredModeDecode(prevIntraPredFlag: number, mpmIdx: number, rem: number, candModeList: number[]): number {
  if (prevIntraPredFlag) return candModeList[mpmIdx]!;
  // spec 8.4.3: sort ascending, increment once per candidate <= current mode
  let a = candModeList[0]!, b = candModeList[1]!, c = candModeList[2]!;
  if (a > b) { const t = a; a = b; b = t; }
  if (b > c) { const t = b; b = c; c = t; }
  if (a > b) { const t = a; a = b; b = t; }
  let intraMode = rem;
  if (intraMode >= a) intraMode++;
  if (intraMode >= b) intraMode++;
  if (intraMode >= c) intraMode++;
  return intraMode;
}

/** chroma pred mode mapping */
export function chromaPredMode(lumaMode: number, chromaIdx: number): number {
  switch (chromaIdx) {
    case 4: return lumaMode;                       // DM
    case 0: return lumaMode === 0 ? 34 : 0;        // planar
    case 1: return lumaMode === 26 ? 34 : 26;
    case 2: return lumaMode === 10 ? 34 : 10;
    case 3: return lumaMode === 1 ? 34 : 1;
    default: throw new Error('bad chroma mode');
  }
}
