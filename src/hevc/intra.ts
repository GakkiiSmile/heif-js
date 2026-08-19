/** HEVC intra prediction (spec 8.4.4): border collection, substitution,
 *  filtering, and the 35 prediction modes. */
import { INTRA_ANGLE, INV_ANGLE } from './tables.ts';
import { Plane } from '../frame.ts';
import { debugEnabled, debugWrite } from '../debug.ts';

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
  const border = new Int32Array(2 * 64 + 1);
  const off = 64;
  const avail = new Uint8Array(2 * 64 + 1);
  const stride = plane.stride;
  const W = plane.width, H = plane.height;

  // availability in 4x4 grid coordinates of this component
  const SubW = cIdx === 0 || ctx.chromaArrayType === 3 ? 1 : 2;
  const SubH = cIdx === 0 || ctx.chromaArrayType === 2 || ctx.chromaArrayType === 3 ? 1 : 2;
  const tbX = (x: number) => (x * SubW) >> log2MinTb;
  const tbY = (y: number) => (y * SubH) >> log2MinTb;
  const currZ = ctx.minTbAddrZs
    ? ctx.minTbAddrZs[tbX(xB) + tbY(yB) * ctx.picWidthInTbs]!
    : -1;

  const canUse = (x: number, y: number): boolean => {
    if (x < 0 || y < 0 || x >= W || y >= H) return false;
    const gi = tbX(x) + tbY(y) * ctx.picWidthInTbs;
    if (ctx.sliceIdMap && ctx.sliceIdMap[gi] !== ctx.currentSliceId) return false;
    if (ctx.tileIdMap && ctx.tileIdMap[gi] !== ctx.currentTileId) return false;
    if (!ctx.minTbAddrZs) return true; // assume decoded (we only predict from decoded area)
    const gz = ctx.minTbAddrZs[tbX(x) + tbY(y) * ctx.picWidthInTbs]!;
    return gz <= currZ;
  };

  let nAvail = 0;
  let firstValue = 0;
  const setAvail = (i: number, x: number, y: number) => {
    if (!canUse(x, y)) return;
    if (nAvail === 0) firstValue = plane.data[x + y * stride]!;
    avail[off + i] = 1;
    border[off + i] = plane.data[x + y * stride]!;
    nAvail++;
  };

  // left column: border[-1-y] for y = nT-1 .. 0 collected bottom-up (order matters for firstValue)
  for (let y = 2 * nT - 1; y >= 0; y--) setAvail(-1 - y, xB - 1, yB + y);
  setAvail(0, xB - 1, yB - 1);
  for (let x = 0; x < 2 * nT; x++) setAvail(1 + x, xB + x, yB - 1);

  // ---- reference sample substitution (8.4.4.2.2)
  if (debugEnabled('HEVC_TU_DEBUG') && xB === 0 && yB === 0) {
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

  if (debugEnabled('HEVC_TU_DEBUG') && xB === 0 && yB === 0) {
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
      const p = (i: number) => border[off + i];
      const biInt = ctx.strongIntraSmoothing && cIdx === 0 && nT === 32 &&
        Math.abs(p(0) + p(64) - 2 * p(32)) < (1 << (bd - 5)) &&
        Math.abs(p(0) + p(-64) - 2 * p(-32)) < (1 << (bd - 5)) ? 1 : 0;
      const pF = new Int32Array(2 * 64 + 1);
      if (biInt) {
        pF[off - 2 * nT] = p(-2 * nT);
        pF[off + 2 * nT] = p(2 * nT);
        pF[off] = p(0);
        for (let i = 1; i <= 63; i++) {
          pF[off - i] = p(0) + ((i * (p(-64) - p(0)) + 32) >> 6);
          pF[off + i] = p(0) + ((i * (p(64) - p(0)) + 32) >> 6);
        }
      } else {
        pF[off - 2 * nT] = p(-2 * nT);
        pF[off + 2 * nT] = p(2 * nT);
        for (let i = -(2 * nT - 1); i <= 2 * nT - 1; i++) {
          pF[off + i] = (p(i + 1) + 2 * p(i) + p(i - 1) + 2) >> 2;
        }
      }
      for (let i = -2 * nT; i <= 2 * nT; i++) border[off + i] = pF[off + i]!;
    }
  }

  const dst = plane.data;
  const dstride = stride;
  const b = (i: number) => border[off + i];

  if (predMode === 0) {
    // planar
    const log2nT = Math.log2(nT);
    for (let y = 0; y < nT; y++) {
      for (let x = 0; x < nT; x++) {
        dst[xB + x + (yB + y) * dstride] =
          ((nT - 1 - x) * b(-1 - y) + (x + 1) * b(1 + nT) +
            (nT - 1 - y) * b(1 + x) + (y + 1) * b(-1 - nT) + nT) >> (log2nT + 1);
      }
    }
    return;
  }
  if (predMode === 1) {
    // DC
    let sum = 0;
    for (let i = 0; i < nT; i++) { sum += b(1 + i) + b(-1 - i); }
    const dc = (sum + nT) >> (Math.log2(nT) + 1);
    for (let y = 0; y < nT; y++) for (let x = 0; x < nT; x++) dst[xB + x + (yB + y) * dstride] = dc;
    // boundary filter (cIdx 0, nT < 32)
    if (cIdx === 0 && nT < 32) {
      dst[xB + yB * dstride] = (b(-1) + 2 * dc + b(1) + 2) >> 2;
      for (let x = 1; x < nT; x++) dst[xB + x + yB * dstride] = (b(1 + x) + 3 * dc + 2) >> 2;
      for (let y = 1; y < nT; y++) dst[xB + (yB + y) * dstride] = (b(-1 - y) + 3 * dc + 2) >> 2;
    }
    return;
  }

  // angular
  const angle = INTRA_ANGLE[predMode]!;
  const ref = new Int32Array(4 * 32 + 1);
  const roff = 64;
  if (predMode >= 18) {
    for (let x = 0; x <= nT; x++) ref[roff + x] = b(x);
    if (angle < 0) {
      const inv = INV_ANGLE[predMode - 11]!;
      if ((nT * angle) >> 5 < -1) {
        for (let x = (nT * angle) >> 5; x <= -1; x++) {
          ref[roff + x] = b(-((x * inv + 128) >> 8));
        }
      }
    } else {
      for (let x = nT + 1; x <= 2 * nT; x++) ref[roff + x] = b(x);
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
        dst[xB + (yB + y) * dstride] = clipBd(b(1) + ((b(-1 - y) - b(0)) >> 1), bd);
      }
    }
  } else {
    for (let x = 0; x <= nT; x++) ref[roff + x] = b(-x);
    if (angle < 0) {
      const inv = INV_ANGLE[predMode - 11]!;
      if ((nT * angle) >> 5 < -1) {
        for (let x = (nT * angle) >> 5; x <= -1; x++) {
          ref[roff + x] = b((x * inv + 128) >> 8);
        }
      }
    } else {
      for (let x = nT + 1; x <= 2 * nT; x++) ref[roff + x] = b(-x);
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
        dst[xB + x + yB * dstride] = clipBd(b(-1) + ((b(1 + x) - b(0)) >> 1), bd);
      }
    }
  }
}

/** MPM candidate derivation (spec 8.4.3). Unavailable neighbors use DC (1). */
export function fillIntraPredModeCandidates(
  candA: number | null, candB: number | null,
): number[] {
  const a = candA ?? 1;
  const b = candB ?? 1;
  if (a === b) {
    if (a < 2) {
      return [0, 1, 26]; // planar, DC, angular 26
    }
    // same angular mode m: [m, m-1, m+1] with wrap in 2..33
    return [a, 2 + ((a - 2 - 1 + 32) % 32), 2 + ((a - 2 + 1) % 32)];
  }
  let third: number;
  if (a !== 0 && b !== 0) third = 0;
  else if (a !== 1 && b !== 1) third = 1;
  else third = 26;
  return [a, b, third];
}

export function intraPredModeDecode(prevIntraPredFlag: number, mpmIdx: number, rem: number, candModeList: number[]): number {
  if (prevIntraPredFlag) return candModeList[mpmIdx]!;
  // spec 8.4.3: sort ascending, increment once per candidate <= current mode
  const l = candModeList.slice().sort((a, b) => a - b);
  let intraMode = rem;
  for (let i = 0; i < 3; i++) {
    if (intraMode >= l[i]!) intraMode++;
  }
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
