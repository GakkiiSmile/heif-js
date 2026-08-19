/** HEVC deblocking filter (spec 8.7.2). In-place on the frame planes. */
import { DecodedFrame, CHROMA_MONO, CHROMA_420, CHROMA_422, CHROMA_444 } from '../frame.ts';
import type { Spt, Pps } from './pps.ts';
import { CHROMA_QP } from './tables.ts';

export interface DeblockInfo {
  qpMap: Int8Array;
  widthInMinCbs: number;
  minCbLog2: number;
  cbfYMap: Uint8Array;
  cbfCbMap: Uint8Array;
  cbfCrMap: Uint8Array;
  cuIdMap: Int32Array;
  tuIdMap: Int32Array;
  tuSizeLog2Map: Int8Array;
  pcmMap: Uint8Array;
  pcmLoopFilterDisable: boolean;
  sliceIdMap: Int32Array;
  tileIdMap: Int32Array;
  loopFilterAcrossSlices: boolean;
  loopFilterAcrossTiles: boolean;
  gridW: number;
  gridH: number;
  bitDepth: number;
  chromaBitDepth: number;
  betaOffsetDiv2: number;
  tcOffsetDiv2: number;
  cbQpOffset: number;
  crQpOffset: number;
  disable: boolean;
  chromaFormat: number;
  sliceBorders: null; // unused: single-slice pictures
  ctbSizeLog2: number;
  tileColBd: number[];
  tileRowBd: number[];
}

const BETA_TAB = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64];
const TC_TAB = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 8, 9, 10, 11, 13, 14, 16, 18, 20, 22, 24];

  const clip3 = (lo: number, hi: number, v: number) => v < lo ? lo : v > hi ? hi : v;

export function applyDeblock(frame: DecodedFrame, info: DeblockInfo): void {
  const spsBdY = info.bitDepth;
  const { qpMap, widthInMinCbs, minCbLog2, gridW, gridH } = info;
  const W = frame.luma.width, H = frame.luma.height;
  const luma = frame.luma;
  const data = luma.data, stride = luma.stride;
  const maxVal = (1 << spsBdY) - 1;
  const bdShift = spsBdY - 8;

  const qpAt = (x: number, y: number): number => {
    const gx = Math.min(widthInMinCbs - 1, x >> minCbLog2);
    const gy = Math.min(qpMap.length / widthInMinCbs - 1, y >> minCbLog2);
    return qpMap[gy * widthInMinCbs + gx]!;
  };
  const g = (x: number, y: number) => (y >> 2) * gridW + (x >> 2);

  const ctbSize = 1 << info.ctbSizeLog2;
  const ctbCols = Math.ceil(W / ctbSize), ctbRows = Math.ceil(H / ctbSize);

  // bS per 4x4 segment on an 8x8-grid edge (vertical: x fixed; horizontal: y fixed)
  const bs = (giP: number, giQ: number): number => {
    if (!info.loopFilterAcrossSlices && info.sliceIdMap[giP] !== info.sliceIdMap[giQ]) return 0;
    if (!info.loopFilterAcrossTiles && info.tileIdMap[giP] !== info.tileIdMap[giQ]) return 0;
    if (info.pcmLoopFilterDisable && (info.pcmMap[giP] || info.pcmMap[giQ])) return 0;
    if (info.cuIdMap[giP] !== info.cuIdMap[giQ]) return 2;
    if (info.tuIdMap[giP] !== info.tuIdMap[giQ]) {
      if (info.cbfYMap[giP] || info.cbfYMap[giQ]) return 1;
    }
    return 0;
  };

  for (let ctbR = 0; ctbR < ctbRows; ctbR++) {
    for (let ctbC = 0; ctbC < ctbCols; ctbC++) {
      const x0 = ctbC * ctbSize, y0 = ctbR * ctbSize;
      // ---- vertical luma edges ----
      for (let x = Math.max(8, x0); x < Math.min(x0 + ctbSize, W); x += 8) {
        // skip tile boundaries when loop filter across tiles disabled
        for (let y = y0; y < Math.min(y0 + ctbSize, H); y += 4) {
          const giP = g(x - 4, y), giQ = g(x, y);
          const b = bs(giP, giQ);
          if (!b) continue;
          filterLuma(data, stride, x, y, true, b, qpAt(x, y), qpAt(x - 4, y), info, spsBdY, maxVal, bdShift);
        }
      }
      // ---- horizontal luma edges ----
      for (let y = Math.max(8, y0); y < Math.min(y0 + ctbSize, H); y += 8) {
        for (let x = x0; x < Math.min(x0 + ctbSize, W); x += 4) {
          const giP = g(x, y - 4), giQ = g(x, y);
          const b = bs(giP, giQ);
          if (!b) continue;
          filterLuma(data, stride, x, y, false, b, qpAt(x, y), qpAt(x, y - 4), info, spsBdY, maxVal, bdShift);
        }
      }
    }
  }

  // ---- chroma edges ----
  if (info.chromaFormat === CHROMA_MONO || info.chromaFormat === 0) return;
  const hS = info.chromaFormat === CHROMA_420 || info.chromaFormat === CHROMA_422 ? 1 : 0;
  const vS = info.chromaFormat === CHROMA_420 ? 1 : 0;
  const chromaMax = (1 << info.chromaBitDepth) - 1;
  const cBdShift = info.chromaBitDepth - 8;

  for (let c = 1; c <= 2; c++) {
    const pl = frame.planes[c]!;
    const cd = pl.data, cs = pl.stride;
    const cW = pl.width, cH = pl.height;
    const cbfMap = c === 1 ? info.cbfCbMap : info.cbfCrMap;
    for (let ctbR = 0; ctbR < ctbRows; ctbR++) {
      for (let ctbC = 0; ctbC < ctbCols; ctbC++) {
        const cx0 = (ctbC * ctbSize) >> hS, cy0 = (ctbR * ctbSize) >> vS;
        const cw = ctbSize >> hS, ch = ctbSize >> vS;
        for (let x = Math.max(4, cx0); x < Math.min(cx0 + cw, cW); x += 4) {
          for (let y = cy0; y < Math.min(cy0 + ch, cH); y += 4) {
            // luma 8x8 grid edge -> chroma 4x4
            const lx = x << hS, ly = y << vS;
            const giP = g(lx - 8, ly), giQ = g(lx, ly);
            if (!info.loopFilterAcrossSlices && info.sliceIdMap[giP] !== info.sliceIdMap[giQ]) continue;
            if (!info.loopFilterAcrossTiles && info.tileIdMap[giP] !== info.tileIdMap[giQ]) continue;
            if (info.pcmLoopFilterDisable && (info.pcmMap[giP] || info.pcmMap[giQ])) continue;
            if (info.cuIdMap[giP] !== info.cuIdMap[giQ]) {
              // bS == 2 -> filter
              const qpAvg = (qpAt(lx, ly) + qpAt(lx - 1, ly) + 1) >> 1;
              filterChroma(cd, cs, x, y, true, qpAvg, c === 1 ? info.cbQpOffset : info.crQpOffset, info, cBdShift, chromaMax);
            }
          }
        }
        for (let y = Math.max(4, cy0); y < Math.min(cy0 + ch, cH); y += 4) {
          for (let x = cx0; x < Math.min(cx0 + cw, cW); x += 4) {
            const lx = x << hS, ly = y << vS;
            const giP = g(lx, ly - 8), giQ = g(lx, ly);
            if (!info.loopFilterAcrossSlices && info.sliceIdMap[giP] !== info.sliceIdMap[giQ]) continue;
            if (!info.loopFilterAcrossTiles && info.tileIdMap[giP] !== info.tileIdMap[giQ]) continue;
            if (info.pcmLoopFilterDisable && (info.pcmMap[giP] || info.pcmMap[giQ])) continue;
            if (info.cuIdMap[giP] !== info.cuIdMap[giQ]) {
              const qpAvg = (qpAt(lx, ly) + qpAt(lx, ly - 1) + 1) >> 1;
              filterChroma(cd, cs, x, y, false, qpAvg, c === 1 ? info.cbQpOffset : info.crQpOffset, info, cBdShift, chromaMax);
            }
          }
        }
      }
    }
    void cbfMap;
  }
}

function filterLuma(
  data: Uint16Array, stride: number, x: number, y: number, vertical: boolean,
  bS: number, qpQ: number, qpP: number, info: DeblockInfo, bitDepth: number,
  maxVal: number, bdShift: number,
): void {
  const qPL = (qpQ + qpP + 1) >> 1;
  const qBeta = clip3(0, 51, qPL + 2 * info.betaOffsetDiv2);
  const beta = BETA_TAB[qBeta]! * (1 << bdShift);
  const qTc = clip3(0, 53, qPL + 2 * (bS - 1) + 2 * info.tcOffsetDiv2);
  const tc = TC_TAB[qTc]! * (1 << bdShift);

  // fetch p/q, 4 lines (k) x 4 taps (i)
  const p: number[][] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  const q: number[][] = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  if (vertical) {
    for (let k = 0; k < 4; k++) {
      const row = (y + k) * stride + x;
      for (let i = 0; i < 4; i++) {
        q[k]![i] = data[row + i]!;
        p[k]![i] = data[row - i - 1]!;
      }
    }
  } else {
    for (let k = 0; k < 4; k++) {
      const col = x + k;
      for (let i = 0; i < 4; i++) {
        q[k]![i] = data[col + (y + i) * stride]!;
        p[k]![i] = data[col + (y - i - 1) * stride]!;
      }
    }
  }

  const dp0 = Math.abs(p[0]![2]! - 2 * p[0]![1]! + p[0]![0]!);
  const dp3 = Math.abs(p[3]![2]! - 2 * p[3]![1]! + p[3]![0]!);
  const dq0 = Math.abs(q[0]![2]! - 2 * q[0]![1]! + q[0]![0]!);
  const dq3 = Math.abs(q[3]![2]! - 2 * q[3]![1]! + q[3]![0]!);
  const dp = dp0 + dp3, dq = dq0 + dq3;
  const d = dp0 + dq0 + dp3 + dq3;
  if (d >= beta) return;

  const dpq0 = dp0 + dq0, dpq3 = dp3 + dq3;
  const dSam0 = 2 * dpq0 < (beta >> 2) &&
    Math.abs(p[0]![3]! - p[0]![0]!) + Math.abs(q[0]![0]! - q[0]![3]!) < (beta >> 3) &&
    Math.abs(p[0]![0]! - q[0]![0]!) < ((5 * tc + 1) >> 1);
  const dSam3 = 2 * dpq3 < (beta >> 2) &&
    Math.abs(p[3]![3]! - p[3]![0]!) + Math.abs(q[3]![0]! - q[3]![3]!) < (beta >> 3) &&
    Math.abs(p[3]![0]! - q[3]![0]!) < ((5 * tc + 1) >> 1);
  const dE = dSam0 && dSam3 ? 2 : 1;
  const dEp = dp < ((beta + (beta >> 1)) >> 3) ? 1 : 0;
  const dEq = dq < ((beta + (beta >> 1)) >> 3) ? 1 : 0;

  const clip1 = (v: number) => v < 0 ? 0 : v > maxVal ? maxVal : v;

  if (vertical) {
    for (let k = 0; k < 4; k++) {
      const row = (y + k) * stride + x;
      const p0 = p[k]![0]!, p1 = p[k]![1]!, p2 = p[k]![2]!, p3 = p[k]![3]!;
      const q0 = q[k]![0]!, q1 = q[k]![1]!, q2 = q[k]![2]!, q3 = q[k]![3]!;
      if (dE === 2) {
        data[row - 1] = clip3(p0 - 2 * tc, p0 + 2 * tc, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
        data[row - 2] = clip3(p1 - 2 * tc, p1 + 2 * tc, (p2 + p1 + p0 + q0 + 2) >> 2);
        data[row - 3] = clip3(p2 - 2 * tc, p2 + 2 * tc, (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3);
        data[row + 0] = clip3(q0 - 2 * tc, q0 + 2 * tc, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
        data[row + 1] = clip3(q1 - 2 * tc, q1 + 2 * tc, (p0 + q0 + q1 + q2 + 2) >> 2);
        data[row + 2] = clip3(q2 - 2 * tc, q2 + 2 * tc, (p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3);
      } else {
        let delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
        if (Math.abs(delta) < tc * 10) {
          delta = clip3(-tc, tc, delta);
          data[row - 1] = clip1(p0 + delta);
          data[row] = clip1(q0 - delta);
          if (dEp) {
            const dp = clip3(-(tc >> 1), tc >> 1, (((p2 + p0 + 1) >> 1) - p1 + delta) >> 1);
            data[row - 2] = clip1(p1 + dp);
          }
          if (dEq) {
            const dq2 = clip3(-(tc >> 1), tc >> 1, (((q2 + q0 + 1) >> 1) - q1 - delta) >> 1);
            data[row + 1] = clip1(q1 + dq2);
          }
        }
      }
    }
  } else {
    for (let k = 0; k < 4; k++) {
      const col = x + k;
      const p0 = p[k]![0]!, p1 = p[k]![1]!, p2 = p[k]![2]!, p3 = p[k]![3]!;
      const q0 = q[k]![0]!, q1 = q[k]![1]!, q2 = q[k]![2]!, q3 = q[k]![3]!;
      const ry = (v: number, off: number) => col + (y + off) * stride;
      if (dE === 2) {
        data[ry(0, -1)] = clip3(p0 - 2 * tc, p0 + 2 * tc, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
        data[ry(0, -2)] = clip3(p1 - 2 * tc, p1 + 2 * tc, (p2 + p1 + p0 + q0 + 2) >> 2);
        data[ry(0, -3)] = clip3(p2 - 2 * tc, p2 + 2 * tc, (2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3);
        data[ry(0, 0)] = clip3(q0 - 2 * tc, q0 + 2 * tc, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
        data[ry(0, 1)] = clip3(q1 - 2 * tc, q1 + 2 * tc, (p0 + q0 + q1 + q2 + 2) >> 2);
        data[ry(0, 2)] = clip3(q2 - 2 * tc, q2 + 2 * tc, (p0 + q0 + q1 + 3 * q2 + 2 * q3 + 4) >> 3);
      } else {
        let delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
        if (Math.abs(delta) < tc * 10) {
          delta = clip3(-tc, tc, delta);
          data[ry(0, -1)] = clip1(p0 + delta);
          data[ry(0, 0)] = clip1(q0 - delta);
          if (dEp) {
            const dp = clip3(-(tc >> 1), tc >> 1, (((p2 + p0 + 1) >> 1) - p1 + delta) >> 1);
            data[ry(0, -2)] = clip1(p1 + dp);
          }
          if (dEq) {
            const dq2 = clip3(-(tc >> 1), tc >> 1, (((q2 + q0 + 1) >> 1) - q1 - delta) >> 1);
            data[ry(0, 1)] = clip1(q1 + dq2);
          }
        }
      }
    }
  }
}

function filterChroma(
  data: Uint16Array, stride: number, x: number, y: number, vertical: boolean,
  qpAvg: number, qpOffset: number, info: DeblockInfo, bdShift: number, maxVal: number,
): void {
  const qPi = Math.min(57, Math.max(0, qpAvg + qpOffset));
  const qP_C = info.chromaFormat === CHROMA_420 ? CHROMA_QP[qPi]! : qPi;
  const Q = clip3(0, 53, qP_C + 1 * 2 + 2 * info.tcOffsetDiv2); // bS=2
  const tc = TC_TAB[Q]! * (1 << bdShift);
  const clip1 = (v: number) => v < 0 ? 0 : v > maxVal ? maxVal : v;
  for (let k = 0; k < 4; k++) {
    let p0: number, p1: number, q0: number, q1: number, idx: number;
    if (vertical) {
      idx = (y + k) * stride + x;
      p0 = data[idx - 1]!; p1 = data[idx - 2]!; q0 = data[idx]!; q1 = data[idx + 1]!;
    } else {
      idx = x + k + y * stride;
      p0 = data[idx - stride]!; p1 = data[idx - 2 * stride]!; q0 = data[idx]!; q1 = data[idx + stride]!;
    }
    const delta = clip3(-tc, tc, (((q0 - p0) * 4 + p1 - q1 + 4) >> 3));
    if (vertical) {
      data[idx - 1] = clip1(p0 + delta);
      data[idx] = clip1(q0 - delta);
    } else {
      data[idx - stride] = clip1(p0 + delta);
      data[idx] = clip1(q0 - delta);
    }
  }
}
