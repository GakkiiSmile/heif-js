/** HEVC inverse transforms + dequantization (spec 8.6). */
import { LEVEL_SCALE } from './tables.ts';

/** DCT-II integer approximation basis (32x32); NxN transforms use strided rows. */
export const MAT_DCT: number[][] = [
  [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
  [90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13, 4, -4, -13, -22, -31, -38, -46, -54, -61, -67, -73, -78, -82, -85, -88, -90, -90],
  [90, 87, 80, 70, 57, 43, 25, 9, -9, -25, -43, -57, -70, -80, -87, -90, -90, -87, -80, -70, -57, -43, -25, -9, 9, 25, 43, 57, 70, 80, 87, 90],
  [90, 82, 67, 46, 22, -4, -31, -54, -73, -85, -90, -88, -78, -61, -38, -13, 13, 38, 61, 78, 88, 90, 85, 73, 54, 31, 4, -22, -46, -67, -82, -90],
  [89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89, 89, 75, 50, 18, -18, -50, -75, -89, -89, -75, -50, -18, 18, 50, 75, 89],
  [88, 67, 31, -13, -54, -82, -90, -78, -46, -4, 38, 73, 90, 85, 61, 22, -22, -61, -85, -90, -73, -38, 4, 46, 78, 90, 82, 54, 13, -31, -67, -88],
  [87, 57, 9, -43, -80, -90, -70, -25, 25, 70, 90, 80, 43, -9, -57, -87, -87, -57, -9, 43, 80, 90, 70, 25, -25, -70, -90, -80, -43, 9, 57, 87],
  [85, 46, -13, -67, -90, -73, -22, 38, 82, 88, 54, -4, -61, -90, -78, -31, 31, 78, 90, 61, 4, -54, -88, -82, -38, 22, 73, 90, 67, 13, -46, -85],
  [83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83, 83, 36, -36, -83, -83, -36, 36, 83],
  [82, 22, -54, -90, -61, 13, 78, 85, 31, -46, -90, -67, 4, 73, 88, 38, -38, -88, -73, -4, 67, 90, 46, -31, -85, -78, -13, 61, 90, 54, -22, -82],
  [80, 9, -70, -87, -25, 57, 90, 43, -43, -90, -57, 25, 87, 70, -9, -80, -80, -9, 70, 87, 25, -57, -90, -43, 43, 90, 57, -25, -87, -70, 9, 80],
  [78, -4, -82, -73, 13, 85, 67, -22, -88, -61, 31, 90, 54, -38, -90, -46, 46, 90, 38, -54, -90, -31, 61, 88, 22, -67, -85, -13, 73, 82, 4, -78],
  [75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75, 75, -18, -89, -50, 50, 89, 18, -75, -75, 18, 89, 50, -50, -89, -18, 75],
  [73, -31, -90, -22, 78, 67, -38, -90, -13, 82, 61, -46, -88, -4, 85, 54, -54, -85, 4, 88, 46, -61, -82, 13, 90, 38, -67, -78, 22, 90, 31, -73],
  [70, -43, -87, 9, 90, 25, -80, -57, 57, 80, -25, -90, -9, 87, 43, -70, -70, 43, 87, -9, -90, -25, 80, 57, -57, -80, 25, 90, 9, -87, -43, 70],
  [67, -54, -78, 38, 85, -22, -90, 4, 90, 13, -88, -31, 82, 46, -73, -61, 61, 73, -46, -82, 31, 88, -13, -90, -4, 90, 22, -85, -38, 78, 54, -67],
  [64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64, 64, -64, -64, 64],
  [61, -73, -46, 82, 31, -88, -13, 90, -4, -90, 22, 85, -38, -78, 54, 67, -67, -54, 78, 38, -85, -22, 90, 4, -90, 13, 88, -31, -82, 46, 73, -61],
  [57, -80, -25, 90, -9, -87, 43, 70, -70, -43, 87, 9, -90, 25, 80, -57, -57, 80, 25, -90, 9, 87, -43, -70, 70, 43, -87, -9, 90, -25, -80, 57],
  [54, -85, -4, 88, -46, -61, 82, 13, -90, 38, 67, -78, -22, 90, -31, -73, 73, 31, -90, 22, 78, -67, -38, 90, -13, -82, 61, 46, -88, 4, 85, -54],
  [50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50, 50, -89, 18, 75, -75, -18, 89, -50, -50, 89, -18, -75, 75, 18, -89, 50],
  [46, -90, 38, 54, -90, 31, 61, -88, 22, 67, -85, 13, 73, -82, 4, 78, -78, -4, 82, -73, -13, 85, -67, -22, 88, -61, -31, 90, -54, -38, 90, -46],
  [43, -90, 57, 25, -87, 70, 9, -80, 80, -9, -70, 87, -25, -57, 90, -43, -43, 90, -57, -25, 87, -70, -9, 80, -80, 9, 70, -87, 25, 57, -90, 43],
  [38, -88, 73, -4, -67, 90, -46, -31, 85, -78, 13, 61, -90, 54, 22, -82, 82, -22, -54, 90, -61, -13, 78, -85, 31, 46, -90, 67, 4, -73, 88, -38],
  [36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36, 36, -83, 83, -36, -36, 83, -83, 36],
  [31, -78, 90, -61, 4, 54, -88, 82, -38, -22, 73, -90, 67, -13, -46, 85, -85, 46, 13, -67, 90, -73, 22, 38, -82, 88, -54, -4, 61, -90, 78, -31],
  [25, -70, 90, -80, 43, 9, -57, 87, -87, 57, -9, -43, 80, -90, 70, -25, -25, 70, -90, 80, -43, -9, 57, -87, 87, -57, 9, 43, -80, 90, -70, 25],
  [22, -61, 85, -90, 73, -38, -4, 46, -78, 90, -82, 54, -13, -31, 67, -88, 88, -67, 31, 13, -54, 82, -90, 78, -46, 4, 38, -73, 90, -85, 61, -22],
  [18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18, 18, -50, 75, -89, 89, -75, 50, -18, -18, 50, -75, 89, -89, 75, -50, 18],
  [13, -38, 61, -78, 88, -90, 85, -73, 54, -31, 4, 22, -46, 67, -82, 90, -90, 82, -67, 46, -22, -4, 31, -54, 73, -85, 90, -88, 78, -61, 38, -13],
  [9, -25, 43, -57, 70, -80, 87, -90, 90, -87, 80, -70, 57, -43, 25, -9, -9, 25, -43, 57, -70, 80, -87, 90, -90, 87, -80, 70, -57, 43, -25, 9],
  [4, -13, 22, -31, 38, -46, 54, -61, 67, -73, 78, -82, 85, -88, 90, -90, 90, -90, 88, -85, 82, -78, 73, -67, 61, -54, 46, -38, 31, -22, 13, -4],
];

/** DST-IV basis for 4x4 luma */
export const MAT_DST4 = [
  [29, 55, 74, 84],
  [74, 74, 0, -74],
  [84, -29, -74, 55],
  [55, -84, 74, -29],
];

/** dequantized coefficient block (int16) */
export function dequant(
  coeffs: ArrayLike<number>, positions: ArrayLike<number>, nCoeffs: number,
  nT: number, qP: number, bitDepth: number,
  scalingFactors: Uint8Array | null, rotateCoeffs: boolean,
): Int16Array {
  const out = new Int16Array(nT * nT);
  let bdShift = bitDepth + Math.log2(nT) - 5;
  let m = 16;
  if (!scalingFactors) {
    m = 1;
    bdShift -= 4;
  }
  const offset = 1 << (bdShift - 1);
  for (let i = 0; i < nCoeffs; i++) {
    const sourcePos = positions[i]!;
    let pos = sourcePos;
    if (rotateCoeffs) {
      // transform_skip_rotation for 4x4 intra is a 180-degree reversal.
      pos = nT * nT - 1 - pos;
    }
    const mf = scalingFactors ? scalingFactors[sourcePos]! : m;
    const fact = mf * LEVEL_SCALE[qP % 6]! * 2 ** Math.floor(qP / 6);
    const c = coeffs[i]!;
    let v = Math.floor((c * fact + offset) / 2 ** bdShift);
    if (v < -32768) v = -32768; else if (v > 32767) v = 32767;
    out[pos] = v;
  }
  return out;
}

/**
 * Inverse transform; adds residual into plane data at (xT,yT).
 * `dst` true for 4x4 luma (DST instead of DCT).
 */
export function addInverseTransform(
  planeData: Uint16Array, stride: number, xT: number, yT: number,
  coeff: Int16Array, nT: number, bitDepth: number, useDst: boolean,
  residualOut?: Int32Array,
): void {
  const postShift = 20 - bitDepth;
  const rnd1 = 64, rnd2 = 1 << (postShift - 1);
  const fact = nT === 4 ? 8 : nT === 8 ? 4 : nT === 16 ? 2 : 1;
  const g = new Int16Array(nT * nT);

  // vertical pass
  for (let c = 0; c < nT; c++) {
    let lastCol = nT - 1;
    while (lastCol >= 0 && coeff[c + lastCol * nT] === 0) lastCol--;
    for (let i = 0; i < nT; i++) {
      let sum = 0;
      if (useDst) {
        for (let j = 0; j < nT; j++) sum += MAT_DST4[j]![i]! * coeff[c + j * nT]!;
      } else {
        for (let j = 0; j <= lastCol; j++) sum += MAT_DCT[fact * j]![i]! * coeff[c + j * nT]!;
      }
      let v = (sum + rnd1) >> 7;
      if (v < -32768) v = -32768; else if (v > 32767) v = 32767;
      g[c + i * nT] = v;
    }
  }
  // horizontal pass + add
  const maxVal = (1 << bitDepth) - 1;
  for (let y = 0; y < nT; y++) {
    const rowOff = (yT + y) * stride + xT;
    const gOff = y * nT;
    for (let i = 0; i < nT; i++) {
      let sum = 0;
      if (useDst) {
        for (let j = 0; j < nT; j++) sum += MAT_DST4[j]![i]! * g[gOff + j]!;
      } else {
        for (let j = 0; j < nT; j++) {
          const cval = g[gOff + j]!;
          if (cval !== 0) sum += MAT_DCT[fact * j]![i]! * cval;
        }
      }
      const out = (sum + rnd2) >> postShift;
      if (residualOut) residualOut[i + y * nT] = out;
      const v = planeData[rowOff + i]! + out;
      planeData[rowOff + i] = v < 0 ? 0 : v > maxVal ? maxVal : v;
    }
  }
}

/** transform skip: residual = coeff << 7-ish scaling (8.6.3 bypass transform) */
export function addTransformSkip(
  planeData: Uint16Array, stride: number, xT: number, yT: number,
  coeff: Int16Array, nT: number, bitDepth: number, rdpcmMode = 0, extendedPrecision = false,
  residualOut?: Int32Array,
): void {
  const bdShift = Math.max(20 - bitDepth, extendedPrecision ? 11 : 0);
  const tsShift = (extendedPrecision ? Math.min(5, bdShift - 2) : 5) + Math.log2(nT);
  const rnd = 1 << (bdShift - 1);
  const maxVal = (1 << bitDepth) - 1;
  const residual = new Int32Array(nT * nT);
  for (let y = 0; y < nT; y++) {
    for (let x = 0; x < nT; x++) {
      const c = coeff[x + y * nT]!;
      let out = (c * (1 << tsShift) + rnd) >> bdShift;
      if (rdpcmMode === 1 && x > 0) out += residual[x - 1 + y * nT]!;
      else if (rdpcmMode === 2 && y > 0) out += residual[x + (y - 1) * nT]!;
      residual[x + y * nT] = out;
      if (residualOut) residualOut[x + y * nT] = out;
      const v = planeData[(yT + y) * stride + xT + x]! + out;
      planeData[(yT + y) * stride + xT + x] = v < 0 ? 0 : v > maxVal ? maxVal : v;
    }
  }
}
