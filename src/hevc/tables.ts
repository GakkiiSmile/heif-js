/** HEVC constant tables (spec chapter 8). */

/** intraPredModeAngle for modes 0..34 */
export const INTRA_ANGLE = [
  0, 0, 32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26,
  -32, -26, -21, -17, -13, -9, -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32,
];

/** inverse angles for modes 11..25 */
export const INV_ANGLE = [
  -4096, -1638, -910, -630, -482, -390, -315, -256,
  -315, -390, -482, -630, -910, -1638, -4096,
];

/** scan orders: 0=diagonal, 1=horizontal, 2=vertical; SCAN[k] is for size 2^(k+1) (k=0 -> 2x2) */
export const SCAN: number[][][] = [];
for (let log2 = 1; log2 <= 5; log2++) {
  const n = 1 << log2;
  const diag: number[] = [], horiz: number[] = [], vert: number[] = [];
  for (let d = 0; d < 2 * n - 1; d++) {
    const xs: number[] = [];
    for (let x = 0; x < n; x++) {
      const y = d - x;
      if (y >= 0 && y < n) xs.push(x);
    }
    for (const x of xs) diag.push(x + (d - x) * n);
  }
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) horiz.push(x + y * n);
  for (let x = 0; x < n; x++) for (let y = 0; y < n; y++) vert.push(x + y * n);
  SCAN.push([diag, horiz, vert]);
}

/** levelScale[qp%6] (8.6.1) */
export const LEVEL_SCALE = [40, 45, 51, 57, 64, 72];

/** chroma QP mapping (spec Table 8-22), input qPi 0..57 (already clipped, in 8-bit domain) */
export const CHROMA_QP = Array.from({ length: 58 }, (_, qPi) => {
  if (qPi < 30) return qPi;
  if (qPi >= 43) return qPi - 6;
  const TAB = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37];
  return TAB[qPi - 30]!;
});

/** Number of context increments per last-sig-prefix bin (ctx = min(prefix>>2,3) + offset) */
export function sigCtxCtxInc(xc: number, yc: number, log2Tb: number, scanIdx: number): number {
  // spec Table 9-46: ctxIdxInc for last_significant_coefficient_{x,y}_prefix
  if (log2Tb === 2) {
    if (scanIdx === 0) return (xc + yc) <= 3 ? 0 : 1;
    if (scanIdx === 1) return yc ? 1 : 0;
    return xc ? 1 : 0;
  } else if (log2Tb === 3) {
    if (scanIdx === 0) {
      const a = (xc + yc) > 4 ? 2 : 0;
      if (xc + yc === 0) return a;
      return a + 1 + Math.min(3, (xc + yc) >> 1);
    }
    if (scanIdx === 1) return yc ? 3 + (yc >> 1) : 0;
    return xc ? 3 + (xc >> 1) : 0;
  } else {
    if (scanIdx === 0) {
      const a = (xc + yc) > 4 ? 2 : 0;
      if (xc + yc === 0) return a;
      return a + 1 + Math.min(3, (xc + yc) >> 1);
    }
    if (scanIdx === 1) return yc ? 3 + (yc - 2 >> 1) : 0;
    return xc ? 3 + (xc - 2 >> 1) : 0;
  }
}
