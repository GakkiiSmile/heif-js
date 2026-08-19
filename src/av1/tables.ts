export interface TransformSize {
  w4: number; h4: number; logW: number; logH: number;
  min: number; max: number; sub: number; ctx: number;
}

export const TX_4X4 = 0, TX_8X8 = 1, TX_16X16 = 2, TX_32X32 = 3, TX_64X64 = 4;
export const RTX_4X8 = 5, RTX_8X4 = 6, RTX_8X16 = 7, RTX_16X8 = 8;
export const RTX_16X32 = 9, RTX_32X16 = 10, RTX_32X64 = 11, RTX_64X32 = 12;
export const RTX_4X16 = 13, RTX_16X4 = 14, RTX_8X32 = 15, RTX_32X8 = 16;
export const RTX_16X64 = 17, RTX_64X16 = 18;

export const transformSizes: TransformSize[] = [
  { w4: 1, h4: 1, logW: 0, logH: 0, min: 0, max: 0, sub: 0, ctx: 0 },
  { w4: 2, h4: 2, logW: 1, logH: 1, min: 1, max: 1, sub: 0, ctx: 1 },
  { w4: 4, h4: 4, logW: 2, logH: 2, min: 2, max: 2, sub: 1, ctx: 2 },
  { w4: 8, h4: 8, logW: 3, logH: 3, min: 3, max: 3, sub: 2, ctx: 3 },
  { w4: 16, h4: 16, logW: 4, logH: 4, min: 4, max: 4, sub: 3, ctx: 4 },
  { w4: 1, h4: 2, logW: 0, logH: 1, min: 0, max: 1, sub: 0, ctx: 1 },
  { w4: 2, h4: 1, logW: 1, logH: 0, min: 0, max: 1, sub: 0, ctx: 1 },
  { w4: 2, h4: 4, logW: 1, logH: 2, min: 1, max: 2, sub: 1, ctx: 2 },
  { w4: 4, h4: 2, logW: 2, logH: 1, min: 1, max: 2, sub: 1, ctx: 2 },
  { w4: 4, h4: 8, logW: 2, logH: 3, min: 2, max: 3, sub: 2, ctx: 3 },
  { w4: 8, h4: 4, logW: 3, logH: 2, min: 2, max: 3, sub: 2, ctx: 3 },
  { w4: 8, h4: 16, logW: 3, logH: 4, min: 3, max: 4, sub: 3, ctx: 4 },
  { w4: 16, h4: 8, logW: 4, logH: 3, min: 3, max: 4, sub: 3, ctx: 4 },
  { w4: 1, h4: 4, logW: 0, logH: 2, min: 0, max: 2, sub: 5, ctx: 1 },
  { w4: 4, h4: 1, logW: 2, logH: 0, min: 0, max: 2, sub: 6, ctx: 1 },
  { w4: 2, h4: 8, logW: 1, logH: 3, min: 1, max: 3, sub: 7, ctx: 2 },
  { w4: 8, h4: 2, logW: 3, logH: 1, min: 1, max: 3, sub: 8, ctx: 2 },
  { w4: 4, h4: 16, logW: 2, logH: 4, min: 2, max: 4, sub: 9, ctx: 3 },
  { w4: 16, h4: 4, logW: 4, logH: 2, min: 2, max: 4, sub: 10, ctx: 3 },
];

// Layout column 1 from dav1d_max_txfm_size_for_bs (4:2:0).
export const maxTransform420 = [
  3, 3, 3, 3, 10, 16, 9, 2, 8, 14, 15, 7, 1, 6, 6, 13, 5, 0, 0, 5, 0, 0,
];

export const maxTransform422 = [
  3, 3, 0, 3, 3, 10, 0, 9, 2, 7, 0, 0, 7, 1, 5, 0, 0, 5, 0, 0, 0, 0,
];

export const maxTransform444 = [
  3, 3, 3, 3, 3, 10, 3, 3, 10, 16, 9, 9, 2, 8, 14, 15, 7, 1, 6, 13, 5, 0,
];

// Luma column from dav1d_max_txfm_size_for_bs.
export const maxTransformLuma = [
  4, 4, 4, 4, 12, 18, 11, 3, 10, 16, 17, 9, 2, 8, 14, 15, 7, 1, 6, 13, 5, 0,
];

export const partitionTypeCount = [7, 9, 9, 9, 3];

// [block level][partition] -> [first block size, optional second block size].
export const partitionBlockSizes: number[][][] = [
  [[0], [1], [2], [0], [3, 1], [1, 3], [3, 2], [2, 3]],
  [[3], [4], [6], [3], [7, 4], [4, 7], [7, 6], [6, 7], [5], [10]],
  [[7], [8], [11], [7], [12, 8], [8, 12], [12, 11], [11, 12], [9], [15]],
  [[12], [13], [16], [12], [17, 13], [13, 17], [17, 16], [16, 17], [14], [19]],
  [[17], [18], [20], [21]],
];

export const txTypesPerSet = [
  9, 0, 3, 1, 2,
  9, 0, 10, 11, 3, 1, 2,
  9, 10, 11, 0, 1, 2, 4, 5, 3, 6, 7, 8,
  9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 4, 5, 3, 6, 7, 8,
];

export const txTypeClass = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 0];
export const txTypeFromUvMode = [0, 1, 2, 0, 3, 1, 2, 2, 1, 3, 1, 2, 3];

export const cflAllowed = new Set([7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]);

/** Normative AV1 coefficient scan in the column-major representation used by dav1d. */
export function coefficientScan(tx: number): Uint16Array {
  const info = transformSizes[tx]!;
  const width = Math.min(32, info.w4 * 4);
  const height = Math.min(32, info.h4 * 4);
  const result: number[] = [];
  for (let diagonal = 0; diagonal < width + height - 1; diagonal++) {
    const minX = Math.max(0, diagonal - height + 1);
    const maxX = Math.min(width - 1, diagonal);
    const ascending = width > height || width === height && !(diagonal & 1);
    if (ascending) {
      for (let x = minX; x <= maxX; x++) result.push(x * height + diagonal - x);
    } else {
      for (let x = maxX; x >= minX; x--) result.push(x * height + diagonal - x);
    }
  }
  return new Uint16Array(result);
}
