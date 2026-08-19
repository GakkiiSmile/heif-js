/** Bit-exact AV1 inverse transforms (AV1 specification section 7.13). */

type OneDimensionalType = 0 | 1 | 2 | 3; // DCT, ADST, FLIPADST, IDENTITY

const COS128 = new Int16Array([
  4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036,
  4017, 3996, 3973, 3948, 3920, 3889, 3857, 3822,
  3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461,
  3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967,
  2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359,
  2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
  1567, 1474, 1380, 1285, 1189, 1092, 995, 897,
  799, 700, 601, 501, 401, 301, 201, 101, 0,
]);
const COS256 = new Int16Array(256);
for (let angle = 0; angle < 256; angle++) {
  COS256[angle] = angle <= 64 ? COS128[angle]! :
    angle <= 128 ? -COS128[128 - angle]! :
    angle <= 192 ? -COS128[angle - 128]! : COS128[256 - angle]!;
}
const POWERS_OF_TWO = new Float64Array(32);
const CLAMP_MINIMUM = new Float64Array(32);
const CLAMP_MAXIMUM = new Float64Array(32);
for (let bits = 0; bits < 32; bits++) {
  const power = 2 ** bits;
  POWERS_OF_TWO[bits] = power;
  CLAMP_MINIMUM[bits] = -(power / 2);
  CLAMP_MAXIMUM[bits] = power / 2 - 1;
}

const TX_1D_TYPES: readonly (readonly [OneDimensionalType, OneDimensionalType])[] = [
  [0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [0, 2], [2, 2], [1, 2],
  [2, 1], [3, 3], [0, 3], [3, 0], [1, 3], [3, 1], [2, 3], [3, 2],
];

const TRANSFORM_ROW_SHIFT = [0, 1, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2];

// AV1 transform dimensions are at most 64 and the transform is synchronous
// with no user callbacks, so these internal work buffers can be reused safely
// across calls. Returned output storage remains caller-owned.
const MAX_TRANSFORM_LENGTH = 64;
const TRANSFORM_INTERMEDIATE = new Int32Array(MAX_TRANSFORM_LENGTH * MAX_TRANSFORM_LENGTH);
const TRANSFORM_LINE = new Int32Array(MAX_TRANSFORM_LENGTH);
const TRANSFORM_PERMUTATION = new Int32Array(MAX_TRANSFORM_LENGTH);
const TRANSFORM_LINE_VIEWS: (Int32Array | undefined)[] = [];

function transformLine(length: number): Int32Array {
  let view = TRANSFORM_LINE_VIEWS[length];
  if (!view) {
    view = TRANSFORM_LINE.subarray(0, length);
    TRANSFORM_LINE_VIEWS[length] = view;
  }
  return view;
}

/** Apply the normative separable inverse transform to row-major dequantized coefficients. */
export function inverseTransform2d(
  coefficients: Int32Array, width: number, height: number,
  txSize: number, txType: number, bitDepth: number,
  output: Int32Array = new Int32Array(width * height),
): Int32Array {
  const area = width * height;
  if (output.length < area) throw new RangeError('AV1 inverse-transform output is too small');
  if (txType === 16) return inverseWht2d(coefficients, output);
  const types = TX_1D_TYPES[txType] ?? TX_1D_TYPES[0]!;
  const verticalType = types[0], horizontalType = types[1];
  const rowClampRange = bitDepth + 8;
  const columnClampRange = Math.max(bitDepth + 6, 16);
  const columnMinimum = -(2 ** (columnClampRange - 1));
  const columnMaximum = 2 ** (columnClampRange - 1) - 1;
  const rowShift = TRANSFORM_ROW_SHIFT[txSize] ?? 0;
  const rectangular = Math.abs(Math.log2(width) - Math.log2(height)) === 1;
  const maximumLength = Math.max(width, height);
  const standardSize = area <= TRANSFORM_INTERMEDIATE.length && maximumLength <= MAX_TRANSFORM_LENGTH;
  const intermediate = standardSize ? TRANSFORM_INTERMEDIATE : new Int32Array(area);
  const permutation = standardSize ? TRANSFORM_PERMUTATION : new Int32Array(maximumLength);
  const row = standardSize ? transformLine(width) : new Int32Array(width);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let value = coefficients[y * width + x]!;
      if (rectangular) value = round2(value * 2896, 12);
      row[x] = value;
    }
    inverse1d(row, horizontalType, rowClampRange, permutation);
    for (let x = 0; x < width; x++) {
      intermediate[y * width + x] = clip(round2(row[x]!, rowShift), columnMinimum, columnMaximum);
    }
  }

  const column = standardSize ? transformLine(height) : new Int32Array(height);
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) column[y] = intermediate[y * width + x]!;
    inverse1d(column, verticalType, columnClampRange, permutation);
    for (let y = 0; y < height; y++) output[y * width + x] = round2(column[y]!, 4);
  }
  return output;
}

function inverse1d(
  values: Int32Array, type: OneDimensionalType, clampRange: number, permutation: Int32Array,
): void {
  if (type === 0) inverseDct(values, clampRange, permutation);
  else if (type === 1 || type === 2) {
    inverseAdst(values, clampRange, permutation);
    if (type === 2) values.reverse();
  } else inverseIdentity(values);
}

function inverseDct(values: Int32Array, clampRange: number, permutation: Int32Array): void {
  const n = Math.log2(values.length);
  permutation.set(values, 0);
  for (let i = 0; i < values.length; i++) values[i] = permutation[bitReverse(n, i)]!;

  if (n === 6) for (let i = 0; i <= 15; i++) {
    butterfly(values, 32 + i, 63 - i, 63 - 4 * bitReverse(4, i), false);
  }
  if (n >= 5) for (let i = 0; i <= 7; i++) {
    butterfly(values, 16 + i, 31 - i, 6 + (bitReverse(3, 7 - i) << 3), false);
  }
  if (n === 6) for (let i = 0; i <= 15; i++) {
    hadamard(values, 32 + i * 2, 33 + i * 2, !!(i & 1), clampRange);
  }
  if (n >= 4) for (let i = 0; i <= 3; i++) {
    butterfly(values, 8 + i, 15 - i, 12 + (bitReverse(2, 3 - i) << 4), false);
  }
  if (n >= 5) for (let i = 0; i <= 7; i++) {
    hadamard(values, 16 + 2 * i, 17 + 2 * i, !!(i & 1), clampRange);
  }
  if (n === 6) for (let i = 0; i <= 3; i++) for (let j = 0; j <= 1; j++) {
    butterfly(values, 62 - i * 4 - j, 33 + i * 4 + j,
      60 - 16 * bitReverse(2, i) + 64 * j, true);
  }
  if (n >= 3) for (let i = 0; i <= 1; i++) {
    butterfly(values, 4 + i, 7 - i, 56 - 32 * i, false);
  }
  if (n >= 4) for (let i = 0; i <= 3; i++) {
    hadamard(values, 8 + 2 * i, 9 + 2 * i, !!(i & 1), clampRange);
  }
  if (n >= 5) for (let i = 0; i <= 1; i++) for (let j = 0; j <= 1; j++) {
    butterfly(values, 30 - 4 * i - j, 17 + 4 * i + j,
      24 + (j << 6) + ((1 - i) << 5), true);
  }
  if (n === 6) for (let i = 0; i <= 7; i++) for (let j = 0; j <= 1; j++) {
    hadamard(values, 32 + i * 4 + j, 35 + i * 4 - j, !!(i & 1), clampRange);
  }
  for (let i = 0; i <= 1; i++) {
    butterfly(values, 2 * i, 2 * i + 1, 32 + 16 * i, i === 0);
  }
  if (n >= 3) for (let i = 0; i <= 1; i++) {
    hadamard(values, 4 + 2 * i, 5 + 2 * i, !!i, clampRange);
  }
  if (n >= 4) for (let i = 0; i <= 1; i++) {
    butterfly(values, 14 - i, 9 + i, 48 + 64 * i, true);
  }
  if (n >= 5) for (let i = 0; i <= 3; i++) for (let j = 0; j <= 1; j++) {
    hadamard(values, 16 + 4 * i + j, 19 + 4 * i - j, !!(i & 1), clampRange);
  }
  if (n === 6) for (let i = 0; i <= 1; i++) for (let j = 0; j <= 3; j++) {
    butterfly(values, 61 - i * 8 - j, 34 + i * 8 + j,
      56 - i * 32 + (j >> 1) * 64, true);
  }
  for (let i = 0; i <= 1; i++) hadamard(values, i, 3 - i, false, clampRange);
  if (n >= 3) butterfly(values, 6, 5, 32, true);
  if (n >= 4) for (let i = 0; i <= 1; i++) for (let j = 0; j <= 1; j++) {
    hadamard(values, 8 + 4 * i + j, 11 + 4 * i - j, !!i, clampRange);
  }
  if (n >= 5) for (let i = 0; i <= 3; i++) {
    butterfly(values, 29 - i, 18 + i, 48 + (i >> 1) * 64, true);
  }
  if (n === 6) for (let i = 0; i <= 3; i++) for (let j = 0; j <= 3; j++) {
    hadamard(values, 32 + 8 * i + j, 39 + 8 * i - j, !!(i & 1), clampRange);
  }
  if (n >= 3) for (let i = 0; i <= 3; i++) hadamard(values, i, 7 - i, false, clampRange);
  if (n >= 4) for (let i = 0; i <= 1; i++) butterfly(values, 13 - i, 10 + i, 32, true);
  if (n >= 5) for (let i = 0; i <= 1; i++) for (let j = 0; j <= 3; j++) {
    hadamard(values, 16 + i * 8 + j, 23 + i * 8 - j, !!i, clampRange);
  }
  if (n === 6) for (let i = 0; i <= 7; i++) {
    butterfly(values, 59 - i, 36 + i, i < 4 ? 48 : 112, true);
  }
  if (n >= 4) for (let i = 0; i <= 7; i++) hadamard(values, i, 15 - i, false, clampRange);
  if (n >= 5) for (let i = 0; i <= 3; i++) butterfly(values, 27 - i, 20 + i, 32, true);
  if (n === 6) for (let i = 0; i <= 7; i++) {
    hadamard(values, 32 + i, 47 - i, false, clampRange);
    hadamard(values, 48 + i, 63 - i, true, clampRange);
  }
  if (n >= 5) for (let i = 0; i <= 15; i++) hadamard(values, i, 31 - i, false, clampRange);
  if (n === 6) for (let i = 0; i <= 7; i++) butterfly(values, 55 - i, 40 + i, 32, true);
  if (n === 6) for (let i = 0; i <= 31; i++) hadamard(values, i, 63 - i, false, clampRange);
}

function inverseAdst(values: Int32Array, clampRange: number, permutation: Int32Array): void {
  if (values.length === 4) {
    inverseAdst4(values);
    return;
  }
  const n = Math.log2(values.length);
  adstInputPermutation(values, permutation);
  if (n === 3) {
    for (let i = 0; i <= 3; i++) butterfly(values, 2 * i, 2 * i + 1, 60 - 16 * i, true);
    for (let i = 0; i <= 3; i++) hadamard(values, i, 4 + i, false, clampRange);
    for (let i = 0; i <= 1; i++) butterfly(values, 4 + 3 * i, 5 + i, 48 - 32 * i, true);
    for (let i = 0; i <= 1; i++) for (let j = 0; j <= 1; j++) {
      hadamard(values, 4 * j + i, 2 + 4 * j + i, false, clampRange);
    }
    for (let i = 0; i <= 1; i++) butterfly(values, 2 + 4 * i, 3 + 4 * i, 32, true);
  } else {
    for (let i = 0; i <= 7; i++) butterfly(values, 2 * i, 2 * i + 1, 62 - 8 * i, true);
    for (let i = 0; i <= 7; i++) hadamard(values, i, 8 + i, false, clampRange);
    for (let i = 0; i <= 1; i++) {
      butterfly(values, 8 + 2 * i, 9 + 2 * i, 56 - 32 * i, true);
      butterfly(values, 13 + 2 * i, 12 + 2 * i, 8 + 32 * i, true);
    }
    for (let i = 0; i <= 3; i++) for (let j = 0; j <= 1; j++) {
      hadamard(values, 8 * j + i, 4 + 8 * j + i, false, clampRange);
    }
    for (let i = 0; i <= 1; i++) for (let j = 0; j <= 1; j++) {
      butterfly(values, 4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i, true);
    }
    for (let i = 0; i <= 1; i++) for (let j = 0; j <= 3; j++) {
      hadamard(values, 4 * j + i, 2 + 4 * j + i, false, clampRange);
    }
    for (let i = 0; i <= 3; i++) butterfly(values, 2 + 4 * i, 3 + 4 * i, 32, true);
  }
  adstOutputPermutation(values, permutation);
}

function inverseAdst4(values: Int32Array): void {
  const t0 = values[0]!, t1 = values[1]!, t2 = values[2]!, t3 = values[3]!;
  let s0 = 1321 * t0;
  let s1 = 2482 * t0;
  const s3 = 3344 * t1;
  const b7 = t0 - t2 + t3;
  s0 += 3803 * t2;
  s1 -= 1321 * t2;
  const s2 = 3344 * b7;
  s0 += 2482 * t3;
  s1 -= 3803 * t3;
  const x0 = s0 + s3;
  const x1 = s1 + s3;
  const x2 = s2;
  const x3 = s0 + s1 - s3;
  values[0] = round2(x0, 12);
  values[1] = round2(x1, 12);
  values[2] = round2(x2, 12);
  values[3] = round2(x3, 12);
}

function adstInputPermutation(values: Int32Array, permutation: Int32Array): void {
  const length = values.length;
  permutation.set(values, 0);
  for (let i = 0; i < length; i++) values[i] = permutation[(i & 1) ? i - 1 : length - i - 1]!;
}

function adstOutputPermutation(values: Int32Array, permutation: Int32Array): void {
  const n = Math.log2(values.length);
  permutation.set(values, 0);
  for (let i = 0; i < values.length; i++) {
    const a = (i >> 3) & 1;
    const b = ((i >> 2) & 1) ^ ((i >> 3) & 1);
    const c = ((i >> 1) & 1) ^ ((i >> 2) & 1);
    const d = (i & 1) ^ ((i >> 1) & 1);
    const index = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n);
    values[i] = (i & 1) ? -permutation[index]! : permutation[index]!;
  }
}

function inverseIdentity(values: Int32Array): void {
  if (values.length === 4) for (let i = 0; i < 4; i++) values[i] = round2(values[i]! * 5793, 12);
  else if (values.length === 8) for (let i = 0; i < 8; i++) values[i] = values[i]! * 2;
  else if (values.length === 16) for (let i = 0; i < 16; i++) values[i] = round2(values[i]! * 11586, 12);
  else for (let i = 0; i < values.length; i++) values[i] = values[i]! * 4;
}

function inverseWht2d(coefficients: Int32Array, output: Int32Array): Int32Array {
  const intermediate = TRANSFORM_INTERMEDIATE;
  const row = transformLine(4);
  for (let y = 0; y < 4; y++) {
    for (let x = 0; x < 4; x++) row[x] = coefficients[y * 4 + x]!;
    inverseWht1d(row, 2);
    intermediate.set(row, y * 4);
  }
  const column = row;
  for (let x = 0; x < 4; x++) {
    for (let y = 0; y < 4; y++) column[y] = intermediate[y * 4 + x]!;
    inverseWht1d(column, 0);
    for (let y = 0; y < 4; y++) output[y * 4 + x] = column[y]!;
  }
  return output;
}

function inverseWht1d(values: Int32Array, shift: number): void {
  let a = floorShift(values[0]!, shift);
  let c = floorShift(values[1]!, shift);
  let d = floorShift(values[2]!, shift);
  let b = floorShift(values[3]!, shift);
  a += c;
  d -= b;
  const e = floorShift(a - d, 1);
  b = e - b;
  c = e - c;
  a -= b;
  d += c;
  values[0] = a; values[1] = b; values[2] = c; values[3] = d;
}

function butterfly(values: Int32Array, a: number, b: number, angle: number, flip: boolean): void {
  const first = values[a]!, second = values[b]!;
  const cosine = COS256[angle & 255]!, sine = COS256[(angle - 64) & 255]!;
  const x = round2(first * cosine - second * sine, 12);
  const y = round2(first * sine + second * cosine, 12);
  if (flip) { values[a] = y; values[b] = x; }
  else { values[a] = x; values[b] = y; }
}

function hadamard(values: Int32Array, a: number, b: number, flip: boolean, range: number): void {
  if (flip) { const temporary = a; a = b; b = temporary; }
  const first = values[a]!, second = values[b]!;
  const minimum = CLAMP_MINIMUM[range]!, maximum = CLAMP_MAXIMUM[range]!;
  values[a] = clip(first + second, minimum, maximum);
  values[b] = clip(first - second, minimum, maximum);
}

function bitReverse(bits: number, value: number): number {
  let output = 0;
  for (let bit = 0; bit < bits; bit++) output |= ((value >> bit) & 1) << (bits - 1 - bit);
  return output;
}

function round2(value: number, bits: number): number {
  const divisor = POWERS_OF_TWO[bits]!;
  return bits ? Math.floor((value + divisor / 2) / divisor) : value;
}

function floorShift(value: number, bits: number): number {
  return bits ? Math.floor(value / POWERS_OF_TWO[bits]!) : value;
}

function clip(value: number, minimum: number, maximum: number): number {
  return value < minimum ? minimum : value > maximum ? maximum : value;
}
