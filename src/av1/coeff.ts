import { lo_ctx_offsets, skip_ctx } from './tables_data.ts';
import { coefficientScan, transformSizes, txTypeClass, txTypeFromUvMode, txTypesPerSet } from './tables.ts';
import type { MsacDecoder } from './msac.ts';

export interface CoefficientResult {
  eob: number;
  txType: number;
  coefficients: Int32Array;
  context: number;
}

// Skipped transforms never read coefficient storage during reconstruction.
// Sharing one empty view avoids allocating and retaining a zero-filled block for
// every all-skip transform in sparse images.
const EMPTY_COEFFICIENTS = new Int32Array(0);
const TOKEN_SCRATCH = new Int32Array(32 * 32);
const LEVEL_SCRATCH = new Uint8Array(32 * 34 + 8);
const POSITION_SCRATCH = new Uint16Array(32 * 32);

interface CoefficientOptions {
  msac: MsacDecoder;
  modeCdf: Record<string, any>;
  coefCdf: Record<string, any>;
  tx: number;
  blockSize: number;
  plane: number;
  intra: boolean;
  yMode: number;
  uvMode: number;
  reducedTransformSet: boolean;
  qIdx: number;
  lossless?: boolean;
  subsamplingX?: number;
  subsamplingY?: number;
  lumaTxType?: number;
  above: Uint8Array;
  left: Uint8Array;
}

export function decodeCoefficients(options: CoefficientOptions): CoefficientResult {
  const { msac, modeCdf, coefCdf, tx, blockSize, plane, intra, yMode, uvMode,
    reducedTransformSet, qIdx, lossless = false, subsamplingX = 1, subsamplingY = 1,
    lumaTxType = 0, above, left } = options;
  const chroma = plane !== 0 ? 1 : 0;
  const info = transformSizes[tx]!;
  const width = Math.min(info.w4, 8) * 4;
  const height = Math.min(info.h4, 8) * 4;
  const skipContext = getSkipContext(info, blockSize, above, left, !!chroma, subsamplingX, subsamplingY);
  const allSkip = msac.boolAdapt(coefCdf.skip[info.ctx][skipContext]);
  if (allSkip) {
    return { eob: -1, txType: lossless ? 16 : 0, coefficients: EMPTY_COEFFICIENTS, context: 0x40 };
  }
  const coefficients = new Int32Array(width * height);

  let txType: number;
  if (lossless) txType = 16;
  else if (info.max + +intra >= 4) txType = 0;
  else if (chroma) txType = intra ? (txTypeFromUvMode[uvMode] ?? 0) : getUvInterTxType(info, lumaTxType);
  else if (!qIdx) txType = 0;
  else if (intra) {
    if (reducedTransformSet || info.min === 2) {
      const index = msac.symbol(modeCdf.txtp_intra2[info.min][yMode], 4);
      txType = txTypesPerSet[index]!;
    } else {
      const index = msac.symbol(modeCdf.txtp_intra1[info.min][yMode], 6);
      txType = txTypesPerSet[index + 5]!;
    }
  } else {
    if (reducedTransformSet || info.max === 3) {
      const index = msac.boolAdapt(modeCdf.txtp_inter3[info.min]);
      txType = index ? 0 : 9;
    } else if (info.min === 2) {
      const index = msac.symbol(modeCdf.txtp_inter2, 11);
      txType = txTypesPerSet[index + 12]!;
    } else {
      const index = msac.symbol(modeCdf.txtp_inter1[info.min], 15);
      txType = txTypesPerSet[index + 24]!;
    }
  }

  const slw = Math.min(info.logW, 3), slh = Math.min(info.logH, 3);
  const sizeContext = slw + slh;
  const txClass = txTypeClass[txType]!;
  const is1d = txClass !== 0 ? 1 : 0;
  const eobTable = coefCdf[`eob_bin_${16 << sizeContext}`];
  const eobCdf = sizeContext <= 4 ? eobTable[chroma][is1d] : eobTable[chroma];
  let eob = msac.symbol(eobCdf, 4 + sizeContext);
  if (eob > 1) {
    const eobBin = eob - 2;
    const high = msac.boolAdapt(coefCdf.eob_hi_bit[info.ctx][chroma][eobBin]);
    eob = ((high | 2) << eobBin) | msac.bools(eobBin);
  }

  const tokens = TOKEN_SCRATCH;
  // 2-D transforms use the transform height as their column-major stride.
  // AV1's 1-D coefficient contexts use a fixed padded stride of 16.
  const levelsStride = txClass === 0 ? height : 16;
  const levelsLength = levelsStride * (Math.max(width, height) + 2) + 8;
  const levels = LEVEL_SCRATCH;
  levels.fill(0, 0, levelsLength);
  const scan = coefficientScan(tx);
  const eobBase = coefCdf.eob_base_tok[info.ctx][chroma];
  const base = coefCdf.base_tok[info.ctx][chroma];
  const high = coefCdf.br_tok[Math.min(info.ctx, 3)][chroma];
  const positions = POSITION_SCRATCH;
  let positionCount = 0;
  let dcToken = 0;
  const classMask = (txClass === 1 ? height : width) - 1;
  const classShift = Math.log2(classMask + 1);

  if (eob) {
    let context = 1 + +(eob > 2 << sizeContext) + +(eob > 4 << sizeContext);
    let token = msac.symbol(eobBase[context], 2) + 1;
    let rc: number, x: number, y: number, levelIndex: number;
    if (txClass === 0) {
      rc = scan[eob]!; x = Math.floor(rc / height); y = rc % height; levelIndex = rc;
    } else {
      x = eob & classMask; y = eob >> classShift;
      rc = txClass === 1 ? eob : x * height + y;
      levelIndex = x * 16 + y;
    }
    if (token === 3) {
      context = (txClass === 0 ? ((x | y) > 1) : y !== 0) ? 14 : 7;
      token = msac.hiToken(high[context]);
      levels[levelIndex] = token + (3 << 6);
    } else {
      levels[levelIndex] = token * 0x41;
    }
    tokens[rc] = token;
    positions[positionCount++] = rc;

    const nonsquare = tx >= 5 ? 1 : 0;
    const offsetTable = lo_ctx_offsets[nonsquare + (tx & nonsquare)]!;
    for (let index = eob - 1; index > 0; index--) {
      if (txClass === 0) {
        rc = scan[index]!; x = Math.floor(rc / height); y = rc % height; levelIndex = rc;
      } else {
        x = index & classMask; y = index >> classShift;
        rc = txClass === 1 ? index : x * height + y;
        levelIndex = x * 16 + y;
      }
      const low = getLowContext(levels, levelsStride, txClass, offsetTable, x, y, levelIndex);
      token = msac.symbol(base[low & 0xff], 3);
      if (token === 3) {
        const axis = txClass === 0 ? x | y : y;
        const magnitude = (low >>> 8) & 63;
        context = (axis > (txClass === 0 ? 1 : 0) ? 14 : 7) +
          (magnitude > 12 ? 6 : (magnitude + 1) >> 1);
        token = msac.hiToken(high[context]);
        levels[levelIndex] = token + (3 << 6);
      } else {
        levels[levelIndex] = token * 0x41;
      }
      if (token) {
        tokens[rc] = token;
        positions[positionCount++] = rc;
      }
    }

    const dcContext = txClass === 0 ? 0 :
      getLowContext(levels, levelsStride, txClass, offsetTable, 0, 0, 0) & 0xff;
    dcToken = msac.symbol(base[dcContext], 3);
    if (dcToken === 3) {
      let magnitude: number;
      if (txClass === 0) magnitude = (levels[1]! + levels[levelsStride]! + levels[levelsStride + 1]!) & 63;
      else magnitude = (getLowContext(levels, levelsStride, txClass, offsetTable, 0, 0, 0) >>> 8) & 63;
      const context = magnitude > 12 ? 6 : (magnitude + 1) >> 1;
      dcToken = msac.hiToken(high[context]);
    }
  } else {
    const branch = msac.symbol(eobBase[0], 2);
    dcToken = 1 + branch;
    if (branch === 2) dcToken = msac.hiToken(high[0]);
  }

  let cumulativeLevel = 0;
  let dcSignLevel = 1 << 6;
  if (dcToken) {
    const dcSignContext = getDcSignContext(above, left, info.w4, info.h4);
    const sign = msac.boolAdapt(coefCdf.dc_sign[chroma][dcSignContext]);
    if (dcToken === 15) {
      dcToken = readGolomb(msac) + 15;
    }
    coefficients[0] = sign ? -dcToken : dcToken;
    cumulativeLevel = dcToken;
    dcSignLevel = (sign - 1) & (2 << 6);
  }
  // Positions were discovered from the end of the scan towards DC. Read sign
  // bits in normative ascending scan order by walking the compact buffer back.
  for (let position = positionCount - 1; position >= 0; position--) {
    const rc = positions[position]!;
    let token = tokens[rc]!;
    const sign = msac.boolEqui();
    if (token === 15) {
      token = readGolomb(msac) + 15;
    }
    coefficients[rc] = sign ? -token : token;
    cumulativeLevel += token;
  }

  return {
    eob,
    txType,
    coefficients,
    context: Math.min(cumulativeLevel, 63) | dcSignLevel,
  };
}

function getUvInterTxType(info: typeof transformSizes[number], lumaTxType: number): number {
  if (info.max === 3) return lumaTxType === 9 ? 9 : 0;
  if (info.min === 2 &&
      (lumaTxType === 12 || lumaTxType === 13 || lumaTxType === 14 || lumaTxType === 15)) {
    return 0;
  }
  return lumaTxType;
}

function getSkipContext(info: typeof transformSizes[number], blockSize: number,
  above: Uint8Array, left: Uint8Array, chroma: boolean, subsamplingX: number, subsamplingY: number): number {
  const dimensions = BLOCK_DIMENSIONS[blockSize]!;
  if (chroma) {
    const notOneBlock = dimensions[2] - +(!!dimensions[2] && !!subsamplingX) > info.logW ||
      dimensions[3] - +(!!dimensions[3] && !!subsamplingY) > info.logH;
    let ca = 0, cl = 0;
    const aboveLength = Math.min(above.length, 1 << info.logW);
    const leftLength = Math.min(left.length, 1 << info.logH);
    for (let i = 0; i < aboveLength; i++) ca ||= +(above[i] !== 0x40);
    for (let i = 0; i < leftLength; i++) cl ||= +(left[i] !== 0x40);
    return 7 + +notOneBlock * 3 + ca + cl;
  }
  if (dimensions[2] === info.logW && dimensions[3] === info.logH) return 0;
  let a = 0, l = 0;
  for (let i = 0; i < info.w4; i++) a |= above[i]!;
  for (let i = 0; i < info.h4; i++) l |= left[i]!;
  return skip_ctx[Math.min(a & 0x3f, 4)]![Math.min(l & 0x3f, 4)]!;
}

function getLowContext(levels: Uint8Array, stride: number, txClass: number,
  offsets: number[][], x: number, y: number, index: number): number {
  let magnitude = levels[index + 1]! + levels[index + stride]!;
  let highMagnitude = magnitude;
  let offset: number;
  if (txClass === 0) {
    magnitude += levels[index + stride + 1]!;
    highMagnitude = magnitude;
    magnitude += levels[index + 2]! + levels[index + 2 * stride]!;
    offset = offsets[Math.min(y, 4)]![Math.min(x, 4)]!;
  } else {
    magnitude += levels[index + 2]!;
    highMagnitude = magnitude;
    magnitude += levels[index + 3]! + levels[index + 4]!;
    offset = 26 + (y > 1 ? 10 : y * 5);
  }
  return (offset + (magnitude > 512 ? 4 : (magnitude + 64) >> 7)) | (highMagnitude << 8);
}

function getDcSignContext(above: Uint8Array, left: Uint8Array, width: number, height: number): number {
  let sum = -width - height;
  for (let i = 0; i < width; i++) sum += above[i]! >> 6;
  for (let i = 0; i < height; i++) sum += left[i]! >> 6;
  return +(sum !== 0) + +(sum > 0);
}

function readGolomb(msac: MsacDecoder): number {
  let length = 0;
  let value = 1;
  while (!msac.boolEqui() && length < 32) length++;
  for (let i = 0; i < length; i++) value = value * 2 + msac.boolEqui();
  return value - 1;
}

const BLOCK_DIMENSIONS = [
  [32, 32, 5, 5], [32, 16, 5, 4], [16, 32, 4, 5], [16, 16, 4, 4],
  [16, 8, 4, 3], [16, 4, 4, 2], [8, 16, 3, 4], [8, 8, 3, 3],
  [8, 4, 3, 2], [8, 2, 3, 1], [4, 16, 2, 4], [4, 8, 2, 3],
  [4, 4, 2, 2], [4, 2, 2, 1], [4, 1, 2, 0], [2, 8, 1, 3],
  [2, 4, 1, 2], [2, 2, 1, 1], [2, 1, 1, 0], [1, 4, 0, 2],
  [1, 2, 0, 1], [1, 1, 0, 0],
] as const;
