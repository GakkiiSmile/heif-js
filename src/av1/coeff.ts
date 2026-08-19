import { lo_ctx_offsets, skip_ctx } from './tables_data.ts';
import { coefficientScan, transformSizes, txTypeClass, txTypeFromUvMode, txTypesPerSet } from './tables.ts';
import type { MsacDecoder } from './msac.ts';

export interface CoefficientResult {
  eob: number;
  txType: number;
  coefficients: Int32Array;
  context: number;
}

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
  const coefficients = new Int32Array(width * height);
  const skipContext = getSkipContext(info, blockSize, above, left, !!chroma, subsamplingX, subsamplingY);
  const allSkip = msac.boolAdapt(coefCdf.skip[info.ctx][skipContext]);
  if (allSkip) {
    return { eob: -1, txType: lossless ? 16 : 0, coefficients, context: 0x40 };
  }

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

  const tokens = new Int32Array(width * height);
  // 2-D transforms use the transform height as their column-major stride.
  // AV1's 1-D coefficient contexts use a fixed padded stride of 16.
  const levelsStride = txClass === 0 ? height : 16;
  const levels = new Uint8Array(levelsStride * (Math.max(width, height) + 2) + 8);
  const scan = coefficientScan(tx);
  const eobBase = coefCdf.eob_base_tok[info.ctx][chroma];
  const base = coefCdf.base_tok[info.ctx][chroma];
  const high = coefCdf.br_tok[Math.min(info.ctx, 3)][chroma];
  const positions: { rc: number; scanIndex: number }[] = [];
  let dcToken = 0;

  const positionFor = (index: number): { rc: number; x: number; y: number; levelIndex: number } => {
    if (txClass === 0) {
      const rc = scan[index]!;
      return { rc, x: Math.floor(rc / height), y: rc % height, levelIndex: rc };
    }
    const mask = (txClass === 1 ? height : width) - 1;
    const x = index & mask, y = index >> Math.log2(mask + 1);
    return txClass === 1
      ? { rc: index, x, y, levelIndex: x * 16 + y }
      : { rc: x * height + y, x, y, levelIndex: x * 16 + y };
  };

  if (eob) {
    let context = 1 + +(eob > 2 << sizeContext) + +(eob > 4 << sizeContext);
    let token = msac.symbol(eobBase[context], 2) + 1;
    let p = positionFor(eob);
    if (token === 3) {
      context = (txClass === 0 ? ((p.x | p.y) > 1) : p.y !== 0) ? 14 : 7;
      token = msac.hiToken(high[context]);
      levels[p.levelIndex] = token + (3 << 6);
    } else {
      levels[p.levelIndex] = token * 0x41;
    }
    tokens[p.rc] = token;
    positions.push({ rc: p.rc, scanIndex: eob });

    const nonsquare = tx >= 5 ? 1 : 0;
    const offsetTable = lo_ctx_offsets[nonsquare + (tx & nonsquare)]!;
    for (let index = eob - 1; index > 0; index--) {
      p = positionFor(index);
      const lo = getLowContext(levels, levelsStride, txClass, offsetTable, p.x, p.y, p.levelIndex);
      token = msac.symbol(base[lo.context], 3);
      if (token === 3) {
        const axis = txClass === 0 ? p.x | p.y : p.y;
        const magnitude = lo.highMagnitude & 63;
        context = (axis > (txClass === 0 ? 1 : 0) ? 14 : 7) +
          (magnitude > 12 ? 6 : (magnitude + 1) >> 1);
        token = msac.hiToken(high[context]);
        levels[p.levelIndex] = token + (3 << 6);
      } else {
        levels[p.levelIndex] = token * 0x41;
      }
      if (token) {
        tokens[p.rc] = token;
        positions.push({ rc: p.rc, scanIndex: index });
      }
    }

    const dcContext = txClass === 0 ? 0 :
      getLowContext(levels, levelsStride, txClass, offsetTable, 0, 0, 0).context;
    dcToken = msac.symbol(base[dcContext], 3);
    if (dcToken === 3) {
      let magnitude: number;
      if (txClass === 0) magnitude = (levels[1]! + levels[levelsStride]! + levels[levelsStride + 1]!) & 63;
      else magnitude = getLowContext(levels, levelsStride, txClass, offsetTable, 0, 0, 0).highMagnitude & 63;
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
  positions.sort((a, b) => a.scanIndex - b.scanIndex);
  for (const position of positions) {
    let token = tokens[position.rc]!;
    const sign = msac.boolEqui();
    if (token === 15) {
      token = readGolomb(msac) + 15;
    }
    coefficients[position.rc] = sign ? -token : token;
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
    const ca = +above.subarray(0, 1 << info.logW).some(value => value !== 0x40);
    const cl = +left.subarray(0, 1 << info.logH).some(value => value !== 0x40);
    return 7 + +notOneBlock * 3 + ca + cl;
  }
  if (dimensions[2] === info.logW && dimensions[3] === info.logH) return 0;
  let a = 0, l = 0;
  for (const value of above.subarray(0, info.w4)) a |= value;
  for (const value of left.subarray(0, info.h4)) l |= value;
  return skip_ctx[Math.min(a & 0x3f, 4)]![Math.min(l & 0x3f, 4)]!;
}

function getLowContext(levels: Uint8Array, stride: number, txClass: number,
  offsets: number[][], x: number, y: number, index: number): { context: number; highMagnitude: number } {
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
  return { context: offset + (magnitude > 512 ? 4 : (magnitude + 64) >> 7), highMagnitude };
}

function getDcSignContext(above: Uint8Array, left: Uint8Array, width: number, height: number): number {
  let sum = -width - height;
  for (const value of above.subarray(0, width)) sum += value >> 6;
  for (const value of left.subarray(0, height)) sum += value >> 6;
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
