/**
 * HEVC CABAC arithmetic decoder engine (H.265 §9.3).
 * Bit-reading past the end of the slice returns 0 (matches reference behavior).
 */
import { debugEnabled, debugWrite } from '../debug.ts';

export interface CabacDebugOptions {
  trace: boolean;
  operationTrace: boolean;
  bypassTrace: boolean;
  stateTrace: boolean;
}

/** Capture debug flags once before decoding starts; CABAC hot loops never read process.env. */
export function captureCabacDebugOptions(): CabacDebugOptions {
  return {
    trace: debugEnabled('CABAC_TRACE'),
    operationTrace: debugEnabled('CABAC_OP_TRACE'),
    bypassTrace: debugEnabled('BYP_TRACE'),
    stateTrace: debugEnabled('HEVC_TRACE'),
  };
}

// rangeTabLps (spec Table 9-44): LPS range by state (rows) and (codIRange>>6)-4
const LPS_TABLE_ROWS: readonly (readonly number[])[] = [
  [128, 176, 208, 240],
  [128, 167, 197, 227],
  [128, 158, 187, 216],
  [123, 150, 178, 205],
  [116, 142, 169, 195],
  [111, 135, 160, 185],
  [105, 128, 152, 175],
  [100, 122, 144, 166],
  [95, 116, 137, 158],
  [90, 110, 130, 150],
  [85, 104, 123, 142],
  [81, 99, 117, 135],
  [77, 94, 111, 128],
  [73, 89, 105, 122],
  [69, 85, 100, 116],
  [66, 80, 95, 110],
  [62, 76, 90, 104],
  [59, 72, 86, 99],
  [56, 69, 81, 94],
  [53, 65, 77, 89],
  [51, 62, 73, 85],
  [48, 59, 69, 80],
  [46, 56, 66, 76],
  [43, 53, 63, 72],
  [41, 50, 59, 69],
  [39, 48, 56, 65],
  [37, 45, 54, 62],
  [35, 43, 51, 59],
  [33, 41, 48, 56],
  [32, 39, 46, 53],
  [30, 37, 43, 50],
  [29, 35, 41, 48],
  [27, 33, 39, 45],
  [26, 31, 37, 43],
  [24, 30, 35, 41],
  [23, 28, 33, 39],
  [22, 27, 32, 37],
  [21, 26, 30, 35],
  [20, 24, 29, 33],
  [19, 23, 27, 31],
  [18, 22, 26, 30],
  [17, 21, 25, 28],
  [16, 20, 23, 27],
  [15, 19, 22, 25],
  [14, 18, 21, 24],
  [14, 17, 20, 23],
  [13, 16, 19, 22],
  [12, 15, 18, 21],
  [12, 14, 17, 20],
  [11, 14, 16, 19],
  [11, 13, 15, 18],
  [10, 12, 15, 17],
  [10, 12, 14, 16],
  [9, 11, 13, 15],
  [9, 11, 12, 14],
  [8, 10, 12, 14],
  [8, 9, 11, 13],
  [7, 9, 11, 12],
  [7, 9, 10, 12],
  [7, 8, 10, 11],
  [6, 8, 9, 11],
  [6, 7, 9, 10],
  [6, 7, 8, 9],
  [2, 2, 2, 2],
];
const LPS_TABLE = LPS_TABLE_ROWS.flat();


const NEXT_MPS = [
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
  17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
  33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
  49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
];
const NEXT_LPS = [
  0, 0, 1, 2, 2, 4, 4, 5, 6, 7, 8, 9, 9, 11, 11, 12,
  13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24,
  24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33,
  33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
];

// context index layout (offsets into the context table)
export const CTX = {
  SAO_MERGE_FLAG: 0,
  SAO_TYPE_IDX: 1,
  SPLIT_CU_FLAG: 2,
  CU_SKIP_FLAG: 5,
  PART_MODE: 8,
  PREV_INTRA_LUMA_PRED_FLAG: 12,
  INTRA_CHROMA_PRED_MODE: 13,
  CBF_LUMA: 14,
  CBF_CHROMA: 16,
  SPLIT_TRANSFORM_FLAG: 20,
  CU_CHROMA_QP_OFFSET_FLAG: 23,
  CU_CHROMA_QP_OFFSET_IDX: 24,
  LAST_SIG_X_PREFIX: 25,
  LAST_SIG_Y_PREFIX: 43,
  CODED_SUB_BLOCK_FLAG: 61,
  SIGNIFICANT_COEFF_FLAG: 65,
  SIGNIFICANT_COEFF_FLAG_SKIP: 107,
  COEFF_ABS_LEVEL_GREATER1: 109,
  COEFF_ABS_LEVEL_GREATER2: 133,
  CU_QP_DELTA_ABS: 139,
  TRANSFORM_SKIP_FLAG: 141,
  RDPCM_FLAG: 143,
  RDPCM_DIR: 145,
  MERGE_FLAG: 147,
  MERGE_IDX: 148,
  PRED_MODE_FLAG: 149,
  ABS_MVD_GREATER01: 150,
  MVP_LX_FLAG: 152,
  RQT_ROOT_CBF: 153,
  REF_IDX_LX: 154,
  INTER_PRED_IDC: 156,
  CU_TRANSQUANT_BYPASS_FLAG: 161,
  LOG2_RES_SCALE_ABS_PLUS1: 162,
  RES_SCALE_SIGN_FLAG: 170,
  TABLE_LENGTH: 172,
};

// ---- context initialization values (spec tables 9-x, initValue form) ----
const INIT_SPLIT_CU_FLAG = [139, 141, 157, 107, 139, 126, 107, 139, 126];
const INIT_PART_MODE = [184, 154, 139, 154, 154, 154, 139, 154, 154];
const INIT_PREV_INTRA = [184, 154, 183];
const INIT_INTRA_CHROMA = [63, 152, 152];
const INIT_CBF_LUMA = [111, 141, 153, 111];
const INIT_CBF_CHROMA = [94, 138, 182, 154, 149, 107, 167, 154, 149, 92, 167, 154];
const INIT_SPLIT_TRANSFORM = [153, 138, 138, 124, 138, 94, 224, 167, 122];
const INIT_LAST_SIG_PREFIX = [
  110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63,
  125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94, 108, 123, 108,
  125, 110, 124, 110, 95, 94, 125, 111, 111, 79, 125, 126, 111, 111, 79, 108, 123, 93,
];
const INIT_CODED_SUB_BLOCK = [91, 171, 134, 141, 121, 140, 61, 154, 121, 140, 61, 154];
const INIT_SIGNIFICANT_COEFF: readonly number[][] = [
  [111, 111, 125, 110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107,
    125, 141, 179, 153, 125, 107, 125, 141, 179, 153, 125, 140, 139, 182, 182, 152,
    136, 152, 136, 153, 136, 139, 111, 136, 139, 111],
  [155, 154, 139, 153, 139, 123, 123, 63, 153, 166, 183, 140, 136, 153, 154, 166,
    183, 140, 136, 153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 123, 123, 107,
    121, 107, 121, 167, 151, 183, 140, 151, 183, 140],
  [170, 154, 139, 153, 139, 123, 123, 63, 124, 166, 183, 140, 136, 153, 154, 166,
    183, 140, 136, 153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 138, 138, 122,
    121, 122, 121, 167, 151, 183, 140, 151, 183, 140],
];
const INIT_SIG_SKIPMODE = [[141, 111], [140, 140], [140, 140]];
const INIT_GREATER1 = [
  140, 92, 137, 138, 140, 152, 138, 139, 153, 74, 149, 92, 139, 107, 122, 152,
  140, 179, 166, 182, 140, 227, 122, 197, 154, 196, 196, 167, 154, 152, 167, 182,
  182, 134, 149, 136, 153, 121, 136, 137, 169, 194, 166, 167, 154, 167, 137, 182,
  154, 196, 167, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121, 136, 122,
  169, 208, 166, 167, 154, 152, 167, 182,
];
const INIT_GREATER2 = [
  138, 153, 136, 167, 152, 152, 107, 167, 91, 122, 107, 167,
  107, 167, 91, 107, 107, 167,
];
const INIT_SAO_MERGE = [153, 153, 153];
const INIT_SAO_TYPE = [200, 185, 160];
const INIT_CU_QP_DELTA_ABS = [154, 154];
const INIT_TRANSFORM_SKIP = [139, 139];
const INIT_TRANSQUANT_BYPASS = [154, 154, 154];
const INIT_154 = new Uint8Array(12).fill(154);

function initContexts(ctx: { state: Uint8Array; mps: Uint8Array }, initType: number, qp: number) {
  // `initValues` is an array slice start; count consecutive values are consumed
  const set = (idx: number, initValues: ArrayLike<number>, offset: number, count = 1) => {
    for (let i = 0; i < count; i++) {
      const initValue = initValues[offset + i]!;
      const slopeIdx = initValue >> 4, intersecIdx = initValue & 0xF;
      const m = slopeIdx * 5 - 45, n = (intersecIdx << 3) - 16;
      const q = qp < 0 ? 0 : qp > 51 ? 51 : qp;
      let pre = ((m * q) >> 4) + n;
      if (pre < 1) pre = 1; else if (pre > 126) pre = 126;
      const mps = pre <= 63 ? 0 : 1;
      ctx.mps[idx + i] = mps;
      ctx.state[idx + i] = mps ? pre - 64 : 63 - pre;
    }
  };
  set(CTX.SPLIT_CU_FLAG, INIT_SPLIT_CU_FLAG, initType * 3, 3);
  set(CTX.PART_MODE, INIT_PART_MODE, initType !== 2 ? initType * 4 : 5, 4);
  set(CTX.PREV_INTRA_LUMA_PRED_FLAG, INIT_PREV_INTRA, initType);
  set(CTX.INTRA_CHROMA_PRED_MODE, INIT_INTRA_CHROMA, initType);
  set(CTX.CBF_LUMA, INIT_CBF_LUMA, initType === 0 ? 0 : 2, 2);
  set(CTX.CBF_CHROMA, INIT_CBF_CHROMA, initType * 4, 4);
  set(CTX.SPLIT_TRANSFORM_FLAG, INIT_SPLIT_TRANSFORM, initType * 3, 3);
  set(CTX.LAST_SIG_X_PREFIX, INIT_LAST_SIG_PREFIX, initType * 18, 18);
  set(CTX.LAST_SIG_Y_PREFIX, INIT_LAST_SIG_PREFIX, initType * 18, 18);
  set(CTX.CODED_SUB_BLOCK_FLAG, INIT_CODED_SUB_BLOCK, initType * 4, 4);
  const sig = INIT_SIGNIFICANT_COEFF[initType]!;
  set(CTX.SIGNIFICANT_COEFF_FLAG, sig, 0, sig.length);
  set(CTX.SIGNIFICANT_COEFF_FLAG_SKIP, INIT_SIG_SKIPMODE[initType]!, 0, 2);
  set(CTX.COEFF_ABS_LEVEL_GREATER1, INIT_GREATER1, initType * 24, 24);
  set(CTX.COEFF_ABS_LEVEL_GREATER2, INIT_GREATER2, initType * 6, 6);
  set(CTX.SAO_MERGE_FLAG, INIT_SAO_MERGE, initType);
  set(CTX.SAO_TYPE_IDX, INIT_SAO_TYPE, initType);
  set(CTX.CU_QP_DELTA_ABS, INIT_CU_QP_DELTA_ABS, 0, 2);
  set(CTX.TRANSFORM_SKIP_FLAG, INIT_TRANSFORM_SKIP, 0, 2);
  set(CTX.CU_TRANSQUANT_BYPASS_FLAG, INIT_TRANSQUANT_BYPASS, initType);
  set(CTX.CU_CHROMA_QP_OFFSET_FLAG, INIT_154, 0);
  set(CTX.CU_CHROMA_QP_OFFSET_IDX, INIT_154, 0);
  set(CTX.LOG2_RES_SCALE_ABS_PLUS1, INIT_154, 0, 8);
  set(CTX.RES_SCALE_SIGN_FLAG, INIT_154, 0, 2);
}

export class Cabac {
  private states = new Uint8Array(CTX.TABLE_LENGTH);
  private mpsBits = new Uint8Array(CTX.TABLE_LENGTH);
  codIRange = 510;
  codIOffset = 0;
  private bitPos = 0; // absolute bit position in rbsp
  private bitEnd: number;
  private rbsp: Uint8Array;
  private debug: CabacDebugOptions;

  constructor(
    rbsp: Uint8Array, sliceDataStartByte: number,
    debug: CabacDebugOptions = captureCabacDebugOptions(),
  ) {
    this.rbsp = rbsp;
    this.debug = debug;
    this.bitPos = sliceDataStartByte * 8;
    this.bitEnd = rbsp.length * 8;
    this.codIOffset = this.readBits(9);
    if (this.codIOffset >= 510) this.codIOffset = 509; // corrupt stream guard
  }

  initContexts(initType: number, qp: number) {
    initContexts({ state: this.states, mps: this.mpsBits }, initType, qp);
  }

  saveContexts(): { states: Uint8Array; mps: Uint8Array } {
    return { states: this.states.slice(), mps: this.mpsBits.slice() };
  }

  loadContexts(ctx: { states: Uint8Array; mps: Uint8Array }) {
    this.states.set(ctx.states);
    this.mpsBits.set(ctx.mps);
  }

  /** Byte position immediately following CABAC's look-ahead buffer. */
  rawBytePosition(): number {
    return (this.bitPos + 7) >> 3;
  }

  private readBit(): number {
    const p = this.bitPos;
    if (p >= this.bitEnd) { this.bitPos = p + 1; return 0; }
    const bit = (this.rbsp[p >> 3] >> (7 - (p & 7))) & 1;
    this.bitPos = p + 1;
    return bit;
  }
  private readBits(n: number): number {
    let v = 0;
    for (let i = 0; i < n; i++) v = (v << 1) | this.readBit();
    return v;
  }

  dump(tag: string) {
    if (this.debug.stateTrace) {
      debugWrite(`[tr] cabac_state ${tag} range=${this.codIRange} offset=${this.codIOffset}\n`);
    }
  }

  decodeBin(ctxIdx: number): number {
    const state = this.states[ctxIdx]!;
    const qIdx = (this.codIRange >> 6) - 4;
    const rLps = LPS_TABLE[(state << 2) | qIdx]!;
    this.codIRange -= rLps;
    let bit: number;
    const trace = this.debug.trace;
    const rIn = trace ? this.codIRange + rLps : 0;
    const oIn = trace ? this.codIOffset : 0;
    if (this.codIOffset >= this.codIRange) {
      // LPS
      this.codIOffset -= this.codIRange;
      this.codIRange = rLps;
      bit = 1 - this.mpsBits[ctxIdx]!;
      if (state === 0) this.mpsBits[ctxIdx] = bit;
      this.states[ctxIdx] = NEXT_LPS[state]!;
    } else {
      bit = this.mpsBits[ctxIdx]!;
      this.states[ctxIdx] = NEXT_MPS[state]!;
    }
    // renormalize
    while (this.codIRange < 256) {
      this.codIOffset = ((this.codIOffset << 1) | this.readBit()) & 511;
      this.codIRange <<= 1;
    }
    if (trace) debugWrite(`[cb] ctx=${ctxIdx} st=${state} q=${qIdx} lps=${rLps} rIn=${rIn} oIn=${oIn} bit=${bit} rOut=${this.codIRange} oOut=${this.codIOffset}\n`);
    if (this.debug.operationTrace) debugWrite(`OP C ${ctxIdx} ${bit}\n`);
    return bit;
  }

  decodeBypass(): number {
    // offset may exceed 9 bits here (offset < range <= 510, shifted), keep full precision
    this.codIOffset = (this.codIOffset << 1) | this.readBit();
    if (this.codIOffset >= this.codIRange) {
      this.codIOffset -= this.codIRange;
      if (this.debug.bypassTrace) debugWrite('1');
      if (this.debug.operationTrace) debugWrite('OP B 1\n');
      return 1;
    }
    if (this.debug.bypassTrace) debugWrite('0');
    if (this.debug.operationTrace) debugWrite('OP B 0\n');
    return 0;
  }

  decodeTerminate(): number {
    this.codIRange -= 2;
    if (this.codIOffset >= this.codIRange) {
      this.codIOffset -= this.codIRange; // decoder moves past; stream ends
      this.codIRange = 2;
      return 1;
    }
    while (this.codIRange < 256) {
      this.codIOffset = ((this.codIOffset << 1) | this.readBit()) & 511;
      this.codIRange <<= 1;
    }
    return 0;
  }

  /** read n bypass bits (FL binarization) */
  readFL(n: number): number {
    let v = 0;
    for (let i = 0; i < n; i++) v = (v << 1) | this.decodeBypass();
    return v;
  }

  /** bypass EGk golomb suffix for coeff_abs_level_remaining (bit-exact with reference decoders) */
  readEGk(k: number): number {
    let base = 0;
    let n = k;
    for (; ;) {
      const bit = this.decodeBypass();
      if (bit === 0) break;
      if (n >= 31) throw new Error('CABAC: EGk overflow');
      base += 1 << n;
      n++;
    }
    let suffix = 0;
    for (let i = 0; i < n; i++) suffix = (suffix << 1) | this.decodeBypass();
    return base + suffix;
  }
}
