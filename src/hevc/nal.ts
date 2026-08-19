/** HEVC NAL unit handling: hvcC parsing, length-prefixed extraction, RBSP unescaping. */

export const NAL_VPS = 32, NAL_SPS = 33, NAL_PPS = 34, NAL_AUD = 35,
  NAL_PREFIX_SEI = 39, NAL_SUFFIX_SEI = 40, NAL_SLICE_TRAIL_N = 0, NAL_SLICE_TRAIL_R = 1,
  NAL_SLICE_TSA_N = 2, NAL_SLICE_TSA_R = 3, NAL_SLICE_STSA = 4, NAL_SLICE_RADL = 6,
  NAL_SLICE_RASL = 8, NAL_SLICE_BLA = 16, NAL_SLICE_IDR_N_LP = 19, NAL_SLICE_IDR_W_RADL = 20,
  NAL_SLICE_CRA = 21, NAL_SLICE_IRAP_MIN = 16, NAL_SLICE_IRAP_MAX = 23;

export interface HevcNal {
  type: number;
  /** Unescaped RBSP for decoder-consumed NALs; original EBSP view for ignored metadata NALs. */
  rbsp: Uint8Array;
  /** Original EBSP indices of removed emulation-prevention bytes. */
  skippedBytes?: number[];
}

function snapshotSharedInput(bytes: Uint8Array): Uint8Array {
  // Zero-copy views are only safe for immutable ArrayBuffer-backed input. A
  // SharedArrayBuffer producer could otherwise mutate CABAC/RBSP bytes while a
  // synchronous decode is in progress.
  return bytes.buffer instanceof ArrayBuffer ? bytes : new Uint8Array(bytes);
}

/** Remove 0x000003 emulation prevention bytes. */
export function unescapeRbsp(u8: Uint8Array): Uint8Array {
  return unescapeRbspSparse(snapshotSharedInput(u8)).rbsp;
}

function unescapeRbspSparse(u8: Uint8Array): Pick<HevcNal, 'rbsp' | 'skippedBytes'> {
  const skippedBytes: number[] = [];
  let zeros = 0;
  for (let i = 0; i < u8.length; i++) {
    const b = u8[i]!;
    if (zeros >= 2 && b === 3) {
      if (i + 1 >= u8.length || u8[i + 1]! > 3) throw new Error('HEVC: invalid emulation-prevention byte');
      skippedBytes.push(i);
      zeros = 0;
      continue;
    }
    zeros = b === 0 ? zeros + 1 : 0;
  }
  // Most short parameter sets, and some complete slices, contain no escape
  // bytes. Preserve their original view instead of allocating and copying.
  if (!skippedBytes.length) return { rbsp: u8, skippedBytes };

  const out = new Uint8Array(u8.length - skippedBytes.length);
  let skippedIndex = 0, destination = 0;
  for (let source = 0; source < u8.length; source++) {
    if (skippedBytes[skippedIndex] === source) {
      skippedIndex++;
      continue;
    }
    out[destination++] = u8[source]!;
  }
  return { rbsp: out, skippedBytes };
}

function decoderNalPayload(type: number, nal: Uint8Array): Pick<HevcNal, 'rbsp' | 'skippedBytes'> {
  // VCL NAL units plus VPS/SPS/PPS are the only payloads consumed by this
  // still-image decoder. AUD/SEI and other metadata retain their original view.
  return type <= NAL_PPS ? unescapeRbspSparse(nal) : { rbsp: nal, skippedBytes: [] };
}

/** Map one RBSP byte offset back to EBSP using sparse removed-byte positions. */
export function rbspOffsetToEbsp(rbspOffset: number, skippedBytes: readonly number[]): number {
  // For skipped byte i, all following RBSP offsets shift once when
  // skippedBytes[i] - i <= rbspOffset. This monotonic key permits binary search.
  let lo = 0, hi = skippedBytes.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (skippedBytes[mid]! - mid <= rbspOffset) lo = mid + 1;
    else hi = mid;
  }
  return rbspOffset + lo;
}

/** Count sparse escape positions in an inclusive EBSP byte range. */
export function countSkippedBytesInRange(
  skippedBytes: readonly number[], startInclusive: number, endInclusive: number,
): number {
  if (endInclusive < startInclusive || !skippedBytes.length) return 0;
  let lo = 0, hi = skippedBytes.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (skippedBytes[mid]! < startInclusive) lo = mid + 1;
    else hi = mid;
  }
  const first = lo;
  hi = skippedBytes.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (skippedBytes[mid]! <= endInclusive) lo = mid + 1;
    else hi = mid;
  }
  return lo - first;
}

export interface HvcC {
  lengthSize: number;               // bytes per NAL length field (usually 4)
  paramSets: HevcNal[];
}

export function parseHvcC(payload: Uint8Array): HvcC {
  payload = snapshotSharedInput(payload);
  // layout: [0] version; [1] profile info; [2..5] compat flags; [6..11] constraint flags;
  // [12] level_idc; [13..14] min_spatial_segmentation; [15] parallelismType;
  // [16] chromaFormat; [17] bitDepthLumaMinus8; [18] bitDepthChromaMinus8;
  // [19..20] avgFrameRate;
  // [21] constantFrameRate(2)|numTemporalLayers(3)|temporalIdNested(1)|lengthSizeMinusOne(2)
  // [22] numArrays; then per array: [1] header, [2] count, then NALs with 16-bit lengths
  if (payload.length < 23 || payload[0] !== 1) throw new Error('HEVC: invalid hvcC configuration');
  const lengthSizeMinusOne = payload[21]! & 0x03;
  if (lengthSizeMinusOne === 2) throw new Error('HEVC: reserved hvcC NAL length size');
  const lengthSize = lengthSizeMinusOne + 1;
  const numArrays = payload[22]!;
  const paramSets: HevcNal[] = [];
  let pos = 23;
  for (let a = 0; a < numArrays; a++) {
    if (pos + 3 > payload.length) throw new Error('HEVC: truncated hvcC array');
    const nalType = payload[pos]! & 0x3F;
    pos += 1;
    const numNals = (payload[pos]! << 8) | payload[pos + 1]!;
    pos += 2;
    for (let n = 0; n < numNals; n++) {
      if (pos + 2 > payload.length) throw new Error('HEVC: truncated hvcC NAL length');
      const length = (payload[pos]! << 8) | payload[pos + 1]!;
      pos += 2;
      if (length < 2 || pos + length > payload.length) throw new Error('HEVC: truncated hvcC NAL unit');
      const nal = payload.subarray(pos, pos + length);
      validateNalHeader(nal);
      const actualType = (nal[0]! >> 1) & 0x3f;
      if (actualType !== nalType) throw new Error('HEVC: hvcC array type does not match its NAL unit');
      paramSets.push({ type: nalType, ...decoderNalPayload(nalType, nal) });
      pos += length;
    }
  }
  if (pos !== payload.length) throw new Error('HEVC: trailing bytes in hvcC configuration');
  return { lengthSize, paramSets };
}

/** Extract NAL units from an hvc1 item payload (length-prefixed). */
export function nalsFromLengthPrefixed(u8: Uint8Array, lengthSize: number): HevcNal[] {
  u8 = snapshotSharedInput(u8);
  if (lengthSize !== 1 && lengthSize !== 2 && lengthSize !== 4) {
    throw new Error(`HEVC: invalid NAL length size ${lengthSize}`);
  }
  const out: HevcNal[] = [];
  let pos = 0;
  while (pos + lengthSize <= u8.length) {
    let len = 0;
    for (let i = 0; i < lengthSize; i++) len = len * 256 + u8[pos + i];
    pos += lengthSize;
    if (len < 2 || pos + len > u8.length) throw new Error('HEVC: invalid length-prefixed NAL unit');
    const nal = u8.subarray(pos, pos + len);
    pos += len;
    validateNalHeader(nal);
    const type = (nal[0]! >> 1) & 0x3F;
    out.push({ type, ...decoderNalPayload(type, nal) });
  }
  if (pos !== u8.length) throw new Error('HEVC: truncated NAL length field');
  return out;
}

/** Extract NAL units from an Annex-B stream (00 00 01 start codes). */
export function nalsFromAnnexB(u8: Uint8Array): HevcNal[] {
  u8 = snapshotSharedInput(u8);
  const out: HevcNal[] = [];
  let starts: number[] = [];
  let i = 0;
  while (i + 3 <= u8.length) {
    if (u8[i] === 0 && u8[i + 1] === 0 && (u8[i + 2] === 1)) { starts.push(i + 3); i += 3; }
    else if (u8[i] === 0 && u8[i + 1] === 0 && u8[i + 2] === 0 && i + 4 <= u8.length && u8[i + 3] === 1) { starts.push(i + 4); i += 4; }
    else i++;
  }
  for (let s = 0; s < starts.length; s++) {
    const start = starts[s];
    const end = s + 1 < starts.length ? (u8[starts[s + 1] - 4] === 0 && u8[starts[s + 1] - 3] === 0 && u8[starts[s + 1] - 2] === 0 ? starts[s + 1] - 4 : starts[s + 1] - 3) : u8.length;
    const nal = u8.subarray(start, Math.max(start, end));
    if (!nal.length) continue;
    validateNalHeader(nal);
    const type = (nal[0]! >> 1) & 0x3F;
    out.push({ type, ...decoderNalPayload(type, nal) });
  }
  return out;
}

export function nalHeader(rbsp: Uint8Array): { type: number; layerId: number; tidPlus1: number } {
  validateNalHeader(rbsp);
  const h = rbsp[0]!;
  return { type: (h >> 1) & 0x3F, layerId: ((h & 1) << 5) | (rbsp[1]! >> 3), tidPlus1: rbsp[1]! & 7 };
}

function validateNalHeader(nal: Uint8Array): void {
  if (nal.length < 2 || (nal[0]! & 0x80) || (nal[1]! & 7) === 0) {
    throw new Error('HEVC: invalid NAL header');
  }
}
