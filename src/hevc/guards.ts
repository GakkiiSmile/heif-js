/** HEVC syntax validation shared by slice setup and focused regression tests. */

export function checkedSliceQp(bitDepth: number, initQpMinus26: number, sliceQpDelta: number): number {
  if (!Number.isSafeInteger(bitDepth) || bitDepth < 8 || bitDepth > 16) {
    throw new Error('HEVC: unsupported luma bit depth for QP validation');
  }
  const qpBdOffset = (bitDepth - 8) * 6;
  const minimumInit = -(26 + qpBdOffset);
  if (!Number.isSafeInteger(initQpMinus26) || initQpMinus26 < minimumInit || initQpMinus26 > 25) {
    throw new Error('HEVC: init_qp_minus26 is out of range for the luma bit depth');
  }
  const sliceQp = 26 + initQpMinus26 + sliceQpDelta;
  if (!Number.isSafeInteger(sliceQp) || sliceQp < -qpBdOffset || sliceQp > 51) {
    throw new Error('HEVC: slice_qp_delta produces an out-of-range luma QP');
  }
  return sliceQp;
}

export function validateSaoOffsetScale(scale: number, bitDepth: number, component: string): void {
  if (!Number.isSafeInteger(bitDepth) || bitDepth < 8 || bitDepth > 16) {
    throw new Error(`HEVC: unsupported ${component} bit depth for SAO validation`);
  }
  const maximum = Math.max(0, bitDepth - 10);
  if (!Number.isSafeInteger(scale) || scale < 0 || scale > maximum) {
    throw new Error(`HEVC: ${component} SAO offset scale ${scale} exceeds the bit-depth limit ${maximum}`);
  }
}

/** Convert cumulative entry-point offsets into the exact substreams signalled by one slice. */
export function sliceSubstreamStarts(
  dataStart: number, entryPointOffsets: readonly number[], payloadLength: number,
): number[] {
  if (!Number.isSafeInteger(dataStart) || !Number.isSafeInteger(payloadLength) ||
      payloadLength <= 0 || dataStart < 0 || dataStart >= payloadLength) {
    throw new Error('HEVC: slice data start is outside the NAL payload');
  }
  const starts = [dataStart];
  let previous = dataStart;
  for (const offset of entryPointOffsets) {
    const start = dataStart + offset;
    if (!Number.isSafeInteger(offset) || offset <= 0 || !Number.isSafeInteger(start) ||
        start <= previous || start >= payloadLength) {
      throw new Error('HEVC: slice entry-point offset is outside the NAL payload');
    }
    starts.push(start);
    previous = start;
  }
  return starts;
}
