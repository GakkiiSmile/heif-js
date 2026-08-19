import { Av1BitReader } from './bitreader.ts';
import { DEFAULT_DECODE_LIMITS } from '../limits.ts';

export const OBU_SEQUENCE_HEADER = 1;
export const OBU_FRAME_HEADER = 3;
export const OBU_TILE_GROUP = 4;
export const OBU_FRAME = 6;

export interface Av1Obu {
  type: number;
  temporalId: number;
  spatialId: number;
  payload: Uint8Array;
}

export interface Av1TileData {
  index: number;
  row: number;
  column: number;
  data: Uint8Array;
}

export interface Av1SegmentFeature {
  deltaQ: number;
  deltaLfYVertical: number;
  deltaLfYHorizontal: number;
  deltaLfU: number;
  deltaLfV: number;
  reference: number;
  skip: boolean;
  globalMv: boolean;
}

export interface Av1FilmGrain {
  seed: number;
  yPoints: [number, number][];
  chromaScalingFromLuma: boolean;
  uvPoints: [[number, number][], [number, number][]];
  scalingShift: number;
  arCoeffLag: number;
  arCoeffsY: number[];
  arCoeffsUv: [number[], number[]];
  arCoeffShift: number;
  grainScaleShift: number;
  uvMult: [number, number];
  uvLumaMult: [number, number];
  uvOffset: [number, number];
  overlap: boolean;
  clipToRestrictedRange: boolean;
}

export interface Av1SequenceHeader {
  profile: number;
  stillPicture: boolean;
  reducedStillPictureHeader: boolean;
  timingInfoPresent: boolean;
  decoderModelInfoPresent: boolean;
  equalPictureInterval: boolean;
  displayModelInfoPresent: boolean;
  numOperatingPoints: number;
  seqLevelIdx: number[];
  seqTier: number[];
  decoderModelParamPresent: boolean[];
  operatingPointIdc: number[];
  bufferRemovalDelayBits: number;
  framePresentationDelayBits: number;
  widthBits: number;
  heightBits: number;
  maxWidth: number;
  maxHeight: number;
  frameIdNumbersPresent: boolean;
  frameIdBits: number;
  deltaFrameIdBits: number;
  sb128: boolean;
  filterIntra: boolean;
  intraEdgeFilter: boolean;
  interIntra: boolean;
  maskedCompound: boolean;
  warpedMotion: boolean;
  dualFilter: boolean;
  orderHint: boolean;
  orderHintBits: number;
  jointCompound: boolean;
  refFrameMvs: boolean;
  screenContentTools: number;
  forceIntegerMv: number;
  superRes: boolean;
  cdef: boolean;
  restoration: boolean;
  bitDepth: number;
  monochrome: boolean;
  colorPrimaries: number;
  transferCharacteristics: number;
  matrixCoefficients: number;
  fullRange: boolean;
  subsamplingX: number;
  subsamplingY: number;
  chromaSamplePosition: number;
  separateUvDeltaQ: boolean;
  filmGrainPresent: boolean;
}

export interface Av1FrameHeader {
  frameType: number;
  showFrame: boolean;
  errorResilientMode: boolean;
  disableCdfUpdate: boolean;
  refreshContext: boolean;
  allowScreenContentTools: boolean;
  forceIntegerMv: boolean;
  frameSizeOverride: boolean;
  width: number;
  upscaledWidth: number;
  superResDenominator: number;
  height: number;
  renderWidth: number;
  renderHeight: number;
  allowIntrabc: boolean;
  tileCols: number;
  tileRows: number;
  tileColsLog2: number;
  tileRowsLog2: number;
  tileColStarts: number[];
  tileRowStarts: number[];
  contextUpdateTileId: number;
  tileSizeBytes: number;
  baseQIdx: number;
  yDcDelta: number;
  uDcDelta: number;
  uAcDelta: number;
  vDcDelta: number;
  vAcDelta: number;
  usingQmatrix: boolean;
  qmY: number;
  qmU: number;
  qmV: number;
  segmentationEnabled: boolean;
  segments: Av1SegmentFeature[];
  lastActiveSegmentId: number;
  segmentationPreskip: boolean;
  segmentQIndices: number[];
  segmentLossless: boolean[];
  allLossless: boolean;
  deltaQPresent: boolean;
  deltaQResLog2: number;
  deltaLfPresent: boolean;
  deltaLfResLog2: number;
  deltaLfMulti: boolean;
  loopFilterLevels: [number, number, number, number];
  loopFilterSharpness: number;
  loopFilterModeRefDeltaEnabled: boolean;
  loopFilterRefDeltas: number[];
  loopFilterModeDeltas: number[];
  cdefDamping: number;
  cdefBits: number;
  cdefYStrength: number[];
  cdefUvStrength: number[];
  restorationTypes: [number, number, number];
  restorationUnitSizeLog2: [number, number];
  txModeSwitchable: boolean;
  reducedTransformSet: boolean;
  filmGrain: Av1FilmGrain | null;
  headerBits: number;
  tileData: Uint8Array;
}

export function parseObus(data: Uint8Array, maxObus = DEFAULT_DECODE_LIMITS.maxBoxes): Av1Obu[] {
  const result: Av1Obu[] = [];
  let position = 0;
  while (position < data.length) {
    if (result.length >= maxObus) throw new Error(`AV1: OBU count exceeds configured limit ${maxObus}`);
    const header = data[position++]!;
    if (header & 0x80 || header & 1) throw new Error('AV1: invalid OBU header');
    const type = (header >> 3) & 0xf;
    const hasExtension = !!(header & 4);
    const hasSize = !!(header & 2);
    let temporalId = 0, spatialId = 0;
    if (hasExtension) {
      if (position >= data.length) throw new Error('AV1: truncated OBU extension');
      const extension = data[position++]!;
      if (extension & 7) throw new Error('AV1: invalid OBU extension');
      temporalId = extension >> 5;
      spatialId = (extension >> 3) & 3;
    }
    let size = data.length - position;
    if (hasSize) {
      size = 0;
      let shift = 0;
      let complete = false;
      for (let i = 0; i < 8; i++) {
        if (position >= data.length) throw new Error('AV1: truncated OBU size');
        const byte = data[position++]!;
        size += (byte & 0x7f) * 2 ** shift;
        if (!Number.isSafeInteger(size)) throw new Error('AV1: OBU size exceeds JavaScript safe range');
        if (!(byte & 0x80)) { complete = true; break; }
        shift += 7;
      }
      if (!complete) throw new Error('AV1: invalid OBU size');
    }
    if (!Number.isSafeInteger(size) || size < 0 || position > data.length - size) {
      throw new Error('AV1: truncated OBU');
    }
    result.push({ type, temporalId, spatialId, payload: data.subarray(position, position + size) });
    position += size;
  }
  return result;
}

export function parseTileGroup(payload: Uint8Array, header: Av1FrameHeader): Av1TileData[] {
  const reader = new Av1BitReader(payload);
  const tileCount = header.tileCols * header.tileRows;
  const haveTilePosition = tileCount > 1 ? !!reader.bit() : false;
  let start = 0, end = tileCount - 1;
  if (haveTilePosition) {
    const bits = header.tileColsLog2 + header.tileRowsLog2;
    start = reader.bits(bits);
    end = reader.bits(bits);
  }
  if (start > end || end >= tileCount) throw new Error('AV1: invalid tile-group range');
  reader.byteAlign();
  let position = reader.bitPosition >> 3;
  const result: Av1TileData[] = [];
  for (let index = start; index <= end; index++) {
    let size = payload.length - position;
    if (index !== end) {
      if (!header.tileSizeBytes || position + header.tileSizeBytes > payload.length) {
        throw new Error('AV1: truncated tile-size field');
      }
      size = 1;
      for (let byte = 0; byte < header.tileSizeBytes; byte++) {
        size += payload[position++]! * 2 ** (byte * 8);
      }
    }
    if (size < 0 || position + size > payload.length) throw new Error('AV1: truncated tile payload');
    result.push({
      index,
      row: Math.floor(index / header.tileCols),
      column: index % header.tileCols,
      data: payload.subarray(position, position + size),
    });
    position += size;
  }
  if (position !== payload.length) throw new Error('AV1: trailing bytes after tile group');
  return result;
}

export function parseSequenceHeader(data: Uint8Array): Av1SequenceHeader {
  const reader = new Av1BitReader(data);
  const profile = reader.bits(3);
  const stillPicture = !!reader.bit();
  const reducedStillPictureHeader = !!reader.bit();
  let timingInfoPresent = false;
  let decoderModelInfoPresent = false;
  let equalPictureInterval = false;
  let displayModelInfoPresent = false;
  let numOperatingPoints = 1;
  const decoderModelParamPresent: boolean[] = [];
  const operatingPointIdc: number[] = [];
  const seqLevelIdx: number[] = [], seqTier: number[] = [];
  let bufferRemovalDelayBits = 0;
  let framePresentationDelayBits = 0;

  if (reducedStillPictureHeader) {
    seqLevelIdx.push(reader.bits(5));
    seqTier.push(0);
  } else {
    timingInfoPresent = !!reader.bit();
    if (timingInfoPresent) {
      reader.bits(32); reader.bits(32);
      equalPictureInterval = !!reader.bit();
      if (equalPictureInterval) reader.uvlc();
      decoderModelInfoPresent = !!reader.bit();
      if (decoderModelInfoPresent) {
        const encoderDecoderBufferDelayBits = reader.bits(5) + 1;
        reader.bits(32);
        bufferRemovalDelayBits = reader.bits(5) + 1;
        framePresentationDelayBits = reader.bits(5) + 1;
        void encoderDecoderBufferDelayBits;
      }
    }
    displayModelInfoPresent = !!reader.bit();
    numOperatingPoints = reader.bits(5) + 1;
    for (let i = 0; i < numOperatingPoints; i++) {
      operatingPointIdc.push(reader.bits(12));
      const level = reader.bits(5);
      seqLevelIdx.push(level);
      seqTier.push(level > 7 ? reader.bit() : 0);
      let decoderPresent = false;
      if (decoderModelInfoPresent) {
        decoderPresent = !!reader.bit();
        if (decoderPresent) {
          reader.bits(bufferRemovalDelayBits);
          reader.bits(bufferRemovalDelayBits);
          reader.bit();
        }
      }
      decoderModelParamPresent.push(decoderPresent);
      if (displayModelInfoPresent && reader.bit()) reader.bits(4);
    }
  }

  const widthBits = reader.bits(4) + 1;
  const heightBits = reader.bits(4) + 1;
  const maxWidth = reader.bits(widthBits) + 1;
  const maxHeight = reader.bits(heightBits) + 1;
  let frameIdNumbersPresent = false, deltaFrameIdBits = 0, frameIdBits = 0;
  if (!reducedStillPictureHeader) {
    frameIdNumbersPresent = !!reader.bit();
    if (frameIdNumbersPresent) {
      deltaFrameIdBits = reader.bits(4) + 2;
      frameIdBits = reader.bits(3) + deltaFrameIdBits + 1;
    }
  }

  const sb128 = !!reader.bit();
  const filterIntra = !!reader.bit();
  const intraEdgeFilter = !!reader.bit();
  let interIntra = false, maskedCompound = false, warpedMotion = false, dualFilter = false;
  let orderHint = false, orderHintBits = 0, jointCompound = false, refFrameMvs = false;
  let screenContentTools = 2, forceIntegerMv = 2;
  if (!reducedStillPictureHeader) {
    interIntra = !!reader.bit();
    maskedCompound = !!reader.bit();
    warpedMotion = !!reader.bit();
    dualFilter = !!reader.bit();
    orderHint = !!reader.bit();
    if (orderHint) {
      jointCompound = !!reader.bit();
      refFrameMvs = !!reader.bit();
    }
    screenContentTools = reader.bit() ? 2 : reader.bit();
    forceIntegerMv = screenContentTools ? (reader.bit() ? 2 : reader.bit()) : 2;
    if (orderHint) orderHintBits = reader.bits(3) + 1;
  }
  const superRes = !!reader.bit();
  const cdef = !!reader.bit();
  const restoration = !!reader.bit();
  let highBitDepth = reader.bit();
  if (profile === 2 && highBitDepth) highBitDepth += reader.bit();
  const bitDepth = highBitDepth === 0 ? 8 : highBitDepth === 1 ? 10 : 12;
  const monochrome = profile === 1 ? false : !!reader.bit();
  const colorDescriptionPresent = !!reader.bit();
  let colorPrimaries = 2, transferCharacteristics = 2, matrixCoefficients = 2;
  if (colorDescriptionPresent) {
    colorPrimaries = reader.bits(8);
    transferCharacteristics = reader.bits(8);
    matrixCoefficients = reader.bits(8);
  }
  let fullRange: boolean;
  let subsamplingX = 0, subsamplingY = 0, chromaSamplePosition = 0;
  if (monochrome) {
    fullRange = !!reader.bit();
    subsamplingX = subsamplingY = 1;
  } else if (colorPrimaries === 1 && transferCharacteristics === 13 && matrixCoefficients === 0) {
    fullRange = true;
  } else {
    fullRange = !!reader.bit();
    if (profile === 0) subsamplingX = subsamplingY = 1;
    else if (profile === 1) subsamplingX = subsamplingY = 0;
    else if (bitDepth === 12) {
      subsamplingX = reader.bit();
      subsamplingY = subsamplingX ? reader.bit() : 0;
    } else {
      subsamplingX = 1;
    }
    if (subsamplingX && subsamplingY) chromaSamplePosition = reader.bits(2);
  }
  const separateUvDeltaQ = monochrome ? false : !!reader.bit();
  const filmGrainPresent = !!reader.bit();

  return {
    profile, stillPicture, reducedStillPictureHeader, timingInfoPresent,
    decoderModelInfoPresent, equalPictureInterval, displayModelInfoPresent,
    numOperatingPoints, seqLevelIdx, seqTier, decoderModelParamPresent, operatingPointIdc,
    bufferRemovalDelayBits, framePresentationDelayBits,
    widthBits, heightBits, maxWidth, maxHeight,
    frameIdNumbersPresent, frameIdBits, deltaFrameIdBits, sb128, filterIntra,
    intraEdgeFilter, interIntra, maskedCompound, warpedMotion, dualFilter, orderHint,
    orderHintBits, jointCompound, refFrameMvs, screenContentTools, forceIntegerMv,
    superRes, cdef, restoration, bitDepth, monochrome, colorPrimaries,
    transferCharacteristics, matrixCoefficients, fullRange, subsamplingX,
    subsamplingY, chromaSamplePosition, separateUvDeltaQ, filmGrainPresent,
  };
}

export function parseFrameHeader(
  data: Uint8Array, sequence: Av1SequenceHeader, temporalId = 0, spatialId = 0,
): Av1FrameHeader {
  const reader = new Av1BitReader(data);
  if (!sequence.reducedStillPictureHeader && reader.bit()) throw new Error('AV1: show-existing-frame is not a still image');
  const frameType = sequence.reducedStillPictureHeader ? 0 : reader.bits(2);
  const showFrame = sequence.reducedStillPictureHeader || !!reader.bit();
  if (showFrame && sequence.decoderModelInfoPresent && !sequence.equalPictureInterval) {
    reader.bits(sequence.framePresentationDelayBits);
  }
  if (!showFrame) reader.bit(); // showable_frame
  const errorResilientMode = frameType === 0 && showFrame || frameType === 3 ||
    sequence.reducedStillPictureHeader || !!reader.bit();
  const disableCdfUpdate = !!reader.bit();
  const allowScreenContentTools = sequence.screenContentTools === 2 ? !!reader.bit() : !!sequence.screenContentTools;
  let forceIntegerMv = allowScreenContentTools ?
    (sequence.forceIntegerMv === 2 ? !!reader.bit() : !!sequence.forceIntegerMv) : false;
  if (frameType === 0 || frameType === 2) forceIntegerMv = true;
  if (sequence.frameIdNumbersPresent) reader.bits(sequence.frameIdBits);
  const frameSizeOverride = sequence.reducedStillPictureHeader ? false : frameType === 3 || !!reader.bit();
  if (sequence.orderHint) reader.bits(sequence.orderHintBits);

  // Key/intra-only still-image path.
  if (frameType !== 0 && frameType !== 2) throw new Error('AV1: only key/intra still frames are supported');
  if (sequence.decoderModelInfoPresent && reader.bit()) {
    for (let index = 0; index < sequence.numOperatingPoints; index++) {
      if (!sequence.decoderModelParamPresent[index]) continue;
      const idc = sequence.operatingPointIdc[index] ?? 0;
      const inTemporalLayer = (idc >> temporalId) & 1;
      const inSpatialLayer = (idc >> (spatialId + 8)) & 1;
      if (!idc || inTemporalLayer && inSpatialLayer) reader.bits(sequence.bufferRemovalDelayBits);
    }
  }
  if (!(frameType === 0 && showFrame)) {
    const refreshFrameFlags = reader.bits(8);
    if (refreshFrameFlags !== 0xff && errorResilientMode && sequence.orderHint) {
      for (let index = 0; index < 8; index++) reader.bits(sequence.orderHintBits);
    }
  }
  let upscaledWidth = sequence.maxWidth, height = sequence.maxHeight;
  if (frameSizeOverride) {
    upscaledWidth = reader.bits(sequence.widthBits) + 1;
    height = reader.bits(sequence.heightBits) + 1;
  }
  let width = upscaledWidth;
  let superResDenominator = 8;
  if (sequence.superRes && reader.bit()) {
    superResDenominator = reader.bits(3) + 9;
    width = Math.max(Math.floor((upscaledWidth * 8 + (superResDenominator >> 1)) / superResDenominator), Math.min(16, upscaledWidth));
  }
  let renderWidth = upscaledWidth, renderHeight = height;
  if (reader.bit()) {
    renderWidth = reader.bits(16) + 1;
    renderHeight = reader.bits(16) + 1;
  }
  const allowIntrabc = allowScreenContentTools && width === upscaledWidth ? !!reader.bit() : false;
  const refreshContext = sequence.reducedStillPictureHeader || disableCdfUpdate ? false : !reader.bit();

  const sbSize = sequence.sb128 ? 128 : 64;
  const sbw = Math.ceil(width / sbSize), sbh = Math.ceil(height / sbSize);
  const tileLog2 = (start: number, target: number): number => {
    let value = 0;
    while ((start << value) < target) value++;
    return value;
  };
  const minCols = tileLog2(4096 / sbSize, sbw);
  const maxCols = tileLog2(1, Math.min(sbw, 64));
  const maxRows = tileLog2(1, Math.min(sbh, 64));
  const minTiles = Math.max(tileLog2(4096 * 2304 / (sbSize * sbSize), sbw * sbh), minCols);
  const uniformTiles = !!reader.bit();
  let tileCols = 1, tileRows = 1, log2Cols = minCols, log2Rows = 0;
  const tileColStarts: number[] = [];
  const tileRowStarts: number[] = [];
  if (uniformTiles) {
    while (log2Cols < maxCols && reader.bit()) log2Cols++;
    const tileWidth = 1 + ((sbw - 1) >> log2Cols);
    tileCols = 0;
    for (let sbx = 0; sbx < sbw; sbx += tileWidth) tileColStarts[tileCols++] = sbx;
    log2Rows = Math.max(minTiles - log2Cols, 0);
    while (log2Rows < maxRows && reader.bit()) log2Rows++;
    const tileHeight = 1 + ((sbh - 1) >> log2Rows);
    tileRows = 0;
    for (let sby = 0; sby < sbh; sby += tileHeight) tileRowStarts[tileRows++] = sby;
  } else {
    const maxTileWidthSb = 4096 / sbSize;
    let maxTileAreaSb = 4096 * 2304 / (sbSize * sbSize);
    let widestTile = 0;
    tileCols = 0;
    for (let sbx = 0; sbx < sbw && tileCols < 64; tileCols++) {
      const maxWidth = Math.min(sbw - sbx, maxTileWidthSb);
      const tileWidth = maxWidth > 1 ? 1 + reader.uniform(maxWidth) : 1;
      tileColStarts[tileCols] = sbx;
      sbx += tileWidth;
      widestTile = Math.max(widestTile, tileWidth);
    }
    log2Cols = tileLog2(1, tileCols);
    if (minTiles) maxTileAreaSb /= 2 ** (minTiles + 1);
    const maxTileHeightSb = Math.max(Math.floor(maxTileAreaSb / widestTile), 1);
    tileRows = 0;
    for (let sby = 0; sby < sbh && tileRows < 64; tileRows++) {
      const maxHeight = Math.min(sbh - sby, maxTileHeightSb);
      const tileHeight = maxHeight > 1 ? 1 + reader.uniform(maxHeight) : 1;
      tileRowStarts[tileRows] = sby;
      sby += tileHeight;
    }
    log2Rows = tileLog2(1, tileRows);
  }
  tileColStarts.push(sbw);
  tileRowStarts.push(sbh);
  let tileSizeBytes = 0;
  let contextUpdateTileId = 0;
  if (log2Cols || log2Rows) {
    contextUpdateTileId = reader.bits(log2Cols + log2Rows);
    if (contextUpdateTileId >= tileCols * tileRows) throw new Error('AV1: invalid context-update tile');
    tileSizeBytes = reader.bits(2) + 1;
  }

  const baseQIdx = reader.bits(8);
  const readDeltaQ = (): number => reader.bit() ? reader.signedBits(7) : 0;
  const yDcDelta = readDeltaQ();
  let uDcDelta = 0, uAcDelta = 0, vDcDelta = 0, vAcDelta = 0;
  if (!sequence.monochrome) {
    const diffUv = sequence.separateUvDeltaQ && !!reader.bit();
    uDcDelta = readDeltaQ();
    uAcDelta = readDeltaQ();
    if (diffUv) { vDcDelta = readDeltaQ(); vAcDelta = readDeltaQ(); }
    else { vDcDelta = uDcDelta; vAcDelta = uAcDelta; }
  }
  const usingQmatrix = !!reader.bit();
  let qmY = 0, qmU = 0, qmV = 0;
  if (usingQmatrix) {
    qmY = reader.bits(4);
    qmU = reader.bits(4);
    qmV = sequence.separateUvDeltaQ ? reader.bits(4) : qmU;
  }

  const segmentationEnabled = !!reader.bit();
  const segments: Av1SegmentFeature[] = Array.from({ length: 8 }, () => ({
    deltaQ: 0, deltaLfYVertical: 0, deltaLfYHorizontal: 0, deltaLfU: 0, deltaLfV: 0,
    reference: -1, skip: false, globalMv: false,
  }));
  let lastActiveSegmentId = -1, segmentationPreskip = false;
  if (segmentationEnabled) {
    // Key/intra-only frames have PRIMARY_REF_NONE, so update_map and
    // update_data are both implicitly true and temporal prediction is off.
    for (let index = 0; index < 8; index++) {
      const segment = segments[index]!;
      if (reader.bit()) { segment.deltaQ = reader.signedBits(9); lastActiveSegmentId = index; }
      if (reader.bit()) { segment.deltaLfYVertical = reader.signedBits(7); lastActiveSegmentId = index; }
      if (reader.bit()) { segment.deltaLfYHorizontal = reader.signedBits(7); lastActiveSegmentId = index; }
      if (reader.bit()) { segment.deltaLfU = reader.signedBits(7); lastActiveSegmentId = index; }
      if (reader.bit()) { segment.deltaLfV = reader.signedBits(7); lastActiveSegmentId = index; }
      if (reader.bit()) {
        segment.reference = reader.bits(3);
        lastActiveSegmentId = index;
        segmentationPreskip = true;
      }
      segment.skip = !!reader.bit();
      if (segment.skip) { lastActiveSegmentId = index; segmentationPreskip = true; }
      segment.globalMv = !!reader.bit();
      if (segment.globalMv) { lastActiveSegmentId = index; segmentationPreskip = true; }
    }
  }
  let deltaQPresent = false, deltaQResLog2 = 0, deltaLfPresent = false, deltaLfResLog2 = 0, deltaLfMulti = false;
  if (baseQIdx) {
    deltaQPresent = !!reader.bit();
    if (deltaQPresent) {
      deltaQResLog2 = reader.bits(2);
      if (!allowIntrabc) {
        deltaLfPresent = !!reader.bit();
        if (deltaLfPresent) {
          deltaLfResLog2 = reader.bits(2);
          deltaLfMulti = !!reader.bit();
        }
      }
    }
  }

  const deltaLossless = yDcDelta === 0 && uDcDelta === 0 && uAcDelta === 0 && vDcDelta === 0 && vAcDelta === 0;
  const segmentQIndices = segments.map(segment =>
    clamp(baseQIdx + (segmentationEnabled ? segment.deltaQ : 0), 0, 255));
  const segmentLossless = segmentQIndices.map(qIndex => qIndex === 0 && deltaLossless);
  const allLossless = segmentLossless.every(Boolean);
  const loopFilterLevels: [number, number, number, number] = [0, 0, 0, 0];
  let loopFilterSharpness = 0;
  let loopFilterModeRefDeltaEnabled = true;
  const loopFilterRefDeltas = [1, 0, 0, 0, -1, 0, -1, -1];
  const loopFilterModeDeltas = [0, 0];
  if (!allLossless && !allowIntrabc) {
    loopFilterLevels[0] = reader.bits(6); loopFilterLevels[1] = reader.bits(6);
    if (!sequence.monochrome && (loopFilterLevels[0] || loopFilterLevels[1])) {
      loopFilterLevels[2] = reader.bits(6); loopFilterLevels[3] = reader.bits(6);
    }
    loopFilterSharpness = reader.bits(3);
    loopFilterModeRefDeltaEnabled = !!reader.bit();
    if (loopFilterModeRefDeltaEnabled && reader.bit()) {
      for (let i = 0; i < 8; i++) if (reader.bit()) loopFilterRefDeltas[i] = reader.signedBits(7);
      for (let i = 0; i < 2; i++) if (reader.bit()) loopFilterModeDeltas[i] = reader.signedBits(7);
    }
  }

  let cdefDamping = 0, cdefBits = 0;
  const cdefYStrength: number[] = [], cdefUvStrength: number[] = [];
  if (!allLossless && sequence.cdef && !allowIntrabc) {
    cdefDamping = reader.bits(2) + 3;
    cdefBits = reader.bits(2);
    for (let i = 0; i < 1 << cdefBits; i++) {
      cdefYStrength.push(reader.bits(6));
      if (!sequence.monochrome) cdefUvStrength.push(reader.bits(6));
    }
  }
  const restorationTypes: [number, number, number] = [0, 0, 0];
  const restorationUnitSizeLog2: [number, number] = [8, 8];
  if ((!allLossless || width !== upscaledWidth) && sequence.restoration && !allowIntrabc) {
    restorationTypes[0] = reader.bits(2);
    if (!sequence.monochrome) { restorationTypes[1] = reader.bits(2); restorationTypes[2] = reader.bits(2); }
    if (restorationTypes.some(Boolean)) {
      restorationUnitSizeLog2[0] = 6 + +sequence.sb128;
      if (reader.bit()) {
        restorationUnitSizeLog2[0]++;
        if (!sequence.sb128) restorationUnitSizeLog2[0] += reader.bit();
      }
      restorationUnitSizeLog2[1] = restorationUnitSizeLog2[0];
      if ((restorationTypes[1] || restorationTypes[2]) &&
          sequence.subsamplingX === 1 && sequence.subsamplingY === 1) {
        restorationUnitSizeLog2[1] -= reader.bit();
      }
    }
  }
  const txModeSwitchable = allLossless ? false : !!reader.bit();
  const reducedTransformSet = !!reader.bit();
  let filmGrain: Av1FilmGrain | null = null;
  if (sequence.filmGrainPresent && (showFrame || frameType !== 0) && reader.bit()) {
    const seed = reader.bits(16);
    // Key/intra frames always carry a complete grain update.
    const yPoints: [number, number][] = [];
    const yPointCount = reader.bits(4);
    if (yPointCount > 14) throw new Error('AV1: invalid film-grain luma point count');
    for (let index = 0; index < yPointCount; index++) yPoints.push([reader.bits(8), reader.bits(8)]);
    const chromaScalingFromLuma = sequence.monochrome ? false : !!reader.bit();
    const uvPoints: [[number, number][], [number, number][]] = [[], []];
    if (!sequence.monochrome && !chromaScalingFromLuma &&
        !(sequence.subsamplingX === 1 && sequence.subsamplingY === 1 && !yPoints.length)) {
      for (let plane = 0; plane < 2; plane++) {
        const count = reader.bits(4);
        if (count > 10) throw new Error('AV1: invalid film-grain chroma point count');
        for (let index = 0; index < count; index++) uvPoints[plane].push([reader.bits(8), reader.bits(8)]);
      }
    }
    if (sequence.subsamplingX === 1 && sequence.subsamplingY === 1 &&
        !!uvPoints[0].length !== !!uvPoints[1].length) {
      throw new Error('AV1: inconsistent 4:2:0 film-grain chroma points');
    }
    const scalingShift = reader.bits(2) + 8;
    const arCoeffLag = reader.bits(2);
    const arPositionCount = 2 * arCoeffLag * (arCoeffLag + 1);
    const arCoeffsY: number[] = [];
    if (yPoints.length) for (let index = 0; index < arPositionCount; index++) arCoeffsY.push(reader.bits(8) - 128);
    const arCoeffsUv: [number[], number[]] = [[], []];
    for (let plane = 0; plane < 2; plane++) {
      if (uvPoints[plane].length || chromaScalingFromLuma) {
        const count = arPositionCount + +!!yPoints.length;
        for (let index = 0; index < count; index++) arCoeffsUv[plane].push(reader.bits(8) - 128);
        if (!yPoints.length) arCoeffsUv[plane][count] = 0;
      }
    }
    const arCoeffShift = reader.bits(2) + 6;
    const grainScaleShift = reader.bits(2);
    const uvMult: [number, number] = [0, 0];
    const uvLumaMult: [number, number] = [0, 0];
    const uvOffset: [number, number] = [0, 0];
    for (let plane = 0; plane < 2; plane++) {
      if (uvPoints[plane].length) {
        uvMult[plane] = reader.bits(8) - 128;
        uvLumaMult[plane] = reader.bits(8) - 128;
        uvOffset[plane] = reader.bits(9) - 256;
      }
    }
    filmGrain = {
      seed, yPoints, chromaScalingFromLuma, uvPoints, scalingShift, arCoeffLag,
      arCoeffsY, arCoeffsUv, arCoeffShift, grainScaleShift,
      uvMult, uvLumaMult, uvOffset,
      overlap: !!reader.bit(), clipToRestrictedRange: !!reader.bit(),
    };
  }
  const headerBits = reader.bitPosition;
  reader.byteAlign();
  return {
    frameType, showFrame, errorResilientMode, disableCdfUpdate, refreshContext,
    allowScreenContentTools, forceIntegerMv, frameSizeOverride, width, upscaledWidth, superResDenominator,
    height, renderWidth, renderHeight, allowIntrabc, tileCols, tileRows,
    tileColsLog2: log2Cols, tileRowsLog2: log2Rows, tileColStarts, tileRowStarts,
    contextUpdateTileId, tileSizeBytes,
    baseQIdx, yDcDelta, uDcDelta, uAcDelta, vDcDelta, vAcDelta,
    usingQmatrix, qmY, qmU, qmV, segmentationEnabled, segments,
    lastActiveSegmentId, segmentationPreskip, segmentQIndices, segmentLossless, allLossless,
    deltaQPresent,
    deltaQResLog2, deltaLfPresent, deltaLfResLog2, deltaLfMulti, loopFilterLevels,
    loopFilterSharpness, loopFilterModeRefDeltaEnabled, loopFilterRefDeltas, loopFilterModeDeltas,
    cdefDamping, cdefBits, cdefYStrength, cdefUvStrength,
    restorationTypes, restorationUnitSizeLog2, txModeSwitchable, reducedTransformSet, filmGrain, headerBits,
    tileData: data.subarray(reader.bitPosition >> 3),
  };
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
