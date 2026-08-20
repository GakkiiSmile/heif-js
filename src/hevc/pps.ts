/** HEVC parameter sets: SPS / PPS parsing (H.265 §7.3.2). */
import { BitReader } from './bitreader.ts';

export interface Spt {
  spsId: number;
  chromaFormatIdc: number;
  separateColourPlane: boolean;
  width: number;
  height: number;
  conformance: { left: number; right: number; top: number; bottom: number } | null;
  bitDepthLuma: number;
  bitDepthChroma: number;
  log2MaxPocLsb: number;
  numShortTermRefPicSets: number;
  stRpsNumDeltaPocs: number[];
  longTermRefPicsPresent: boolean;
  numLongTermRefPicsSps: number;
  temporalMvpEnabled: boolean;
  log2MinCbSize: number;
  log2CtbSize: number;
  log2MinTbSize: number;
  log2MaxTbSize: number;
  maxTransformHierarchyDepthIntra: number;
  maxTransformHierarchyDepthInter: number;
  ampEnabled: boolean;
  saoEnabled: boolean;
  pcmEnabled: boolean;
  pcmSampleBitDepthLuma: number;
  pcmSampleBitDepthChroma: number;
  log2MinPcmCbSize: number;
  log2MaxPcmCbSize: number;
  pcmLoopFilterDisable: boolean;
  strongIntraSmoothing: boolean;
  scalingListEnabled: boolean;
  // range extensions
  transformSkipRotation: boolean;
  transformSkipContextEnabled: boolean;
  intraSmoothingDisabled: boolean;
  highPrecisionOffsets: boolean;
  persistentRiceInit: boolean;
  implicitRdpcm: boolean;
  explicitRdpcm: boolean;
  extendedPrecision: boolean;
  scalingLists: ScalingLists | null;
  colourPrimaries: number;
  transferCharacteristics: number;
  matrixCoefficients: number;
  fullRange: boolean;
  chromaSampleLocation: number;
}

export interface Pps {
  ppsId: number;
  spsId: number;
  dependentSliceSegmentsEnabled: boolean;
  outputFlagPresent: boolean;
  signDataHiding: boolean;
  cabacInitPresent: boolean;
  numExtraSliceHeaderBits: number;
  sliceChromaQpOffsetsPresent: boolean;
  initQpMinus26: number;
  constrainedIntraPred: boolean;
  transformSkipEnabled: boolean;
  cuQpDeltaEnabled: boolean;
  picCbQpOffset: number;
  picCrQpOffset: number;
  transquantBypassEnabled: boolean;
  tilesEnabled: boolean;
  entropyCodingSync: boolean;
  numTileCols: number;
  numTileRows: number;
  colWidths: number[] | null;
  rowHeights: number[] | null;
  loopFilterAcrossTiles: boolean;
  loopFilterAcrossSlices: boolean;
  deblockingOverrideEnabled: boolean;
  picDisableDeblocking: boolean;
  betaOffsetDiv2: number;
  tcOffsetDiv2: number;
  // range extensions
  crossComponentPredictionEnabled: boolean;
  chromaQpOffsetListEnabled: boolean;
  diffCuChromaQpOffsetDepth: number;
  cbQpOffsetList: number[];
  crQpOffsetList: number[];
  log2MaxTransformSkipBlockSize: number;
  log2SaoOffsetScaleLuma: number;
  log2SaoOffsetScaleChroma: number;
  sliceHeaderExtPresent: boolean;
  diffCuQpDeltaDepth: number;
  scalingLists: ScalingLists | null;
}

function profileTierLevel(r: BitReader, maxSubLayers: number) {
  r.u(2 + 1 + 5 + 32 + 4 + 43 + 1); // general profile stuff
  r.u(8); // general_level_idc
  const subLayerPresent: boolean[] = [], subLayerLevel: boolean[] = [];
  for (let i = 0; i < maxSubLayers - 1; i++) {
    subLayerPresent[i] = !!r.u1();
    subLayerLevel[i] = !!r.u1();
  }
  if (maxSubLayers > 1) {
    for (let i = maxSubLayers - 1; i < 8; i++) r.u(2);
    for (let i = 0; i < maxSubLayers - 1; i++) {
      if (subLayerPresent[i]) r.u(2 + 1 + 5 + 32 + 4 + 43 + 1);
      if (subLayerLevel[i]) r.u(8);
    }
  }
}

export function skipShortTermRefPicSet(r: BitReader, idx: number, numDeltaPocs: number[], deltaIdxPresent = false): number {
  if (idx !== 0 && r.u1()) {
    const deltaIdx = deltaIdxPresent ? r.ue() + 1 : 1;
    if (deltaIdx > idx) throw new Error('HEVC: invalid short-term reference-picture-set index');
    r.u1(); // delta_rps_sign
    r.ue(); // abs_delta_rps_minus1
    const refIdx = idx - deltaIdx;
    const refCount = refIdx >= 0 ? (numDeltaPocs[refIdx] ?? 0) : 0;
    if (refCount > 64) throw new Error('HEVC: too many reference pictures');
    let count = 0;
    for (let j = 0; j <= refCount; j++) {
      const used = r.u1();
      const useDelta = used ? 1 : r.u1();
      if (used || useDelta) count++;
    }
    return count;
  }
  const numNegative = r.ue(), numPositive = r.ue();
  if (numNegative + numPositive > 64) throw new Error('HEVC: too many short-term reference pictures');
  for (let i = 0; i < numNegative; i++) { r.ue(); r.u1(); }
  for (let i = 0; i < numPositive; i++) { r.ue(); r.u1(); }
  return numNegative + numPositive;
}

function subLayerHrd(r: BitReader, cpbCntMinus1: number, subPic: boolean) {
  for (let i = 0; i <= cpbCntMinus1; i++) {
    r.ue(); r.ue();
    if (subPic) { r.ue(); r.ue(); }
    r.u1();
  }
}

function hrdParameters(r: BitReader, commonInfPresent: boolean, maxSubLayersMinus1: number) {
  let nalHrd = false, vclHrd = false, subPic = false;
  if (commonInfPresent) {
    nalHrd = !!r.u1(); vclHrd = !!r.u1();
    if (nalHrd || vclHrd) {
      subPic = !!r.u1();
      if (subPic) { r.u(8); r.u(5); r.u1(); r.u(5); }
      r.u(4); r.u(4);
      if (subPic) r.u(4);
      r.u(5); r.u(5); r.u(5);
    }
  }
  for (let i = 0; i <= maxSubLayersMinus1; i++) {
    const fixedGeneral = !!r.u1();
    const fixedWithin = fixedGeneral || !!r.u1();
    let lowDelay = false;
    if (fixedWithin) r.ue();
    else lowDelay = !!r.u1();
    const cpbCntMinus1 = lowDelay ? 0 : r.ue();
    if (cpbCntMinus1 > 31) throw new Error('HEVC: HRD CPB count is out of range');
    if (nalHrd) subLayerHrd(r, cpbCntMinus1, subPic);
    if (vclHrd) subLayerHrd(r, cpbCntMinus1, subPic);
  }
}

interface VuiColour {
  colourPrimaries: number;
  transferCharacteristics: number;
  matrixCoefficients: number;
  fullRange: boolean;
  chromaSampleLocation: number;
}

function vuiParameters(r: BitReader, maxSubLayersMinus1: number): VuiColour {
  const colour: VuiColour = {
    colourPrimaries: 2, transferCharacteristics: 2, matrixCoefficients: 2,
    fullRange: false, chromaSampleLocation: 0,
  };
  const aspectRatio = r.u1();
  if (aspectRatio) { const idc = r.u(8); if (idc === 0xff) { r.u(16); r.u(16); } }
  if (r.u1()) r.u1(); // overscan_appropriate_flag
  if (r.u1()) { // videoSignalType
    r.u(3);
    colour.fullRange = !!r.u1();
    if (r.u1()) {
      colour.colourPrimaries = r.u(8);
      colour.transferCharacteristics = r.u(8);
      colour.matrixCoefficients = r.u(8);
    }
  }
  if (r.u1()) {
    colour.chromaSampleLocation = r.ue();
    r.ue(); // chroma_sample_loc_type_bottom_field
  }
  r.u1(); // neutral_chroma_indication_flag
  r.u1(); // field_seq_flag
  r.u1(); // frame_field_info_present_flag
  if (r.u1()) { r.ue(); r.ue(); r.ue(); r.ue(); }
  if (r.u1()) {
    r.u(32); r.u(32);
    if (r.u1()) r.ue();
    if (r.u1()) hrdParameters(r, true, maxSubLayersMinus1);
  }
  if (r.u1()) { // bitstream restriction
    r.u1(); r.u1(); r.u1();
    r.ue(); r.ue(); r.ue(); r.ue(); r.ue();
  }
  return colour;
}

export function parseSps(rbsp: Uint8Array): Spt {
  const r = new BitReader(rbsp, 2); // skip 2-byte NAL header
  r.u(4); // sps_video_parameter_set_id
  const maxSubLayersMinus1 = r.u(3);
  if (maxSubLayersMinus1 > 6) throw new Error('HEVC: invalid maximum sub-layer count');
  r.u(1); // temporal_id_nesting
  profileTierLevel(r, maxSubLayersMinus1 + 1);
  const spsId = r.ue();
  if (spsId > 15) throw new Error('HEVC: SPS id is out of range');
  let chromaFormatIdc = r.ue();
  if (chromaFormatIdc > 3) throw new Error('HEVC: invalid chroma format');
  let separateColourPlane = false;
  if (chromaFormatIdc === 3) separateColourPlane = !!r.u1();
  const width = r.ue(), height = r.ue();
  if (!width || !height) throw new Error('HEVC: invalid coded dimensions');
  let conformance: Spt['conformance'] = null;
  if (r.u1()) {
    conformance = { left: r.ue(), right: r.ue(), top: r.ue(), bottom: r.ue() };
  }
  const bitDepthLuma = r.ue() + 8, bitDepthChroma = r.ue() + 8;
  if (bitDepthLuma > 16 || bitDepthChroma > 16) throw new Error('HEVC: unsupported component bit depth');
  const log2MaxPocLsb = r.ue() + 4;
  if (log2MaxPocLsb > 16) throw new Error('HEVC: invalid POC bit width');
  const subLayerOrdering = r.u1();
  const start = subLayerOrdering ? 0 : maxSubLayersMinus1;
  for (let i = start; i <= maxSubLayersMinus1; i++) { r.ue(); r.ue(); r.ue(); }
  const log2MinCbSize = r.ue() + 3;
  const log2CtbSize = log2MinCbSize + r.ue();
  const log2MinTbSize = r.ue() + 2;
  const log2MaxTbSize = log2MinTbSize + r.ue();
  if (log2MinCbSize > 6 || log2CtbSize < 4 || log2CtbSize > 6 ||
      log2MinTbSize > 5 || log2MaxTbSize < log2MinTbSize || log2MaxTbSize > 5) {
    throw new Error('HEVC: invalid coding/transform block sizes');
  }
  const maxTransformHierarchyDepthInter = r.ue();
  const maxTransformHierarchyDepthIntra = r.ue();
  let scalingLists: ScalingLists | null = null;
  const scalingListEnabled = !!r.u1();
  if (scalingListEnabled) {
    if (r.u1()) { // sps_scaling_list_data_present
      scalingLists = parseScalingList(r);
    }
  }
  const ampEnabled = !!r.u1();
  const saoEnabled = !!r.u1();
  const pcmEnabled = !!r.u1();
  let pcmSampleBitDepthLuma = 0, pcmSampleBitDepthChroma = 0;
  let log2MinPcmCbSize = 0, log2MaxPcmCbSize = 0;
  let pcmLoopFilterDisable = false;
  if (pcmEnabled) {
    pcmSampleBitDepthLuma = r.u(4) + 1;
    pcmSampleBitDepthChroma = r.u(4) + 1;
    log2MinPcmCbSize = r.ue() + 3;
    log2MaxPcmCbSize = log2MinPcmCbSize + r.ue();
    pcmLoopFilterDisable = !!r.u1();
  }
  const numStrps = r.ue();
  if (numStrps > 64) throw new Error('HEVC: too many short-term reference-picture sets');
  const rpsNumDeltaPocs: number[] = [];
  for (let i = 0; i < numStrps; i++) rpsNumDeltaPocs.push(skipShortTermRefPicSet(r, i, rpsNumDeltaPocs));
  const longTermRefPicsPresent = !!r.u1();
  let numLongTermRefPicsSps = 0;
  if (longTermRefPicsPresent) {
    numLongTermRefPicsSps = r.ue();
    if (numLongTermRefPicsSps > 32) throw new Error('HEVC: too many long-term reference pictures');
    for (let i = 0; i < numLongTermRefPicsSps; i++) { r.u(log2MaxPocLsb); r.u1(); }
  }
  const temporalMvpEnabled = !!r.u1();
  const strongIntraSmoothing = !!r.u1();
  const vui = r.u1() ? vuiParameters(r, maxSubLayersMinus1) : {
    colourPrimaries: 2, transferCharacteristics: 2, matrixCoefficients: 2,
    fullRange: false, chromaSampleLocation: 0,
  };
  const sps: Spt = {
    spsId, chromaFormatIdc, separateColourPlane, width, height, conformance,
    bitDepthLuma, bitDepthChroma, log2MaxPocLsb,
    numShortTermRefPicSets: numStrps, stRpsNumDeltaPocs: rpsNumDeltaPocs,
    longTermRefPicsPresent, numLongTermRefPicsSps, temporalMvpEnabled,
    log2MinCbSize, log2CtbSize,
    log2MinTbSize, log2MaxTbSize, maxTransformHierarchyDepthIntra, maxTransformHierarchyDepthInter,
    ampEnabled, saoEnabled, pcmEnabled, pcmSampleBitDepthLuma, pcmSampleBitDepthChroma,
    log2MinPcmCbSize, log2MaxPcmCbSize, pcmLoopFilterDisable,
    strongIntraSmoothing, scalingListEnabled,
    transformSkipRotation: false, transformSkipContextEnabled: false, intraSmoothingDisabled: false,
    highPrecisionOffsets: false, persistentRiceInit: false, implicitRdpcm: false,
    explicitRdpcm: false, extendedPrecision: false,
    scalingLists,
    ...vui,
  };
  // re-read hierarchy depth (was consumed above, recompute offset)
  // (parseSps keeps fields in order; maxTransformHierarchyDepthIntra read earlier)
  if (r.u1()) { // sps_extension_present_flag
    const rangeExtension = !!r.u1();
    r.u1(); // sps_multilayer_extension_flag
    r.u1(); // sps_3d_extension_flag
    r.u1(); // sps_scc_extension_flag
    r.u(4); // sps_extension_4bits
    if (rangeExtension) {
      sps.transformSkipRotation = !!r.u1();
      sps.transformSkipContextEnabled = !!r.u1();
      sps.implicitRdpcm = !!r.u1();
      sps.explicitRdpcm = !!r.u1();
      sps.extendedPrecision = !!r.u1();
      sps.intraSmoothingDisabled = !!r.u1();
      sps.highPrecisionOffsets = !!r.u1();
      sps.persistentRiceInit = !!r.u1();
      r.u1(); // cabac_bypass_alignment_enabled_flag
    }
  }
  return sps;
}

export interface ScalingLists {
  lists2d: Uint8Array[][];   // [sizeId][matrixId]
  dc2d: Int32Array[];        // dc coef for sizeId 2,3
}

const DEFAULT_SCALING_4 = new Uint8Array(16).fill(16);
const DEFAULT_SCALING_8_INTRA = new Uint8Array([
  16,16,16,16,16,16,16,16, 16,16,17,16,17,16,17,18,
  17,18,18,17,18,21,19,20, 21,20,19,21,24,22,22,24,
  24,22,22,24,25,25,27,30, 27,25,25,29,31,35,35,31,
  29,36,41,44,41,36,47,54, 54,47,65,70,65,88,88,115,
]);
const DEFAULT_SCALING_8_INTER = new Uint8Array([
  16,16,16,16,16,16,16,16, 16,16,17,17,17,17,17,18,
  18,18,18,18,18,20,20,20, 20,20,20,20,24,24,24,24,
  24,24,24,24,25,25,25,25, 25,25,25,28,28,28,28,28,
  28,33,33,33,33,33,41,41, 41,41,54,54,54,71,71,91,
]);

export function parseScalingList(r: BitReader): ScalingLists {
  // lists[sizeId][matrixId]; sizeId 3 uses matrixId 0 (intra) and 3 (inter) only.
  const lists: Uint8Array[][] = [[], [], [], []];
  const dc: Int32Array[] = [new Int32Array(6), new Int32Array(6), new Int32Array(6), new Int32Array(6)];
  for (let sizeId = 0; sizeId < 4; sizeId++) {
    for (let matrixId = 0; matrixId < 6; matrixId += sizeId === 3 ? 3 : 1) {
      const predMode = r.u1();
      const coefNum = Math.min(64, 1 << (4 + (sizeId << 1)));
      let list: Uint8Array;
      if (!predMode) {
        const delta = r.ue();
        const refId = matrixId - (sizeId === 3 ? delta * 3 : delta);
        if (delta === 0) {
          list = (sizeId === 0 ? DEFAULT_SCALING_4 : matrixId < 3 ? DEFAULT_SCALING_8_INTRA : DEFAULT_SCALING_8_INTER).slice();
        } else {
          if (refId < 0 || !lists[sizeId][refId]) {
            throw new Error('HEVC: scaling-list delta references an earlier matrix that does not exist');
          }
          list = lists[sizeId][refId]!.slice();
          dc[sizeId][matrixId] = dc[sizeId]![refId]! || 16;
        }
      } else {
        let nextCoef = 8;
        if (sizeId > 1) {
          const dcVal = r.se();
          dc[sizeId][matrixId] = dcVal + 8;
          nextCoef = dcVal + 8;
        }
        list = new Uint8Array(coefNum);
        for (let i = 0; i < coefNum; i++) {
          const deltaCoef = r.se();
          nextCoef = (nextCoef + deltaCoef + 256) % 256;
          list[i] = nextCoef;
        }
      }
      lists[sizeId][matrixId] = list;
    }
  }
  return { lists2d: lists, dc2d: dc };
}

export function parsePps(rbsp: Uint8Array): Pps {
  const r = new BitReader(rbsp, 2);
  const ppsId = r.ue();
  const spsId = r.ue();
  if (ppsId > 63 || spsId > 15) throw new Error('HEVC: parameter-set id is out of range');
  const dependentSliceSegmentsEnabled = !!r.u1();
  const outputFlagPresent = !!r.u1();
  const numExtraSliceHeaderBits = r.u(3);
  const signDataHiding = !!r.u1();
  const cabacInitPresent = !!r.u1();
  r.ue(); r.ue(); // num_ref_idx defaults
  const initQpMinus26 = r.se();
  const constrainedIntraPred = !!r.u1();
  const transformSkipEnabled = !!r.u1();
  const cuQpDeltaEnabled = !!r.u1();
  const diffCuQpDeltaDepth = cuQpDeltaEnabled ? r.ue() : 0;
  const picCbQpOffset = r.se();
  const picCrQpOffset = r.se();
  const sliceChromaQpOffsetsPresent = !!r.u1();
  r.u1(); r.u1(); // weighted pred/bipred
  const transquantBypassEnabled = !!r.u1();
  const tilesEnabled = !!r.u1();
  const entropyCodingSync = !!r.u1();
  let numTileCols = 1, numTileRows = 1;
  let colWidths: number[] | null = null, rowHeights: number[] | null = null;
  let loopFilterAcrossTiles = true;
  if (tilesEnabled) {
    numTileCols = r.ue() + 1;
    numTileRows = r.ue() + 1;
    if (numTileCols > 1024 || numTileRows > 1024) throw new Error('HEVC: tile count is out of range');
    const uniform = r.u1();
    if (!uniform) {
      colWidths = []; rowHeights = [];
      for (let i = 0; i < numTileCols - 1; i++) colWidths.push(r.ue() + 1);
      for (let i = 0; i < numTileRows - 1; i++) rowHeights.push(r.ue() + 1);
    }
    loopFilterAcrossTiles = !!r.u1();
  }
  const loopFilterAcrossSlices = !!r.u1();
  let deblockingOverrideEnabled = false, picDisableDeblocking = false;
  let betaOffsetDiv2 = 0, tcOffsetDiv2 = 0;
  if (r.u1()) { // deblocking_filter_control_present
    deblockingOverrideEnabled = !!r.u1();
    picDisableDeblocking = !!r.u1();
    if (!picDisableDeblocking) {
      betaOffsetDiv2 = r.se();
      tcOffsetDiv2 = r.se();
    }
  }
  let scalingLists: ScalingLists | null = null;
  if (r.u1()) scalingLists = parseScalingList(r);
  r.u1(); // lists_modification_present
  r.ue(); // log2_parallel_merge_level_minus2
  const sliceHeaderExtPresent = !!r.u1();
  const pps: Pps = {
    ppsId, spsId, dependentSliceSegmentsEnabled, outputFlagPresent, signDataHiding, cabacInitPresent,
    numExtraSliceHeaderBits, sliceChromaQpOffsetsPresent,
    initQpMinus26, constrainedIntraPred, transformSkipEnabled, cuQpDeltaEnabled,
    picCbQpOffset, picCrQpOffset, transquantBypassEnabled, tilesEnabled, entropyCodingSync,
    numTileCols, numTileRows, colWidths, rowHeights, loopFilterAcrossTiles, loopFilterAcrossSlices,
    deblockingOverrideEnabled, picDisableDeblocking, betaOffsetDiv2, tcOffsetDiv2,
    crossComponentPredictionEnabled: false, chromaQpOffsetListEnabled: false,
    diffCuChromaQpOffsetDepth: 0, cbQpOffsetList: [], crQpOffsetList: [],
    log2MaxTransformSkipBlockSize: 2, sliceHeaderExtPresent, diffCuQpDeltaDepth,
    log2SaoOffsetScaleLuma: 0, log2SaoOffsetScaleChroma: 0,
    scalingLists,
  };
  if (r.u1()) { // pps_extension_present_flag
    const rangeExtension = !!r.u1();
    r.u1(); // pps_multilayer_extension_flag
    r.u1(); // pps_3d_extension_flag
    r.u1(); // pps_scc_extension_flag
    r.u(4); // pps_extension_4bits
    if (rangeExtension) {
      const log2MaxTsMinus2 = transformSkipEnabled ? r.ue() : 0;
      const ccp = !!r.u1();
      const cqoList = !!r.u1();
      if (transformSkipEnabled) pps.log2MaxTransformSkipBlockSize = log2MaxTsMinus2 + 2;
      pps.crossComponentPredictionEnabled = ccp;
      pps.chromaQpOffsetListEnabled = cqoList;
      if (cqoList) {
        pps.diffCuChromaQpOffsetDepth = r.ue();
        const n = r.ue() + 1;
        if (n > 6) throw new Error('HEVC: chroma QP offset list is too long');
        for (let i = 0; i < n; i++) {
          pps.cbQpOffsetList.push(r.se());
          pps.crQpOffsetList.push(r.se());
        }
      }
      pps.log2SaoOffsetScaleLuma = r.ue();
      pps.log2SaoOffsetScaleChroma = r.ue();
    }
  }
  return pps;
}
