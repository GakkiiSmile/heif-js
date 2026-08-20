/** HEVC slice decoder: header parsing, CTU/CU/TU syntax, residual coding (H.265 §7.3, §9.3). */
import { BitReader } from './bitreader.ts';
import { Cabac, CTX, captureCabacDebugOptions } from './cabac.ts';
import type { CabacDebugOptions } from './cabac.ts';
import { parseSps, parsePps, skipShortTermRefPicSet } from './pps.ts';
import type { Spt, Pps, ScalingLists } from './pps.ts';
import { NAL_SPS, NAL_PPS, countSkippedBytesInRange, rbspOffsetToEbsp } from './nal.ts';
import type { HevcNal } from './nal.ts';
import { DecodedFrame, CHROMA_MONO, CHROMA_420, CHROMA_422, CHROMA_444 } from '../frame.ts';
import type { SampleArray } from '../frame.ts';
import { SCAN, CHROMA_QP } from './tables.ts';
import {
  chromaPredMode, createIntraScratch, fillIntraPredModeCandidates, intraPredict, intraPredModeDecode,
} from './intra.ts';
import type { IntraCtx } from './intra.ts';
import { dequant, addInverseTransform, addTransformSkip } from './transform.ts';
import { applySao } from './sao.ts';
import { applyDeblock } from './deblock.ts';
import type { DeblockInfo } from './deblock.ts';
import { debugEnabled, debugValue, debugWrite } from '../debug.ts';
import { assertDimensions, resolveDecodeLimits } from '../limits.ts';
import type { DecodeOptions, ResolvedDecodeLimits } from '../limits.ts';
import { checkedSliceQp, sliceSubstreamStarts, validateSaoOffsetScale } from './guards.ts';

interface HevcDebugOptions {
  cabac: CabacDebugOptions;
  trace: boolean;
  tuDebug: boolean;
  tuX: number;
  tuY: number;
  tuComponent: number;
  disableScaling: boolean;
  disableDeblock: boolean;
  disableSao: boolean;
  disableCrossComponent: boolean;
}

/** Read environment-driven diagnostics once at the start of each frame. */
function captureHevcDebugOptions(): HevcDebugOptions {
  const cabac = captureCabacDebugOptions();
  return {
    cabac,
    trace: cabac.stateTrace,
    tuDebug: debugEnabled('HEVC_TU_DEBUG'),
    tuX: +(debugValue('HEVC_TU_X') ?? 0),
    tuY: +(debugValue('HEVC_TU_Y') ?? 0),
    tuComponent: +(debugValue('HEVC_TU_C') ?? 0),
    disableScaling: debugEnabled('HEVC_DISABLE_SCALING'),
    disableDeblock: debugEnabled('HEVC_DISABLE_DEBLOCK'),
    disableSao: debugEnabled('HEVC_DISABLE_SAO'),
    disableCrossComponent: debugEnabled('HEVC_DISABLE_CROSS_COMPONENT'),
  };
}

const SIG_MAP_4x4 = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 9];
const CHROMA_422_MODE = [
  0, 1, 2, 2, 2, 2, 3, 5, 7, 8, 10, 12, 13, 15, 17, 18, 19, 20,
  21, 22, 23, 23, 24, 24, 25, 25, 26, 27, 27, 28, 28, 29, 29, 30, 31,
];

function pictureCtbCount(sps: Spt): number {
  const size = 1 << sps.log2CtbSize;
  return Math.ceil(sps.width / size) * Math.ceil(sps.height / size);
}

export class HevcDecoder {
  spsMap = new Map<number, Spt>();
  ppsMap = new Map<number, Pps>();
  decodedSps: Spt | null = null;
  readonly limits: ResolvedDecodeLimits;

  constructor(options: DecodeOptions = {}) {
    this.limits = resolveDecodeLimits(options);
  }

  registerParamSets(nals: HevcNal[]) {
    for (const n of nals) {
      if (n.type === NAL_SPS) {
        const sps = parseSps(n.rbsp);
        this.spsMap.set(sps.spsId, sps);
      } else if (n.type === NAL_PPS) {
        const pps = parsePps(n.rbsp);
        this.ppsMap.set(pps.ppsId, pps);
      }
    }
  }

  decodeFrame(sliceNals: HevcNal[]): DecodedFrame {
    const debug = captureHevcDebugOptions();
    const slices = sliceNals.filter(n => n.type <= 21);
    if (!slices.length) throw new Error('HEVC: picture has no slice NAL units');
    const colourPlanes = splitSeparateColourPlaneSlices(this, slices);
    if (colourPlanes) {
      const decoded = colourPlanes.map((planeSlices, plane) => {
        if (!planeSlices.length) throw new Error(`HEVC: separate colour plane ${plane} has no slices`);
        return new SliceDecoder(this, debug).decode(planeSlices);
      });
      const first = decoded[0]!;
      if (decoded.some(frame => frame.width !== first.width || frame.height !== first.height ||
          frame.bitDepth !== first.bitDepth)) {
        throw new Error('HEVC: separate colour planes have inconsistent dimensions or bit depths');
      }
      // Match the per-plane padding so the full-buffer copies below line up.
      const sourceLuma = decoded[0]!.luma;
      const output = new DecodedFrame(
        first.width, first.height, first.bitDepth, CHROMA_444, first.chromaBitDepth,
        sourceLuma.stride - sourceLuma.width,
      );
      for (let plane = 0; plane < 3; plane++) output.planes[plane]!.data.set(decoded[plane]!.luma.data);
      return output;
    }
    return new SliceDecoder(this, debug).decode(slices);
  }
}

function splitSeparateColourPlaneSlices(decoder: HevcDecoder, slices: HevcNal[]): HevcNal[][] | null {
  const groups: HevcNal[][] = [[], [], []];
  let separate = false, combined = false, currentPlane = 0;
  for (const nal of slices) {
    const rbsp = nal.rbsp;
    const reader = new BitReader(rbsp, 2);
    const firstSlice = !!reader.u1();
    const nalType = (rbsp[0]! >> 1) & 0x3f;
    if (nalType >= 16 && nalType <= 23) reader.u1();
    const ppsId = reader.ue();
    const pps = decoder.ppsMap.get(ppsId);
    const sps = pps ? decoder.spsMap.get(pps.spsId) : undefined;
    if (!pps || !sps) throw new Error(`HEVC: missing parameter sets (pps ${ppsId})`);
    if (!sps.separateColourPlane) {
      if (separate) throw new Error('HEVC: picture mixes separate and combined colour-plane slices');
      combined = true;
      continue;
    }
    if (combined) throw new Error('HEVC: picture mixes separate and combined colour-plane slices');
    separate = true;
    let dependent = false;
    if (!firstSlice) {
      if (pps.dependentSliceSegmentsEnabled) dependent = !!reader.u1();
      const addressBits = Math.max(1, Math.ceil(Math.log2(pictureCtbCount(sps))));
      reader.u(addressBits);
    }
    if (!dependent) {
      reader.u(pps.numExtraSliceHeaderBits);
      reader.ue(); // slice_type
      if (pps.outputFlagPresent) reader.u1();
      currentPlane = reader.u(2);
      if (currentPlane > 2) throw new Error('HEVC: invalid colour_plane_id');
    }
    groups[currentPlane]!.push(nal);
  }
  return separate ? groups : null;
}

interface SaoParams {
  typeL: number; offL: Int32Array; eoL: number; bandL: number;
  typeC: number; offC: Int32Array; eoC: number; bandC: number;
  offC2: Int32Array; bandC2: number;
}
const NO_SAO: SaoParams = {
  typeL: 0, offL: new Int32Array(4), eoL: 0, bandL: 0,
  typeC: 0, offC: new Int32Array(4), eoC: 0, bandC: 0,
  offC2: new Int32Array(4), bandC2: 0,
};

interface SliceHeader {
  nal: HevcNal;
  nalType: number;
  firstSlice: boolean;
  dependent: boolean;
  address: number;
  pps: Pps;
  sps: Spt;
  sliceType: number;
  saoLuma: boolean;
  saoChroma: boolean;
  sliceQpDelta: number;
  sliceCbQp: number;
  sliceCrQp: number;
  cuChromaQpOffsetEnabled: boolean;
  deblockDisable: boolean;
  betaOffset: number;
  tcOffset: number;
  lfAcrossSlices: boolean;
  entryPointOffsets: number[];
  dataStart: number;
}

class SliceDecoder {
  // filled during construction
  private cabac!: Cabac;
  private sps!: Spt;
  private pps!: Pps;
  private frame!: DecodedFrame;
  private rbsp!: Uint8Array;
  private chromaFormat = 0;

  // maps
  private gridW = 0; private gridH = 0;
  private intraPredMode!: Int8Array;
  private intraPredModeC!: Int8Array;
  private intraChromaDerived!: Uint8Array;
  private qpMap!: Int8Array;
  private cbfYMap!: Uint8Array;
  private cbfCbMap!: Uint8Array;
  private cbfCrMap!: Uint8Array;
  private cuIdMap!: Int32Array;
  private tuIdMap!: Int32Array;
  private tqtBypassMap!: Uint8Array;
  private pcmMap!: Uint8Array;
  private tuSizeLog2Map!: Int8Array;
  private minTbAddrZs!: Int32Array;
  private ctDepthMap!: Uint8Array;
  private sliceIdMap!: Int32Array;
  private currentSliceId = 0;
  private tileIdMap!: Int32Array;
  private currentTileId = 0;
  private tileStartX = 0; private tileStartY = 0;
  private tbsW = 0;
  private widthInMinCbs = 0;

  private curTqBypass = false;
  private curCuLog2 = 0;
  private qpMapRows = 0;
  private curQp = 0;
  private nextCuId = 1;
  private nextTuId = 1;
  private currentQgX = -1; private currentQgY = -1;
  private prevQp = 0;
  private cuQpDeltaCoded = false;
  private cuQpDelta = 0;
  private sliceQpY = 26;
  private sliceStartX = 0; private sliceStartY = 0;
  private sliceCbQp = 0; private sliceCrQp = 0;
  private cuChromaQpOffsetEnabled = false;
  private cuChromaQpOffsetCoded = false;
  private cuCbQpOffset = 0; private cuCrQpOffset = 0;
  private scalingFactors: (Uint8Array | null)[][] | null = null;
  private statCoeff = new Int32Array(4);
  private lumaResidual = new Int32Array(32 * 32);
  private lumaResidualStride = 32;
  private componentResidual = new Int32Array(32 * 32);
  private crossPrediction = new Int32Array(32 * 32);
  private transformCoeff = new Int16Array(32 * 32);
  private transformIntermediate = new Int16Array(32 * 32);
  private csbfNeighbors = new Uint8Array(8 * 8);
  private subblockCoeffValue = new Int32Array(16);
  private subblockCoeffScanPos = new Uint8Array(16);
  private subblockCoeffMaxBase = new Uint8Array(16);
  private subblockSigns = new Uint8Array(16);
  private modeScratch = new Int8Array(4);
  private prevFlagScratch = new Uint8Array(4);
  private chromaIdxScratch = new Int8Array(4);
  private candidateModeScratch = [0, 0, 0];
  private intraCtx!: IntraCtx;
  private intraScratch = createIntraScratch();

  private dec: HevcDecoder;
  private debug: HevcDebugOptions;
  constructor(dec: HevcDecoder, debug: HevcDebugOptions) {
    this.dec = dec;
    this.debug = debug;
  }

  private tr(sym: string, val: number | string): void {
    debugWrite(`[tr] ${sym}=${val}\n`);
  }

  decode(nals: HevcNal[]): DecodedFrame {
    const nal = nals[0]!;
    const rbsp = nal.rbsp;
    this.rbsp = rbsp;
    const r = new BitReader(rbsp, 2);
    const firstSlice = r.u1();
    const nalType = (rbsp[0]! >> 1) & 0x3F;
    const isIRAP = nalType >= 16 && nalType <= 23;
    if (isIRAP) r.u1();
    const ppsId = r.ue();
    const pps = this.dec.ppsMap.get(ppsId);
    const sps = pps ? this.dec.spsMap.get(pps.spsId) : undefined;
    if (!pps || !sps) throw new Error(`HEVC: missing parameter sets (pps ${ppsId})`);
    this.sps = sps; this.pps = pps;
    assertDimensions(sps.width, sps.height, this.dec.limits, 'HEVC frame');
    this.dec.decodedSps = sps;
    this.chromaFormat = sps.separateColourPlane ? 0 : sps.chromaFormatIdc;

    if (!firstSlice) {
      if (pps.dependentSliceSegmentsEnabled) r.u1();
      const numCtb = Math.ceil(sps.width / (1 << sps.log2CtbSize)) * Math.ceil(sps.height / (1 << sps.log2CtbSize));
      const addrBits = Math.max(1, Math.ceil(Math.log2(numCtb)));
      const addr = r.u(addrBits);
      if (addr !== 0) throw new Error('HEVC: first slice segment must start at address 0');
    }
    r.u(pps.numExtraSliceHeaderBits);
    const sliceType = r.ue();
    if (sliceType !== 2) throw new Error('HEVC: only I-slices supported (still images)');
    if (pps.outputFlagPresent) r.u1();
    if (sps.separateColourPlane) r.u(2);

    if (nalType !== 19 && nalType !== 20) {
      this.skipNonIdrReferenceSyntax(r, sps);
    }

    let saoLuma = false, saoChroma = false;
    if (sps.saoEnabled) {
      saoLuma = !!r.u1();
      if (this.chromaFormat !== 0) saoChroma = !!r.u1();
    }
    const sliceQpDelta = r.se();
    if (pps.sliceChromaQpOffsetsPresent) {
      this.sliceCbQp = r.se();
      this.sliceCrQp = r.se();
    }
    const cuChromaQpOffsetEnabled = pps.chromaQpOffsetListEnabled ? !!r.u1() : false;
    let deblockDisable = pps.picDisableDeblocking;
    let betaOffset = pps.betaOffsetDiv2, tcOffset = pps.tcOffsetDiv2;
    let lfAcrossSlices = pps.loopFilterAcrossSlices;
    if (pps.deblockingOverrideEnabled) {
      if (r.u1()) { // slice_deblocking_filter_disabled_flag
        deblockDisable = !!r.u1();
        if (!deblockDisable) { betaOffset = r.se(); tcOffset = r.se(); }
      }
    }
    if (pps.loopFilterAcrossSlices && (saoLuma || saoChroma || !deblockDisable)) {
      lfAcrossSlices = !!r.u1();
    }
    const entryPointOffsets: number[] = [];
    if (pps.tilesEnabled || pps.entropyCodingSync) {
      const n = r.ue();
      const maximum = pictureCtbCount(sps);
      if (n > maximum) throw new Error('HEVC: too many slice entry points');
      if (n) {
        const len = r.ue() + 1;
        if (len > 32) throw new Error('HEVC: slice entry-point offset is too wide');
        let acc = 0;
        for (let i = 0; i < n; i++) { acc += r.u(len) + 1; entryPointOffsets.push(acc); }
      }
    }
    if (pps.sliceHeaderExtPresent) {
      const extLen = r.ue();
      r.u(extLen * 8);
    }
    r.u1(); // alignment_bit_equal_to_one
    r.byteAlign();
    const dataStart = r.pos >> 3;
    if (this.debug.trace) this.tr('slice_data_start', dataStart);
    this.adjustEntryPointOffsets(nal, dataStart, entryPointOffsets);

    // ---------- frame setup ----------
    // Edge CTUs of pictures whose size is not a multiple of the coding grid
    // decode partial blocks; padding keeps their overhang from wrapping into
    // the following row. Spec cap for a CTB is 64; a few extra columns/rows
    // also absorb deblock filtering taps past the last real sample.
    const ctbPad = Math.min(1 << sps.log2CtbSize, 64) + 8;
    const frame = new DecodedFrame(
      sps.width, sps.height, sps.bitDepthLuma, this.chromaFormat, sps.bitDepthChroma, ctbPad,
    );
    this.frame = frame;
    const initQp = checkedSliceQp(sps.bitDepthLuma, pps.initQpMinus26, sliceQpDelta);
    this.sliceQpY = initQp;
    this.sliceStartX = 0; this.sliceStartY = 0;
    this.curQp = initQp;
    this.prevQp = initQp;

    const minCbLog2 = sps.log2MinCbSize;
    this.widthInMinCbs = (sps.width + (1 << minCbLog2) - 1) >> minCbLog2;
    const heightInMinCbs = (sps.height + (1 << minCbLog2) - 1) >> minCbLog2;

    this.gridW = (sps.width + 3) >> 2;
    this.gridH = (sps.height + 3) >> 2;
    const gridW = this.gridW, gridH = this.gridH;
    this.intraPredMode = new Int8Array(gridW * gridH).fill(-1);
    this.intraPredModeC = new Int8Array(gridW * gridH).fill(-1);
    this.intraChromaDerived = new Uint8Array(gridW * gridH);
    this.qpMap = new Int8Array(this.widthInMinCbs * heightInMinCbs).fill(initQp);
    this.qpMapRows = heightInMinCbs;
    this.cbfYMap = new Uint8Array(gridW * gridH);
    this.cbfCbMap = new Uint8Array(gridW * gridH);
    this.cbfCrMap = new Uint8Array(gridW * gridH);
    this.cuIdMap = new Int32Array(gridW * gridH);
    this.tuIdMap = new Int32Array(gridW * gridH);
    this.tqtBypassMap = new Uint8Array(gridW * gridH);
    this.pcmMap = new Uint8Array(gridW * gridH);
    this.tuSizeLog2Map = new Int8Array(gridW * gridH).fill(2);
    this.ctDepthMap = new Uint8Array(gridW * gridH);
    this.sliceIdMap = new Int32Array(gridW * gridH).fill(-1);
    this.tileIdMap = new Int32Array(gridW * gridH).fill(-1);

    // scaling factors
    this.scalingFactors = this.debug.disableScaling ? null : buildScalingFactors(sps, pps);
    this.statCoeff.fill(0);

    // z-scan table (6.5.2)
    this.tbsW = gridW;
    const ctbShift = sps.log2CtbSize - 2;
    const picWidthInCtbs = Math.ceil(sps.width / (1 << sps.log2CtbSize));
    this.minTbAddrZs = new Int32Array(gridW * gridH);
    for (let y = 0; y < gridH; y++) {
      for (let x = 0; x < gridW; x++) {
        const ctbX = x >> ctbShift, ctbY = y >> ctbShift;
        let p = 0;
        for (let i = 0; i < ctbShift; i++) {
          const m = 1 << i;
          if (m & x) p += m * m;
          if (m & y) p += 2 * m * m;
        }
        this.minTbAddrZs[x + y * gridW] = ((ctbY * picWidthInCtbs + ctbX) << (ctbShift * 2)) | p;
      }
    }

    this.intraCtx = {
      planes: frame.planes, chromaArrayType: this.chromaFormat,
      strongIntraSmoothing: sps.strongIntraSmoothing,
      intraSmoothingDisabled: sps.intraSmoothingDisabled,
      bitDepth: sps.bitDepthLuma,
      minTbAddrZs: this.minTbAddrZs, picWidthInTbs: gridW, log2MinTb: 2,
      sliceIdMap: this.sliceIdMap, currentSliceId: this.currentSliceId,
      tileIdMap: this.tileIdMap, currentTileId: this.currentTileId,
      scratch: this.intraScratch, tuDebug: this.debug.tuDebug,
    };

    // tiles
    const ctbSize = 1 << sps.log2CtbSize;
    const ctbCols = picWidthInCtbs, ctbRows = Math.ceil(sps.height / ctbSize);
    if (pps.numTileCols > ctbCols || pps.numTileRows > ctbRows) {
      throw new Error('HEVC: tile grid exceeds the coded picture');
    }
    if (pps.colWidths && pps.colWidths.reduce((sum, value) => sum + value, 0) >= ctbCols) {
      throw new Error('HEVC: explicit tile-column widths exceed the coded picture');
    }
    if (pps.rowHeights && pps.rowHeights.reduce((sum, value) => sum + value, 0) >= ctbRows) {
      throw new Error('HEVC: explicit tile-row heights exceed the coded picture');
    }
    const tileColBd: number[] = [0];
    const tileRowBd: number[] = [0];
    if (pps.tilesEnabled) {
      if (pps.colWidths) {
        let x = 0;
        for (let i = 0; i < pps.numTileCols - 1; i++) { x += pps.colWidths[i]!; tileColBd.push(x); }
        tileColBd.push(ctbCols);
      } else {
        for (let i = 1; i < pps.numTileCols; i++) tileColBd.push(Math.floor(i * ctbCols / pps.numTileCols));
        tileColBd.push(ctbCols);
      }
      if (pps.rowHeights) {
        let y = 0;
        for (let i = 0; i < pps.numTileRows - 1; i++) { y += pps.rowHeights[i]!; tileRowBd.push(y); }
        tileRowBd.push(ctbRows);
      } else {
        for (let i = 1; i < pps.numTileRows; i++) tileRowBd.push(Math.floor(i * ctbRows / pps.numTileRows));
        tileRowBd.push(ctbRows);
      }
    } else { tileColBd.push(ctbCols); tileRowBd.push(ctbRows); }

    let tileId = 0;
    for (let tr = 0; tr < tileRowBd.length - 1; tr++) {
      for (let tc = 0; tc < tileColBd.length - 1; tc++, tileId++) {
        for (let y = tileRowBd[tr]!; y < tileRowBd[tr + 1]!; y++) {
          for (let x = tileColBd[tc]!; x < tileColBd[tc + 1]!; x++) {
            this.markGrid(this.tileIdMap, x * ctbSize, y * ctbSize, ctbSize, tileId);
          }
        }
      }
    }

    const scanOrder: number[] = [];
    for (let tr = 0; tr < tileRowBd.length - 1; tr++) {
      for (let tc = 0; tc < tileColBd.length - 1; tc++) {
        for (let y = tileRowBd[tr]!; y < tileRowBd[tr + 1]!; y++) {
          for (let x = tileColBd[tc]!; x < tileColBd[tc + 1]!; x++) {
            scanOrder.push(y * ctbCols + x);
          }
        }
      }
    }

    const saoParams: SaoParams[] = new Array(ctbCols * ctbRows);
    let previousHeader: SliceHeader = {
      nal, nalType, firstSlice: true, dependent: false, address: 0, pps, sps, sliceType,
      saoLuma, saoChroma, sliceQpDelta, sliceCbQp: this.sliceCbQp, sliceCrQp: this.sliceCrQp,
      cuChromaQpOffsetEnabled,
      deblockDisable, betaOffset, tcOffset, lfAcrossSlices, entryPointOffsets, dataStart,
    };
    this.currentSliceId = 0;
    this.decodeAdditionalSlice(previousHeader, scanOrder, ctbCols, ctbRows, ctbSize, tileColBd, tileRowBd, saoParams);
    let sliceRegionId = 0;
    for (let i = 1; i < nals.length; i++) {
      const header = this.parseSliceHeader(nals[i]!, previousHeader);
      if (header.firstSlice) throw new Error('HEVC: image item contains more than one coded picture');
      if (!header.dependent) { previousHeader = header; sliceRegionId++; }
      this.currentSliceId = sliceRegionId;
      saoLuma ||= header.saoLuma;
      saoChroma ||= header.saoChroma;
      this.decodeAdditionalSlice(header, scanOrder, ctbCols, ctbRows, ctbSize, tileColBd, tileRowBd, saoParams);
    }
    this.assertSliceCoverage(ctbCols, ctbRows, ctbSize);

    const dbInfo: DeblockInfo = {
      qpMap: this.qpMap, widthInMinCbs: this.widthInMinCbs, minCbLog2: sps.log2MinCbSize,
      cbfYMap: this.cbfYMap, cbfCbMap: this.cbfCbMap, cbfCrMap: this.cbfCrMap,
      cuIdMap: this.cuIdMap, tuIdMap: this.tuIdMap, tuSizeLog2Map: this.tuSizeLog2Map,
      pcmMap: this.pcmMap, pcmLoopFilterDisable: sps.pcmLoopFilterDisable,
      sliceIdMap: this.sliceIdMap, tileIdMap: this.tileIdMap,
      loopFilterAcrossSlices: lfAcrossSlices, loopFilterAcrossTiles: pps.loopFilterAcrossTiles,
      gridW, gridH, bitDepth: sps.bitDepthLuma, chromaBitDepth: sps.bitDepthChroma,
      betaOffsetDiv2: betaOffset, tcOffsetDiv2: tcOffset,
      cbQpOffset: pps.picCbQpOffset + this.sliceCbQp,
      crQpOffset: pps.picCrQpOffset + this.sliceCrQp,
      disable: deblockDisable, chromaFormat: this.chromaFormat,
      sliceBorders: null, ctbSizeLog2: sps.log2CtbSize, tileColBd, tileRowBd,
    };
    if (!deblockDisable && !this.debug.disableDeblock) applyDeblock(frame, dbInfo);
    if ((saoLuma || saoChroma) && !this.debug.disableSao) {
      applySao(frame, saoParams, tileColBd, tileRowBd, ctbCols, ctbRows, sps, saoLuma, saoChroma,
        pps.loopFilterAcrossTiles, sps.pcmLoopFilterDisable ? this.pcmMap : null, this.gridW,
        this.sliceIdMap, lfAcrossSlices);
    }
    return frame;
  }

  private assertSliceCoverage(ctbCols: number, ctbRows: number, ctbSize: number): void {
    for (let row = 0; row < ctbRows; row++) {
      for (let col = 0; col < ctbCols; col++) {
        const gridX = (col * ctbSize) >> 2;
        const gridY = (row * ctbSize) >> 2;
        if (this.sliceIdMap[gridY * this.gridW + gridX] === -1) {
          throw new Error(`HEVC: slice segments do not cover CTB ${row * ctbCols + col}`);
        }
      }
    }
  }

  private parseSliceHeader(nal: HevcNal, inherited: SliceHeader): SliceHeader {
    const rbsp = nal.rbsp;
    const r = new BitReader(rbsp, 2);
    const firstSlice = !!r.u1();
    const nalType = (rbsp[0]! >> 1) & 0x3f;
    if (nalType >= 16 && nalType <= 23) r.u1(); // no_output_of_prior_pics_flag
    const ppsId = r.ue();
    const pps = this.dec.ppsMap.get(ppsId);
    const sps = pps ? this.dec.spsMap.get(pps.spsId) : undefined;
    if (!pps || !sps) throw new Error(`HEVC: missing parameter sets (pps ${ppsId})`);

    let dependent = false, address = 0;
    if (!firstSlice) {
      if (pps.dependentSliceSegmentsEnabled) dependent = !!r.u1();
      const ctbSize = 1 << sps.log2CtbSize;
      const numCtb = Math.ceil(sps.width / ctbSize) * Math.ceil(sps.height / ctbSize);
      const addrBits = Math.ceil(Math.log2(numCtb));
      if (addrBits) address = r.u(addrBits);
      if (this.debug.trace) this.tr('slice_segment_address', address);
    }

    let sliceType = inherited.sliceType;
    let saoLuma = inherited.saoLuma, saoChroma = inherited.saoChroma;
    let sliceQpDelta = inherited.sliceQpDelta;
    let sliceCbQp = inherited.sliceCbQp, sliceCrQp = inherited.sliceCrQp;
    let cuChromaQpOffsetEnabled = inherited.cuChromaQpOffsetEnabled;
    let deblockDisable = inherited.deblockDisable;
    let betaOffset = inherited.betaOffset, tcOffset = inherited.tcOffset;
    let lfAcrossSlices = inherited.lfAcrossSlices;

    if (!dependent) {
      r.u(pps.numExtraSliceHeaderBits);
      sliceType = r.ue();
      if (sliceType !== 2) throw new Error('HEVC: only I-slices supported (still images)');
      if (pps.outputFlagPresent) r.u1();
      if (sps.separateColourPlane) r.u(2);
      if (nalType !== 19 && nalType !== 20) {
        this.skipNonIdrReferenceSyntax(r, sps);
      }
      saoLuma = false; saoChroma = false;
      if (sps.saoEnabled) {
        saoLuma = !!r.u1();
        if ((sps.separateColourPlane ? 0 : sps.chromaFormatIdc) !== 0) saoChroma = !!r.u1();
      }
      sliceQpDelta = r.se();
      sliceCbQp = 0; sliceCrQp = 0;
      if (pps.sliceChromaQpOffsetsPresent) {
        sliceCbQp = r.se();
        sliceCrQp = r.se();
      }
      cuChromaQpOffsetEnabled = pps.chromaQpOffsetListEnabled ? !!r.u1() : false;
      deblockDisable = pps.picDisableDeblocking;
      betaOffset = pps.betaOffsetDiv2;
      tcOffset = pps.tcOffsetDiv2;
      if (pps.deblockingOverrideEnabled && r.u1()) {
        deblockDisable = !!r.u1();
        if (!deblockDisable) { betaOffset = r.se(); tcOffset = r.se(); }
      }
      lfAcrossSlices = pps.loopFilterAcrossSlices;
      if (pps.loopFilterAcrossSlices && (saoLuma || saoChroma || !deblockDisable)) lfAcrossSlices = !!r.u1();
    }

    const entryPointOffsets: number[] = [];
    if (pps.tilesEnabled || pps.entropyCodingSync) {
      const n = r.ue();
      const maximum = pictureCtbCount(sps);
      if (n > maximum) throw new Error('HEVC: too many slice entry points');
      if (n) {
        const len = r.ue() + 1;
        if (len > 32) throw new Error('HEVC: slice entry-point offset is too wide');
        let acc = 0;
        for (let i = 0; i < n; i++) { acc += r.u(len) + 1; entryPointOffsets.push(acc); }
      }
    }
    if (pps.sliceHeaderExtPresent) {
      const extLen = r.ue();
      r.u(extLen * 8);
    }
    r.u1(); // alignment_bit_equal_to_one
    r.byteAlign();
    const dataStart = r.pos >> 3;
    this.adjustEntryPointOffsets(nal, dataStart, entryPointOffsets);
    return {
      nal, nalType, firstSlice, dependent, address, pps, sps, sliceType,
      saoLuma, saoChroma, sliceQpDelta, sliceCbQp, sliceCrQp,
      cuChromaQpOffsetEnabled,
      deblockDisable, betaOffset, tcOffset, lfAcrossSlices, entryPointOffsets, dataStart,
    };
  }

  private skipNonIdrReferenceSyntax(r: BitReader, sps: Spt): void {
    r.u(sps.log2MaxPocLsb);
    const useSpsRps = sps.numShortTermRefPicSets > 0 ? !!r.u1() : false;
    if (useSpsRps) {
      if (sps.numShortTermRefPicSets > 1) r.u(Math.ceil(Math.log2(sps.numShortTermRefPicSets)));
    } else {
      const counts = sps.stRpsNumDeltaPocs.slice();
      skipShortTermRefPicSet(r, sps.numShortTermRefPicSets, counts, true);
    }
    if (sps.longTermRefPicsPresent) {
      const numLtSps = sps.numLongTermRefPicsSps > 0 ? r.ue() : 0;
      const numLtPics = r.ue();
      if (numLtSps > sps.numLongTermRefPicsSps || numLtPics > 32) {
        throw new Error('HEVC: too many long-term reference pictures in slice header');
      }
      for (let i = 0; i < numLtSps + numLtPics; i++) {
        if (i < numLtSps) {
          if (sps.numLongTermRefPicsSps > 1) r.u(Math.ceil(Math.log2(sps.numLongTermRefPicsSps)));
        } else {
          r.u(sps.log2MaxPocLsb);
          r.u1();
        }
        if (r.u1()) r.ue();
      }
    }
    if (sps.temporalMvpEnabled) r.u1();
  }

  private adjustEntryPointOffsets(nal: HevcNal, dataStart: number, offsets: number[]): void {
    if (!offsets.length || !nal.skippedBytes?.length) return;
    const dataStartEbsp = rbspOffsetToEbsp(dataStart, nal.skippedBytes);
    for (let i = 0; i < offsets.length; i++) {
      const rawOffset = offsets[i]!;
      const rawTarget = dataStartEbsp + rawOffset;
      const skipped = countSkippedBytesInRange(nal.skippedBytes, dataStartEbsp, rawTarget);
      offsets[i] = rawOffset - skipped;
    }
  }

  private decodeAdditionalSlice(
    h: SliceHeader, scanOrder: number[], ctbCols: number, ctbRows: number, ctbSize: number,
    tileColBd: number[], tileRowBd: number[], saoParams: SaoParams[],
  ): void {
    this.sps = h.sps; this.pps = h.pps; this.rbsp = h.nal.rbsp;
    this.chromaFormat = h.sps.separateColourPlane ? 0 : h.sps.chromaFormatIdc;
    this.sliceCbQp = h.sliceCbQp; this.sliceCrQp = h.sliceCrQp;
    this.cuChromaQpOffsetEnabled = h.cuChromaQpOffsetEnabled;
    this.cuChromaQpOffsetCoded = false; this.cuCbQpOffset = 0; this.cuCrQpOffset = 0;
    validateSaoOffsetScale(h.pps.log2SaoOffsetScaleLuma, h.sps.bitDepthLuma, 'luma');
    validateSaoOffsetScale(h.pps.log2SaoOffsetScaleChroma, h.sps.bitDepthChroma, 'chroma');
    if (!h.dependent) {
      this.sliceQpY = checkedSliceQp(h.sps.bitDepthLuma, h.pps.initQpMinus26, h.sliceQpDelta);
      this.sliceStartX = (h.address % ctbCols) * ctbSize;
      this.sliceStartY = Math.floor(h.address / ctbCols) * ctbSize;
      this.curQp = this.sliceQpY; this.prevQp = this.sliceQpY;
    }
    this.currentQgX = -1; this.currentQgY = -1;
    this.cuQpDeltaCoded = false; this.cuQpDelta = 0;
    if (!h.dependent) this.statCoeff.fill(0);

    // Independent slice segments initialize contexts; dependent segments carry
    // the preceding segment's contexts but restart the arithmetic engine.
    const inheritedContexts = h.dependent ? this.cabac.saveContexts() : null;
    this.cabac = new Cabac(h.nal.rbsp, h.dataStart, this.debug.cabac);
    if (inheritedContexts) this.cabac.loadContexts(inheritedContexts);
    else this.cabac.initContexts(0, this.sliceQpY);

    if (h.pps.tilesEnabled || h.pps.entropyCodingSync) {
      this.decodeSliceSubstreams(h, ctbCols, ctbRows, ctbSize, tileColBd, tileRowBd, saoParams);
      return;
    }
    let scan = scanOrder.indexOf(h.address);
    if (scan < 0) throw new Error(`HEVC: invalid slice_segment_address ${h.address}`);
    for (; scan < scanOrder.length; scan++) {
      const ctbAddr = scanOrder[scan]!;
      const col = ctbAddr % ctbCols, row = Math.floor(ctbAddr / ctbCols);
      this.markGrid(this.sliceIdMap, col * ctbSize, row * ctbSize, ctbSize, this.currentSliceId);
      this.currentQgX = -1; this.currentQgY = -1;
      if (h.saoLuma || h.saoChroma) {
        const sliceLeft = ctbAddr - 1 < h.address;
        const sliceTop = ctbAddr - ctbCols < h.address;
        this.readSao(ctbAddr, col === 0 || sliceLeft, row === 0 || sliceTop,
          saoParams, h.saoLuma, h.saoChroma, ctbCols);
      } else saoParams[ctbAddr] = NO_SAO;
      this.decodeCtu(col * ctbSize, row * ctbSize, h.sps.log2CtbSize, 0);
      if (this.cabac.decodeTerminate() === 1) break;
    }
  }

  private decodeSliceSubstreams(
    h: SliceHeader, ctbCols: number, ctbRows: number, ctbSize: number,
    tileColBd: number[], tileRowBd: number[], saoParams: SaoParams[],
  ): void {
    interface Substream {
      ctbs: number[];
      tileId: number;
      tileLeft: number; tileTop: number;
      rowInTile: number;
    }
    const streams: Substream[] = [];
    let tileId = 0;
    for (let tr = 0; tr < tileRowBd.length - 1; tr++) {
      const top = tileRowBd[tr]!, bottom = tileRowBd[tr + 1]!;
      for (let tc = 0; tc < tileColBd.length - 1; tc++, tileId++) {
        const left = tileColBd[tc]!, right = tileColBd[tc + 1]!;
        if (h.pps.entropyCodingSync) {
          for (let row = top; row < bottom; row++) {
            const ctbs: number[] = [];
            for (let col = left; col < right; col++) ctbs.push(row * ctbCols + col);
            streams.push({ ctbs, tileId, tileLeft: left, tileTop: top, rowInTile: row - top });
          }
        } else {
          const ctbs: number[] = [];
          for (let row = top; row < bottom; row++) {
            for (let col = left; col < right; col++) ctbs.push(row * ctbCols + col);
          }
          streams.push({ ctbs, tileId, tileLeft: left, tileTop: top, rowInTile: 0 });
        }
      }
    }

    let startStream = streams.findIndex(stream => stream.ctbs.indexOf(h.address) >= 0);
    if (startStream < 0) throw new Error(`HEVC: invalid slice_segment_address ${h.address}`);
    const inheritedContexts = h.dependent ? this.cabac.saveContexts() : null;
    const wavefrontContexts = new Map<number, { states: Uint8Array; mps: Uint8Array }>();
    const starts = sliceSubstreamStarts(h.dataStart, h.entryPointOffsets, h.nal.rbsp.length);
    if (starts.length > streams.length - startStream) {
      throw new Error('HEVC: slice has more entry points than remaining tile/WPP substreams');
    }

    for (let entry = 0; entry < starts.length; entry++) {
      const streamIndex = startStream + entry;
      const stream = streams[streamIndex]!;
      const start = starts[entry]!;
      this.cabac = new Cabac(h.nal.rbsp, start, this.debug.cabac);
      if (entry === 0 && inheritedContexts) {
        this.cabac.loadContexts(inheritedContexts);
      } else if (h.pps.entropyCodingSync && stream.rowInTile > 0 && wavefrontContexts.has(stream.tileId)) {
        const sync = wavefrontContexts.get(stream.tileId)!;
        if (this.debug.trace) this.tr('wpp_load_ctx8', `${sync.states[CTX.PART_MODE]}/${sync.mps[CTX.PART_MODE]}`);
        this.cabac.loadContexts(sync);
      } else {
        this.cabac.initContexts(0, this.sliceQpY);
      }

      this.currentTileId = stream.tileId;
      this.tileStartX = stream.tileLeft * ctbSize;
      this.tileStartY = stream.tileTop * ctbSize;
      let first = entry === 0 ? stream.ctbs.indexOf(h.address) : 0;
      if (first < 0) first = 0;
      for (let pos = first; pos < stream.ctbs.length; pos++) {
        const ctbAddr = stream.ctbs[pos]!;
        const col = ctbAddr % ctbCols, row = Math.floor(ctbAddr / ctbCols);
        const x = col * ctbSize, y = row * ctbSize;
        this.markGrid(this.sliceIdMap, x, y, ctbSize, this.currentSliceId);
        this.currentQgX = -1; this.currentQgY = -1;
        if (h.saoLuma || h.saoChroma) {
          const noLeft = col === stream.tileLeft || !this.sameSliceAt(x - 1, y);
          const noTop = row === stream.tileTop || !this.sameSliceAt(x, y - 1);
          this.readSao(ctbAddr, noLeft, noTop, saoParams, h.saoLuma, h.saoChroma, ctbCols);
        } else saoParams[ctbAddr] = NO_SAO;
        this.decodeCtu(x, y, h.sps.log2CtbSize, 0);
        if (h.pps.entropyCodingSync && pos === first + 1) {
          const sync = this.cabac.saveContexts();
          if (this.debug.trace) this.tr('wpp_save_ctx8', `${sync.states[CTX.PART_MODE]}/${sync.mps[CTX.PART_MODE]}`);
          wavefrontContexts.set(stream.tileId, sync);
        }
        // A terminating bin ends this subset. The number of subsets processed
        // is bounded by the entry points signalled in this slice header.
        if (this.cabac.decodeTerminate() === 1) break;
      }
    }
  }

  // ---------------- SAO syntax ----------------
  private readSao(ctbAddr: number, atTileLeft: boolean, atTileTop: boolean, params: SaoParams[], lumaOn: boolean, chromaOn: boolean, ctbCols: number) {
    const cabac = this.cabac;
    const sps = this.sps;
    const leftAvail = !atTileLeft && ctbAddr > 0 && params[ctbAddr - 1] !== undefined;
    const upAvail = !atTileTop && ctbAddr >= ctbCols && params[ctbAddr - ctbCols] !== undefined;
    if (this.debug.trace) this.tr('sao_avail', ctbAddr + ' L' + leftAvail + ' U' + upAvail);
    let merge = false;
    if (leftAvail) merge = !!cabac.decodeBin(CTX.SAO_MERGE_FLAG);
    if (!merge && upAvail) merge = !!cabac.decodeBin(CTX.SAO_MERGE_FLAG);
    if (this.debug.trace) this.tr('sao_type_idx', 0);
    if (merge && leftAvail) { params[ctbAddr] = params[ctbAddr - 1]!; return; }
    if (merge) { params[ctbAddr] = params[ctbAddr - ctbCols]!; return; }

    // spec 7.3.7.3: for 4:2:0, SAO is parsed for cIdx 0 (Y), 1 (Cb), 2 (Cr).
    // Cb and Cr share sao_type_idx / band / eo_class, but each has its own
    // sao_offset_abs (and sign bits when BO). Cb/Cr class+band are coded at cIdx=1 only.
    // sao_type_idx TR: first bin regular; second bin bypass: 1 -> EO, 0 -> BO.
    const readAbsOffsets = (bitDepth: number): Int32Array => {
      const cMax = (1 << (Math.min(bitDepth, 10) - 5)) - 1;
      const offs = new Int32Array(4);
      for (let i = 0; i < 4; i++) {
        let v = 0;
        while (v < cMax && cabac.decodeBypass()) v++;
        offs[i] = v;
      }
      return offs;
    };
    const p: SaoParams = { ...NO_SAO, offL: new Int32Array(4), offC: new Int32Array(4), offC2: new Int32Array(4) };
    let typeC = 0;
    // cIdx = 0 (luma)
    if (lumaOn) {
      const b0 = cabac.decodeBin(CTX.SAO_TYPE_IDX);
      if (b0) {
        const isEo = cabac.decodeBypass();
        const abs = readAbsOffsets(sps.bitDepthLuma);
        if (isEo) {
          for (let i = 0; i < 4; i++) abs[i] = i < 2 ? abs[i]! : -abs[i]!;
          p.typeL = 1; p.offL = abs; p.eoL = cabac.readFL(2);
        } else {
          for (let i = 0; i < 4; i++) if (abs[i]) abs[i] = cabac.decodeBypass() ? -abs[i]! : abs[i]!;
          p.typeL = 2; p.offL = abs; p.bandL = cabac.readFL(5);
        }
      }
    }
    // cIdx = 1 (Cb) and cIdx = 2 (Cr)
    if (chromaOn && this.chromaFormat !== 0) {
      const b0c = cabac.decodeBin(CTX.SAO_TYPE_IDX);
      if (b0c) {
        const isEo = cabac.decodeBypass();
        typeC = isEo ? 1 : 2;
        p.typeC = typeC;
        const abs = readAbsOffsets(sps.bitDepthChroma);
        if (isEo) {
          for (let i = 0; i < 4; i++) abs[i] = i < 2 ? abs[i]! : -abs[i]!;
          p.offC = abs; p.eoC = cabac.readFL(2);
        } else {
          for (let i = 0; i < 4; i++) if (abs[i]) abs[i] = cabac.decodeBypass() ? -abs[i]! : abs[i]!;
          p.offC = abs; p.bandC = cabac.readFL(5);
        }
        // Cr: own offsets (+ signs for BO), no type/class/band
        const abs2 = readAbsOffsets(sps.bitDepthChroma);
        if (isEo) {
          for (let i = 0; i < 4; i++) abs2[i] = i < 2 ? abs2[i]! : -abs2[i]!;
        } else {
          for (let i = 0; i < 4; i++) if (abs2[i]) abs2[i] = cabac.decodeBypass() ? -abs2[i]! : abs2[i]!;
          p.bandC2 = cabac.readFL(5);
        }
        p.offC2 = abs2;
      }
    }
    const scaleOffsets = (values: Int32Array, shift: number) => {
      if (shift) for (let i = 0; i < values.length; i++) values[i] = values[i]! << shift;
    };
    scaleOffsets(p.offL, this.pps.log2SaoOffsetScaleLuma);
    scaleOffsets(p.offC, this.pps.log2SaoOffsetScaleChroma);
    scaleOffsets(p.offC2, this.pps.log2SaoOffsetScaleChroma);
    params[ctbAddr] = p;
  }

  // ---------------- CTU / CU / TU ----------------
  private decodeCtu(x0: number, y0: number, log2Cb: number, ctDepth: number) {
    const sps = this.sps;
    const qgLog2 = sps.log2CtbSize - this.pps.diffCuQpDeltaDepth;
    if (this.pps.cuQpDeltaEnabled && log2Cb >= qgLog2) {
      this.cuQpDeltaCoded = false;
      this.cuQpDelta = 0;
    }
    const chromaQgLog2 = sps.log2CtbSize - this.pps.diffCuChromaQpOffsetDepth;
    if (this.cuChromaQpOffsetEnabled && log2Cb >= chromaQgLog2) {
      this.cuChromaQpOffsetCoded = false;
      this.cuCbQpOffset = 0; this.cuCrQpOffset = 0;
    }
    const fits = x0 + (1 << log2Cb) <= sps.width && y0 + (1 << log2Cb) <= sps.height;
    let split: number;
    if (fits && log2Cb > sps.log2MinCbSize) {
      // context: left/top neighbor quadtree nodes with deeper splits
      const condL = x0 > 0 && this.sameSliceAt(x0 - 1, y0) && this.ctDepthAt(x0 - 1, y0) > ctDepth ? 1 : 0;
      const condA = y0 > 0 && this.sameSliceAt(x0, y0 - 1) && this.ctDepthAt(x0, y0 - 1) > ctDepth ? 1 : 0;
      split = this.cabac.decodeBin(CTX.SPLIT_CU_FLAG + condL + condA);
      if (this.debug.trace) this.tr('split_cu_flag', split);
    } else {
      split = log2Cb > sps.log2MinCbSize ? 1 : 0;
    }
    this.setCtDepth(x0, y0, 1 << log2Cb, ctDepth);
    if (split) {
      const half = 1 << (log2Cb - 1);
      this.decodeCtu(x0, y0, log2Cb - 1, ctDepth + 1);
      if (x0 + half < sps.width) this.decodeCtu(x0 + half, y0, log2Cb - 1, ctDepth + 1);
      if (y0 + half < sps.height) {
        this.decodeCtu(x0, y0 + half, log2Cb - 1, ctDepth + 1);
        if (x0 + half < sps.width) this.decodeCtu(x0 + half, y0 + half, log2Cb - 1, ctDepth + 1);
      }
    } else {
      this.decodeCu(x0, y0, log2Cb);
    }
  }

  private ctDepthAt(x: number, y: number): number {
    const gx = Math.min(this.gridW - 1, x >> 2), gy = Math.min(this.gridH - 1, y >> 2);
    return this.ctDepthMap[gy * this.gridW + gx]!;
  }

  private sameSliceAt(x: number, y: number): boolean {
    if (x < 0 || y < 0 || x >= this.sps.width || y >= this.sps.height) return false;
    const i = (y >> 2) * this.gridW + (x >> 2);
    return this.sliceIdMap[i] === this.currentSliceId && this.tileIdMap[i] === this.currentTileId;
  }

  private setCtDepth(x0: number, y0: number, size: number, depth: number) {
    const gs = size >> 2;
    for (let y = 0; y < gs && ((y0 >> 2) + y) < this.gridH; y++) {
      for (let x = 0; x < gs && ((x0 >> 2) + x) < this.gridW; x++) {
        this.ctDepthMap[((y0 >> 2) + y) * this.gridW + (x0 >> 2) + x] = depth;
      }
    }
  }

  private decodeCu(x0: number, y0: number, log2Cb: number) {
    const sps = this.sps;
    const cabac = this.cabac;
    const nS = 1 << log2Cb;
    this.curCuLog2 = log2Cb;
    this.deriveQuantizationParameters(x0, y0);
    this.curTqBypass = this.pps.transquantBypassEnabled ? !!cabac.decodeBin(CTX.CU_TRANSQUANT_BYPASS_FLAG) : false;

    let partNxN = false;
    if (log2Cb === sps.log2MinCbSize) {
      const pb = cabac.decodeBin(CTX.PART_MODE);
      partNxN = !pb;
      if (this.debug.trace) this.tr('part_mode', partNxN ? 'NxN' : '2Nx2N');
    }

    if (!partNxN && sps.pcmEnabled && log2Cb >= sps.log2MinPcmCbSize && log2Cb <= sps.log2MaxPcmCbSize) {
      const pcmFlag = this.cabac.decodeTerminate();
      if (this.debug.trace) this.tr('pcm_flag', pcmFlag);
      if (pcmFlag) {
        const cuId = this.nextCuId++;
        this.markGrid(this.cuIdMap, x0, y0, nS, cuId);
        this.markGrid(this.pcmMap, x0, y0, nS, 1);
        this.readPcmSamples(x0, y0, log2Cb);
        return;
      }
    }

    const numPU = partNxN ? 4 : 1;
    const puSize = numPU === 1 ? nS : nS >> 1;
    const modes = this.modeScratch;
    const prevFlags = this.prevFlagScratch;
    for (let puIdx = 0; puIdx < numPU; puIdx++) {
      const prevFlag = cabac.decodeBin(CTX.PREV_INTRA_LUMA_PRED_FLAG);
      if (this.debug.trace) this.tr('prev_intra_luma_pred_flag', prevFlag);
      prevFlags[puIdx] = prevFlag;
    }
    const chromaIdx = this.chromaIdxScratch;
    chromaIdx.fill(4, 0, numPU);
    // modes are derived sequentially; each PU's mode is visible to later PUs' MPM candidates
    for (let puIdx = 0; puIdx < numPU; puIdx++) {
      const px = x0 + (puIdx & 1) * puSize;
      const py = y0 + (puIdx >> 1) * puSize;
      // neighbors on the min-PU grid: left (x-1,y) and above (x,y-1).
      // The above candidate is only used when it is inside the same CTB
      // (libde265/spec behavior); otherwise it defaults to DC.
      const candA = this.neighborMode(px, py, -1, 0);
      const atCtbTop = (py & ((1 << sps.log2CtbSize) - 1)) === 0;
      const candB = atCtbTop ? null : this.neighborMode(px, py, 0, -1);
      if (this.debug.trace) this.tr('cands', px + ',' + py + ' A=' + candA + ' B=' + candB);
      const cand = fillIntraPredModeCandidates(candA, candB, this.candidateModeScratch);
      let mode: number;
      if (prevFlags[puIdx]) {
        let mpm = 0;
        if (cabac.decodeBypass()) { mpm = 1; if (cabac.decodeBypass()) mpm = 2; }
        if (this.debug.trace) this.tr('mpm_idx', mpm);
        mode = intraPredModeDecode(1, mpm, 0, cand);
      } else {
        let rem = 0;
        for (let i = 0; i < 5; i++) rem = (rem << 1) | cabac.decodeBypass();
        if (this.debug.trace) this.tr('rem_intra_luma_pred_mode', rem);
        mode = intraPredModeDecode(0, 0, rem, cand);
      }
      if (this.debug.trace) this.tr('mode', mode);
      modes[puIdx] = mode;
      for (let y = 0; y < (puSize >> 2); y++) {
        const gy = (py >> 2) + y;
        if (gy >= this.gridH) break;
        for (let x = 0; x < (puSize >> 2); x++) {
          const gx = (px >> 2) + x;
          if (gx >= this.gridW) break;
          this.intraPredMode[gy * this.gridW + gx] = mode;
        }
      }
    }
    if (this.chromaFormat === CHROMA_444) {
      for (let puIdx = 0; puIdx < numPU; puIdx++) {
        const bChroma = cabac.decodeBin(CTX.INTRA_CHROMA_PRED_MODE);
        chromaIdx[puIdx] = bChroma ? cabac.readFL(2) : 4;
        if (this.debug.trace) this.tr('intra_chroma_pred_mode', chromaIdx[puIdx]!);
      }
    } else if (this.chromaFormat !== CHROMA_MONO) {
      const bChroma = cabac.decodeBin(CTX.INTRA_CHROMA_PRED_MODE);
      chromaIdx.fill(bChroma ? cabac.readFL(2) : 4, 0, numPU);
      if (this.debug.trace) this.tr('intra_chroma_pred_mode', chromaIdx[0]!);
    }

    // store chroma-derived modes in 4x4 grid
    for (let puIdx = 0; puIdx < numPU; puIdx++) {
      const px = x0 + (puIdx & 1) * puSize;
      const py = y0 + (puIdx >> 1) * puSize;
      // 4:2:0/4:2:2 signal one chroma mode for the whole CU and derive it
      // from the top-left luma PU. 4:4:4 signals/derives one per PU.
      const lumaForChroma = this.chromaFormat === CHROMA_444 ? modes[puIdx]! : modes[0]!;
      let mc = chromaPredMode(lumaForChroma, chromaIdx[puIdx]!);
      if (this.chromaFormat === CHROMA_422) mc = CHROMA_422_MODE[mc]!;
      for (let y = 0; y < (puSize >> 2); y++) {
        const gy = (py >> 2) + y;
        if (gy >= this.gridH) break;
        for (let x = 0; x < (puSize >> 2); x++) {
          const gx = (px >> 2) + x;
          if (gx >= this.gridW) break;
          const gi = gy * this.gridW + gx;
          this.intraPredModeC[gi] = mc;
          this.intraChromaDerived[gi] = chromaIdx[puIdx] === 4 ? 1 : 0;
        }
      }
    }

    if (this.debug.trace) {
      this.tr('CU', x0 + ',' + y0 + ' log' + log2Cb + ' modes' + modes.subarray(0, numPU).join('/'));
    }
    const cuId = this.nextCuId++;
    this.markGrid(this.cuIdMap, x0, y0, nS, cuId);
    this.markGrid(this.tqtBypassMap, x0, y0, nS, this.curTqBypass ? 1 : 0);

    const intraSplitFlag = partNxN ? 1 : 0;
    const maxTrafoDepth = sps.maxTransformHierarchyDepthIntra + intraSplitFlag;
    const log2Start = log2Cb;
    this.transformTree(x0, y0, x0, y0, log2Start, 0, 0, maxTrafoDepth, intraSplitFlag, x0, y0, 1, 1);
  }

  private markGrid(map: { [i: number]: number; length: number } & (Int32Array | Uint8Array | Int8Array), x0: number, y0: number, size: number, val: number) {
    const gs = size >> 2;
    for (let y = 0; y < gs && ((y0 >> 2) + y) < this.gridH; y++) {
      for (let x = 0; x < gs && ((x0 >> 2) + x) < this.gridW; x++) {
        map[((y0 >> 2) + y) * this.gridW + (x0 >> 2) + x] = val;
      }
    }
  }

  private readPcmSamples(x0: number, y0: number, log2Cb: number): void {
    const contexts = this.cabac.saveContexts();
    const r = new BitReader(this.rbsp, this.cabac.rawBytePosition());
    const n = 1 << log2Cb;
    const readPlane = (cIdx: number, hS: number, vS: number, pcmDepth: number, outputDepth: number) => {
      const plane = this.frame.planes[cIdx]!;
      const x = x0 >> hS, y = y0 >> vS;
      const w = n >> hS, h = n >> vS;
      const shift = Math.max(0, outputDepth - pcmDepth);
      for (let py = 0; py < h; py++) {
        const row = (y + py) * plane.stride + x;
        for (let px = 0; px < w; px++) plane.data[row + px] = r.u(pcmDepth) << shift;
      }
    };
    readPlane(0, 0, 0, this.sps.pcmSampleBitDepthLuma, this.sps.bitDepthLuma);
    if (this.chromaFormat !== CHROMA_MONO) {
      const hS = this.chromaFormat === CHROMA_444 ? 0 : 1;
      const vS = this.chromaFormat === CHROMA_420 ? 1 : 0;
      readPlane(1, hS, vS, this.sps.pcmSampleBitDepthChroma, this.sps.bitDepthChroma);
      readPlane(2, hS, vS, this.sps.pcmSampleBitDepthChroma, this.sps.bitDepthChroma);
    }
    r.byteAlign();
    this.cabac = new Cabac(this.rbsp, r.pos >> 3, this.debug.cabac);
    this.cabac.loadContexts(contexts);
  }

  private neighborMode(x: number, y: number, dx: number, dy: number): number | null {
    const nx = x + dx, ny = y + dy;
    const sps = this.sps;
    if (nx < 0 || ny < 0 || nx >= sps.width || ny >= sps.height) return null;
    const gi = (ny >> 2) * this.gridW + (nx >> 2);
    if (!this.sameSliceAt(nx, ny)) return null;
    const m = this.intraPredMode[gi];
    if (m < 0) return null;
    const currZ = this.minTbAddrZs[(y >> 2) * this.tbsW + (x >> 2)]!;
    return this.minTbAddrZs[gi]! <= currZ ? m : null;
  }

  private transformTree(x0: number, y0: number, xBase: number, yBase: number,
    log2TrafoSize: number, trafoDepth: number, blkIdx: number,
    maxTrafoDepth: number, intraSplitFlag: number, cuX: number, cuY: number,
    parentCbfCb: number, parentCbfCr: number) {
    const sps = this.sps;
    const cabac = this.cabac;

    // split_transform_flag: read or infer
    let split: number;
    if (log2TrafoSize <= sps.log2MaxTbSize && log2TrafoSize > sps.log2MinTbSize &&
      trafoDepth < maxTrafoDepth && !(intraSplitFlag && trafoDepth === 0)) {
      const ctxInc = 5 - log2TrafoSize;
      split = cabac.decodeBin(CTX.SPLIT_TRANSFORM_FLAG + ctxInc);
      if (this.debug.trace) this.tr('split_transform_flag', split);
    } else {
      split = (log2TrafoSize > sps.log2MaxTbSize || (intraSplitFlag && trafoDepth === 0)) ? 1 : 0;
    }

    // chroma cbf: read at every node with log2TrafoSize>2 (even when split!)
    let cbfCb = -1;
    let cbfCr = -1;
    if ((log2TrafoSize > 2 && this.chromaFormat !== CHROMA_MONO) || this.chromaFormat === CHROMA_444) {
      if (parentCbfCb) {
        cbfCb = cabac.decodeBin(CTX.CBF_CHROMA + trafoDepth);
        if (this.debug.trace) this.tr('cbf_cb', cbfCb);
        if (this.chromaFormat === CHROMA_422 && (!split || log2TrafoSize === 3)) {
          cbfCb |= cabac.decodeBin(CTX.CBF_CHROMA + trafoDepth) << 1;
          if (this.debug.trace) this.tr('cbf_cb_422', cbfCb >> 1);
        }
      } else cbfCb = 0;
      if (parentCbfCr) {
        cbfCr = cabac.decodeBin(CTX.CBF_CHROMA + trafoDepth);
        if (this.debug.trace) this.tr('cbf_cr', cbfCr);
        if (this.chromaFormat === CHROMA_422 && (!split || log2TrafoSize === 3)) {
          cbfCr |= cabac.decodeBin(CTX.CBF_CHROMA + trafoDepth) << 1;
          if (this.debug.trace) this.tr('cbf_cr_422', cbfCr >> 1);
        }
      } else cbfCr = 0;
    }
    if (cbfCb < 0) {
      cbfCb = trafoDepth > 0 && log2TrafoSize === 2 ? parentCbfCb : 0;
    }
    if (cbfCr < 0) {
      cbfCr = trafoDepth > 0 && log2TrafoSize === 2 ? parentCbfCr : 0;
    }

    if (split) {
      const half = 1 << (log2TrafoSize - 1);
      this.transformTree(x0, y0, x0, y0, log2TrafoSize - 1, trafoDepth + 1, 0, maxTrafoDepth, intraSplitFlag, cuX, cuY, cbfCb, cbfCr);
      this.transformTree(x0 + half, y0, x0, y0, log2TrafoSize - 1, trafoDepth + 1, 1, maxTrafoDepth, intraSplitFlag, cuX, cuY, cbfCb, cbfCr);
      this.transformTree(x0, y0 + half, x0, y0, log2TrafoSize - 1, trafoDepth + 1, 2, maxTrafoDepth, intraSplitFlag, cuX, cuY, cbfCb, cbfCr);
      this.transformTree(x0 + half, y0 + half, x0, y0, log2TrafoSize - 1, trafoDepth + 1, 3, maxTrafoDepth, intraSplitFlag, cuX, cuY, cbfCb, cbfCr);
      return;
    }

    const cbfLuma = cabac.decodeBin(CTX.CBF_LUMA + (trafoDepth === 0 ? 1 : 0));
    if (this.debug.trace) this.tr('cbf_luma', cbfLuma);

    const tuId = this.nextTuId++;
    this.markGrid(this.tuIdMap, x0, y0, 1 << log2TrafoSize, tuId);
    this.markGrid(this.cbfYMap, x0, y0, 1 << log2TrafoSize, cbfLuma);
    this.markGrid(this.cbfCbMap, x0, y0, 1 << log2TrafoSize, cbfCb);
    this.markGrid(this.cbfCrMap, x0, y0, 1 << log2TrafoSize, cbfCr);
    this.markGrid(this.tuSizeLog2Map, x0, y0, 1 << log2TrafoSize, log2TrafoSize);

    this.transformUnit(x0, y0, xBase, yBase, log2TrafoSize, trafoDepth, blkIdx, cbfLuma, cbfCb, cbfCr, cuX, cuY);
  }

  private transformUnit(x0: number, y0: number, xBase: number, yBase: number,
    log2TrafoSize: number, trafoDepth: number, blkIdx: number,
    cbfLuma: number, cbfCb: number, cbfCr: number, cuX: number, cuY: number) {
    const sps = this.sps;
    const nT = 1 << log2TrafoSize;

    // cu_qp_delta is read once at the first coded TU in a quantization group.
    if (this.pps.cuQpDeltaEnabled && !this.cuQpDeltaCoded && (cbfLuma || cbfCb || cbfCr)) {
      const absVal = this.decodeCuQpDeltaAbs();
      this.cuQpDelta = absVal ? (this.cabac.decodeBypass() ? -absVal : absVal) : 0;
      this.cuQpDeltaCoded = true;
      this.deriveQuantizationParameters(cuX, cuY);
    }
    if (this.cuChromaQpOffsetEnabled && !this.cuChromaQpOffsetCoded &&
      (cbfCb || cbfCr) && !this.curTqBypass) {
      const present = this.cabac.decodeBin(CTX.CU_CHROMA_QP_OFFSET_FLAG);
      let index = 0;
      if (present && this.pps.cbQpOffsetList.length > 1) {
        index = this.cabac.decodeBin(CTX.CU_CHROMA_QP_OFFSET_IDX);
      }
      if (present) {
        this.cuCbQpOffset = this.pps.cbQpOffsetList[index] ?? 0;
        this.cuCrQpOffset = this.pps.crQpOffsetList[index] ?? 0;
      } else {
        this.cuCbQpOffset = 0; this.cuCrQpOffset = 0;
      }
      this.cuChromaQpOffsetCoded = true;
    }

    // luma prediction + residual
    const modeL = this.modeAt(this.intraPredMode, x0, y0);
    this.intraPredDo(x0, y0, log2TrafoSize, 0, modeL);
    this.lumaResidual.fill(0, 0, nT * nT);
    this.lumaResidualStride = nT;
    if (this.debug.tuDebug && x0 === this.debug.tuX && y0 === this.debug.tuY) {
      const luma = this.frame.planes[0]!;
      debugWrite(`TUDBG pred Y=${luma.data[y0 * luma.stride + x0]} nT=${nT} mode=${modeL} qp=${this.curQp}\n`);
    }
    if (cbfLuma) this.residualCoding(x0, y0, log2TrafoSize, 0, modeL);
    if (this.debug.tuDebug && x0 === this.debug.tuX && y0 === this.debug.tuY) {
      const luma = this.frame.planes[0]!;
      debugWrite(`TUDBG post Y=${luma.data[y0 * luma.stride + x0]}\n`);
    }

    // chroma (prediction unconditional; residual when coded)
    if (this.chromaFormat !== CHROMA_MONO) {
      const hS = this.chromaFormat === CHROMA_444 ? 0 : 1;
      const vS = this.chromaFormat === CHROMA_444 || this.chromaFormat === CHROMA_422 ? 0 : 1;
      const gi = (y0 >> 2) * this.gridW + (x0 >> 2);
      const crossComponent = this.pps.crossComponentPredictionEnabled && !!cbfLuma && !!this.intraChromaDerived[gi];
      if (this.chromaFormat !== CHROMA_444 && log2TrafoSize === 2) {
        // Shared 4x4 chroma TB(s) for the four luma 4x4 TUs.
        if (blkIdx === 3) {
          const cx = xBase >> hS, cy = yBase >> vS;
          const modeC = this.modeAtC(cx, cy);
          const decodeComponent = (cIdx: 1 | 2, cbf: number) => {
            const resScale = crossComponent ? this.readCrossComponentScale(cIdx - 1) : 0;
            this.intraPredDo(cx, cy, 2, cIdx, modeC);
            const prediction = resScale ? this.captureBlock(cx, cy, 4, cIdx) : null;
            this.componentResidual.fill(0, 0, 16);
            if (cbf & 1) this.residualCoding(cx, cy, 2, cIdx, modeC);
            if (resScale) this.applyCrossComponent(cx, cy, 4, cIdx, resScale, prediction!);
            if (this.chromaFormat === CHROMA_422) {
              const cy2 = cy + 4;
              const modeC2 = this.modeAtC(cx, cy2);
              this.intraPredDo(cx, cy2, 2, cIdx, modeC2);
              const prediction2 = resScale ? this.captureBlock(cx, cy2, 4, cIdx) : null;
              this.componentResidual.fill(0, 0, 16);
              if (cbf & 2) this.residualCoding(cx, cy2, 2, cIdx, modeC2);
              if (resScale) this.applyCrossComponent(cx, cy2, 4, cIdx, resScale, prediction2!);
            }
          };
          decodeComponent(1, cbfCb);
          decodeComponent(2, cbfCr);
        }
      } else {
        const log2C = this.chromaFormat === CHROMA_444 ? log2TrafoSize : log2TrafoSize - 1;
        const cx = x0 >> hS, cy = y0 >> vS;
        const modeC = this.modeAtC(cx, cy);
        const decodeComponent = (cIdx: 1 | 2, cbf: number) => {
          const resScale = crossComponent ? this.readCrossComponentScale(cIdx - 1) : 0;
          this.intraPredDo(cx, cy, log2C, cIdx, modeC);
          const cSize = 1 << log2C;
          const prediction = resScale ? this.captureBlock(cx, cy, cSize, cIdx) : null;
          this.componentResidual.fill(0, 0, cSize * cSize);
          if (cbf & 1) this.residualCoding(cx, cy, log2C, cIdx, modeC);
          if (resScale) this.applyCrossComponent(cx, cy, cSize, cIdx, resScale, prediction!);
          if (this.chromaFormat === CHROMA_422) {
            const cy2 = cy + (1 << log2C);
            const modeC2 = this.modeAtC(cx, cy2);
            this.intraPredDo(cx, cy2, log2C, cIdx, modeC2);
            const prediction2 = resScale ? this.captureBlock(cx, cy2, cSize, cIdx) : null;
            this.componentResidual.fill(0, 0, cSize * cSize);
            if (cbf & 2) this.residualCoding(cx, cy2, log2C, cIdx, modeC2);
            if (resScale) this.applyCrossComponent(cx, cy2, cSize, cIdx, resScale, prediction2!);
          }
        };
        decodeComponent(1, cbfCb);
        decodeComponent(2, cbfCr);
      }
    }
    void nT;
  }

  private readCrossComponentScale(componentMinus1: number): number {
    let log2AbsPlus1 = 0;
    for (let bin = 0; bin < 4; bin++) {
      if (!this.cabac.decodeBin(CTX.LOG2_RES_SCALE_ABS_PLUS1 + 4 * componentMinus1 + bin)) break;
      log2AbsPlus1++;
    }
    if (!log2AbsPlus1) return 0;
    const sign = this.cabac.decodeBin(CTX.RES_SCALE_SIGN_FLAG + componentMinus1);
    return (1 << (log2AbsPlus1 - 1)) * (sign ? -1 : 1);
  }

  private captureBlock(x0: number, y0: number, nT: number, cIdx: number): Int32Array {
    const plane = this.frame.planes[cIdx]!;
    const out = this.crossPrediction;
    for (let y = 0; y < nT; y++) for (let x = 0; x < nT; x++) {
      out[x + y * nT] = plane.data[(y0 + y) * plane.stride + x0 + x]!;
    }
    return out;
  }

  private applyCrossComponent(x0: number, y0: number, nT: number, cIdx: number, scale: number, prediction: Int32Array): void {
    if (this.debug.disableCrossComponent) return;
    const plane = this.frame.planes[cIdx]!;
    const max = (1 << this.sps.bitDepthChroma) - 1;
    const debug = this.debug.tuDebug && x0 === this.debug.tuX &&
      y0 === this.debug.tuY && cIdx === this.debug.tuComponent;
    for (let y = 0; y < nT; y++) {
      for (let x = 0; x < nT; x++) {
        // H.265 cross-component prediction performs the depth conversion in
        // the unsigned 32-bit domain before interpreting it as signed again.
        const luma = ((this.lumaResidual[x + y * this.lumaResidualStride]! >>> 0) <<
          this.sps.bitDepthChroma) >>> this.sps.bitDepthLuma;
        const delta = (scale * luma) >> 3;
        if (debug && x === 0 && y === 0) debugWrite(`CCPDBG scale=${scale} luma=${luma} pred=${prediction[0]} residual=${this.componentResidual[0]} delta=${delta}\n`);
        const i = (y0 + y) * plane.stride + x0 + x;
        const v = prediction[x + y * nT]! + this.componentResidual[x + y * nT]! + delta;
        plane.data[i] = v < 0 ? 0 : v > max ? max : v;
      }
    }
  }

  private intraPredDo(x0: number, y0: number, log2Size: number, cIdx: number, mode: number) {
    const sps = this.sps;
    const bd = cIdx === 0 ? sps.bitDepthLuma : sps.bitDepthChroma;
    const intraCtx = this.intraCtx;
    intraCtx.bitDepth = bd;
    intraCtx.currentSliceId = this.currentSliceId;
    intraCtx.currentTileId = this.currentTileId;
    intraPredict(intraCtx, cIdx, x0, y0, mode, 1 << log2Size, false);
  }

  /** HEVC 8.6.1 QP prediction, including CTB and WPP availability boundaries. */
  private deriveQuantizationParameters(cuX: number, cuY: number): void {
    const sps = this.sps;
    const qgLog2 = sps.log2CtbSize - this.pps.diffCuQpDeltaDepth;
    const qgSize = 1 << qgLog2;
    const qgX = cuX & -qgSize;
    const qgY = cuY & -qgSize;

    if (qgX !== this.currentQgX || qgY !== this.currentQgY) {
      this.prevQp = this.curQp;
      this.currentQgX = qgX;
      this.currentQgY = qgY;
    }

    const ctbSize = 1 << sps.log2CtbSize;
    const firstInSlice = qgX === this.sliceStartX && qgY === this.sliceStartY;
    const firstInTile = qgX === this.tileStartX && qgY === this.tileStartY;
    const firstInWppRow = this.pps.entropyCodingSync && qgX === this.tileStartX && (qgY & (ctbSize - 1)) === 0;
    const base = firstInSlice || firstInTile || firstInWppRow ? this.sliceQpY : this.prevQp;
    const currentCtbX = qgX >> sps.log2CtbSize;
    const currentCtbY = qgY >> sps.log2CtbSize;
    const minCbLog2 = sps.log2MinCbSize;

    const qpAt = (x: number, y: number): number => {
      const ix = Math.min(this.widthInMinCbs - 1, x >> minCbLog2);
      const iy = Math.min(this.qpMapRows - 1, y >> minCbLog2);
      return this.qpMap[iy * this.widthInMinCbs + ix]!;
    };
    const zAvailable = (x: number, y: number): boolean => {
      if (x < 0 || y < 0) return false;
      if (!this.sameSliceAt(x, y)) return false;
      const gx = x >> 2, gy = y >> 2;
      const cgx = qgX >> 2, cgy = qgY >> 2;
      return this.minTbAddrZs[gy * this.tbsW + gx]! < this.minTbAddrZs[cgy * this.tbsW + cgx]!;
    };

    const leftInCtb = qgX > 0 && ((qgX - 1) >> sps.log2CtbSize) === currentCtbX;
    const aboveInCtb = qgY > 0 && ((qgY - 1) >> sps.log2CtbSize) === currentCtbY;
    const qpa = leftInCtb && zAvailable(qgX - 1, qgY) ? qpAt(qgX - 1, qgY) : base;
    const qpb = aboveInCtb && zAvailable(qgX, qgY - 1) ? qpAt(qgX, qgY - 1) : base;
    const qpPred = (qpa + qpb + 1) >> 1;
    const qpBd = (sps.bitDepthLuma - 8) * 6;
    const mod = 52 + qpBd;
    // cu_qp_delta syntax is unbounded; a non-positive JS modulo would produce
    // a negative QP (and undefined table lookups downstream), so pin the raw
    // value into [0, mod) before applying the bit-depth offset.
    const raw = (qpPred + this.cuQpDelta + 52 + 2 * qpBd) % mod;
    this.curQp = (raw < 0 ? raw + mod : raw) - qpBd;

    const spanCb = Math.max(1, 1 << (this.curCuLog2 - minCbLog2));
    const cbX0 = cuX >> minCbLog2, cbY0 = cuY >> minCbLog2;
    for (let y = 0; y < spanCb && cbY0 + y < this.qpMapRows; y++) {
      for (let x = 0; x < spanCb && cbX0 + x < this.widthInMinCbs; x++) {
        this.qpMap[(cbY0 + y) * this.widthInMinCbs + cbX0 + x] = this.curQp;
      }
    }
    if (this.debug.tuDebug && this.cuQpDeltaCoded) {
      debugWrite(`QPDBG cu=${cuX},${cuY} qg=${qgX},${qgY} pred=${qpPred} delta=${this.cuQpDelta} qpy=${this.curQp}\n`);
    }
  }

  private decodeCuQpDeltaAbs(): number {
    const cabac = this.cabac;
    const first = cabac.decodeBin(CTX.CU_QP_DELTA_ABS);
    if (this.debug.trace) this.tr('qpd_bin0', first);
    if (this.debug.trace) cabac.dump('qpd_after_bin0');
    if (!first) {
      if (this.debug.trace) this.tr('cu_qp_delta_abs', 0);
      return 0;
    }
    let prefix = 1;
    for (let i = 0; i < 4; i++) {
      const b = cabac.decodeBin(CTX.CU_QP_DELTA_ABS + 1);
      if (this.debug.trace) this.tr('qpd_bin' + (i + 1), b);
      if (!b) break;
      prefix++;
    }
    if (this.debug.trace) cabac.dump('qpd_after_prefix');
    const v = prefix === 5 ? cabac.readEGk(0) + 5 : prefix;
    if (this.debug.trace) this.tr('cu_qp_delta_abs', v);
    return v;
  }

  private modeAt(map: Int8Array, x: number, y: number): number {
    const gi = (y >> 2) * this.gridW + (x >> 2);
    const m = map[gi];
    return m !== undefined && m >= 0 ? m : 1;
  }
  private modeAtC(cx: number, cy: number): number {
    // chroma coords -> luma grid lookup
    const lx = this.chromaFormat === CHROMA_444 ? cx : cx * 2;
    const ly = this.chromaFormat === CHROMA_444 || this.chromaFormat === CHROMA_422 ? cy : cy * 2;
    const gi = (ly >> 2) * this.gridW + (lx >> 2);
    const m = this.intraPredModeC[gi];
    return m >= 0 ? m : 1;
  }

  // ---------------- residual coding ----------------
  private coeffBuf = new Int32Array(32 * 32);
  private posBuf = new Int32Array(32 * 32);

  private readLastPrefix(base: number, cIdx: number, log2TrafoSize: number, cMax: number): number {
    let ctxOffset: number, ctxShift: number;
    if (cIdx === 0) {
      ctxOffset = 3 * (log2TrafoSize - 2) + ((log2TrafoSize - 1) >> 2);
      ctxShift = (log2TrafoSize + 1) >> 2;
    } else {
      ctxOffset = 15;
      ctxShift = log2TrafoSize - 2;
    }
    for (let binIdx = 0; binIdx < cMax; binIdx++) {
      if (!this.cabac.decodeBin(base + ctxOffset + (binIdx >> ctxShift))) return binIdx;
    }
    return cMax;
  }

  private residualCoding(x0: number, y0: number, log2TrafoSize: number, cIdx: number, intraMode: number) {
    if (this.debug.trace) this.tr('TU', x0 + ',' + y0 + ' L' + log2TrafoSize + ' c' + cIdx);
    const sps = this.sps;
    const pps = this.pps;
    const cabac = this.cabac;
    const nT = 1 << log2TrafoSize;
    const isLuma = cIdx === 0;

    let scanIdx = 0;
    if (log2TrafoSize === 2 || (log2TrafoSize === 3 && (cIdx === 0 || this.chromaFormat === CHROMA_444))) {
      if (intraMode >= 6 && intraMode <= 14) scanIdx = 2;
      else if (intraMode >= 22 && intraMode <= 30) scanIdx = 1;
    }
    let transformSkip = 0;
    if (pps.transformSkipEnabled && !this.curTqBypass && log2TrafoSize <= pps.log2MaxTransformSkipBlockSize) {
      transformSkip = cabac.decodeBin(CTX.TRANSFORM_SKIP_FLAG + (isLuma ? 0 : 1));
    }
    const rdpcmMode = sps.implicitRdpcm && (this.curTqBypass || transformSkip)
      ? intraMode === 10 ? 1 : intraMode === 26 ? 2 : 0
      : 0;

    const cMax = (log2TrafoSize << 1) - 1;
    if (this.debug.trace) this.cabac.dump('pre_last_sig');
    const prefixX = this.readLastPrefix(CTX.LAST_SIG_X_PREFIX, cIdx, log2TrafoSize, cMax);
    const prefixY = this.readLastPrefix(CTX.LAST_SIG_Y_PREFIX, cIdx, log2TrafoSize, cMax);
    if (this.debug.trace) this.tr('last_sig_xy', prefixX + ',' + prefixY);
    let lastX: number, lastY: number;
    if (prefixX > 3) {
      const nBits = (prefixX >> 1) - 1;
      lastX = ((2 + (prefixX & 1)) << nBits) + cabac.readFL(nBits);
    } else lastX = prefixX;
    if (prefixY > 3) {
      const nBits = (prefixY >> 1) - 1;
      lastY = ((2 + (prefixY & 1)) << nBits) + cabac.readFL(nBits);
    } else lastY = prefixY;
    if (scanIdx === 2) { const t = lastX; lastX = lastY; lastY = t; }

    const sbScan = SCAN[Math.max(0, log2TrafoSize - 3)]![scanIdx]!;
    const posScan = SCAN[1]![scanIdx]!;
    const sbW = nT >> 2;

    if (this.debug.trace) {
      this.tr('mode_c', this.modeAtC(x0, y0) + ' at ' + x0 + ',' + y0);
      this.tr('tu_info', 'nT' + nT + ' scanIdx' + scanIdx + ' last' + lastX + ',' + lastY + ' mode' + intraMode);
    }
    let lastSubBlock: number, lastScanPos: number;
    {
      const sbIdx = sbScan.indexOf((lastX >> 2) + (lastY >> 2) * sbW);
      const posIdx = posScan.indexOf((lastX & 3) + (lastY & 3) * 4);
      lastSubBlock = sbIdx;
      lastScanPos = posIdx;
    }

    const csbfNeighbors = this.csbfNeighbors;
    csbfNeighbors.fill(0, 0, sbW * sbW);
    let c1 = 1;
    let lastInvGreater1Ctx = 0, lastInvGreater1Flag = 0;
    let nCoeff = 0;

    for (let i = lastSubBlock; i >= 0; i--) {
      const sbPos = sbScan[i]!;
      const sbX = sbPos % sbW, sbY = (sbPos / sbW) | 0;
      let inferSbDcSig = 0;
      let subBlockCoded: number;
      if (i < lastSubBlock && i > 0) {
        const nb = csbfNeighbors[sbX + sbY * sbW]!;
        const csbfCtx = (nb & 1) | (nb >> 1);
        subBlockCoded = cabac.decodeBin(CTX.CODED_SUB_BLOCK_FLAG + csbfCtx + (cIdx ? 2 : 0));
        inferSbDcSig = 1;
      } else {
        subBlockCoded = 1;
      }
      if (subBlockCoded) {
        if (sbX > 0) csbfNeighbors[(sbX - 1) + sbY * sbW] |= 1;
        if (sbY > 0) csbfNeighbors[sbX + (sbY - 1) * sbW] |= 2;
      }

      const coeffValue = this.subblockCoeffValue;
      const coeffScanPos = this.subblockCoeffScanPos;
      const coeffMaxBase = this.subblockCoeffMaxBase;
      let nC = 0;

      if (subBlockCoded) {
        const bx = sbX << 2, by = sbY << 2;
        const prevCsbf = csbfNeighbors[sbX + sbY * sbW]!;
        const lastCoeff = i === lastSubBlock ? lastScanPos - 1 : 15;
        if (i === lastSubBlock) {
          coeffValue[0] = 1; coeffMaxBase[0] = 1; coeffScanPos[0] = lastScanPos;
          nC = 1;
        }
        for (let n = lastCoeff; n > 0; n--) {
          const p = posScan[n]!;
          const xC = bx + (p & 3), yC = by + (p >> 2);
          let ctxInc: number;
          if (sps.transformSkipContextEnabled && (this.curTqBypass || transformSkip)) {
            ctxInc = isLuma ? 42 : 43;
          } else {
            ctxInc = sigCtxInc(xC, yC, nT, cIdx, scanIdx, prevCsbf);
          }
          if (this.debug.trace) this.tr('sig_ctx', ctxInc + ' @' + xC + ',' + yC + ' n' + n);
          const significant = cabac.decodeBin(CTX.SIGNIFICANT_COEFF_FLAG + ctxInc);
          if (this.debug.trace) this.tr('significant_coeff_flag', significant);
          if (significant) {
            coeffValue[nC] = 1; coeffMaxBase[nC] = 1; coeffScanPos[nC] = n;
            nC++;
            inferSbDcSig = 0;
          }
        }
        if (lastCoeff >= 0) {
          if (!inferSbDcSig) {
            let ctxInc: number;
            if (sps.transformSkipContextEnabled && (this.curTqBypass || transformSkip)) {
              ctxInc = isLuma ? 42 : 43;
            } else {
              ctxInc = sigCtxInc(bx, by, nT, cIdx, scanIdx, prevCsbf);
            }
            const significant = cabac.decodeBin(CTX.SIGNIFICANT_COEFF_FLAG + ctxInc);
            if (this.debug.trace) this.tr('significant_coeff_flag', significant);
            if (significant) {
              coeffValue[nC] = 1; coeffMaxBase[nC] = 1; coeffScanPos[nC] = 0;
              nC++;
            }
          } else {
            coeffValue[nC] = 1; coeffMaxBase[nC] = 1; coeffScanPos[nC] = 0;
            nC++;
          }
        }
      }

      if (nC > 0) {
        let ctxSet: number;
        if (i === 0 || cIdx > 0) ctxSet = 0;
        else ctxSet = 2;
        if (c1 === 0) ctxSet++;
        c1 = 1;

        let newLastGreater1 = -1;
        const lastGreater1Coeff = Math.min(8, nC);
        for (let c = 0; c < lastGreater1Coeff; c++) {
          let greater1Ctx: number;
          if (c === 0) {
            greater1Ctx = 1;
          } else {
            greater1Ctx = lastInvGreater1Ctx;
            if (greater1Ctx > 0) {
              if (lastInvGreater1Flag === 1) greater1Ctx = 0;
              else greater1Ctx++;
            }
          }
          const ctxIdxInc = ctxSet * 4 + Math.min(greater1Ctx, 3) + (cIdx > 0 ? 16 : 0);
          const flag = cabac.decodeBin(CTX.COEFF_ABS_LEVEL_GREATER1 + ctxIdxInc);
          if (this.debug.trace) this.tr('coeff_abs_level_greater1', flag);
          lastInvGreater1Ctx = greater1Ctx;
          lastInvGreater1Flag = flag;
          if (flag) {
            coeffValue[c] = coeffValue[c]! + 1;
            c1 = 0;
            if (newLastGreater1 === -1) newLastGreater1 = c;
          } else {
            coeffMaxBase[c] = 0;
            if (c1 > 0 && c1 < 3) c1++;
          }
        }
        if (newLastGreater1 !== -1) {
          if (this.debug.trace) cabac.dump('pre_g2');
          const g2 = cabac.decodeBin(CTX.COEFF_ABS_LEVEL_GREATER2 + ctxSet + (cIdx ? 4 : 0));
          if (this.debug.trace) this.tr('coeff_abs_level_greater2', g2);
          if (this.debug.trace) cabac.dump('post_g2');
          coeffValue[newLastGreater1]! += g2;
          coeffMaxBase[newLastGreater1] = g2;
        }

        if (this.debug.trace) cabac.dump('pre_sign');
        const signHidden = !this.curTqBypass && rdpcmMode === 0 &&
          (coeffScanPos[0]! - coeffScanPos[nC - 1]! > 3);
        const signs = this.subblockSigns;
        signs.fill(0, 0, nC);
        for (let n = 0; n < nC - 1; n++) signs[n] = cabac.decodeBypass();
        if (!pps.signDataHiding || !signHidden) signs[nC - 1] = cabac.decodeBypass();

        const sbType = (cIdx === 0 ? 2 : 0) + (transformSkip || this.curTqBypass ? 1 : 0);
        let riceParam = sps.persistentRiceInit ? this.statCoeff[sbType]! >> 2 : 0;
        let firstWithRemaining = true;
        let sumAbs = 0;
        for (let n = 0; n < nC; n++) {
          const base = coeffValue[n]!;
          let remaining = 0;
          if (coeffMaxBase[n]) {
            if (this.debug.trace) cabac.dump('pre_rem');
            remaining = this.decodeCoeffAbsLevelRemaining(riceParam);
            if (this.debug.trace) this.tr('coeff_abs_level_remaining', remaining);
            if (base + remaining > 3 * (1 << riceParam)) {
              const maxRice = sps.persistentRiceInit ? 29 : 4;
              if (riceParam < maxRice) riceParam++;
            }
            if (sps.persistentRiceInit && firstWithRemaining) {
              const statRice = this.statCoeff[sbType]! >> 2;
              if (remaining >= 3 * (1 << statRice)) this.statCoeff[sbType]++;
              else if (2 * remaining < (1 << statRice) && this.statCoeff[sbType]! > 0) this.statCoeff[sbType]--;
            }
            firstWithRemaining = false;
          }
          let curr = base + remaining;
          if (signs[n]) curr = -curr;
          if (pps.signDataHiding && signHidden) {
            sumAbs += curr;
            if (n === nC - 1 && (sumAbs & 1)) curr = -curr;
          }
          const p = posScan[coeffScanPos[n]!]!;
          const xC = (sbX << 2) + (p & 3);
          const yC = (sbY << 2) + (p >> 2);
          if (nCoeff < 32 * 32) {
            this.coeffBuf[nCoeff] = Math.max(-32768, Math.min(32767, curr));
            this.posBuf[nCoeff] = xC + yC * nT;
            nCoeff++;
          }
        }
      }
    }

    if (nCoeff === 0) return;

    const bitDepth = cIdx === 0 ? sps.bitDepthLuma : sps.bitDepthChroma;
    const qpBdOffset = (bitDepth - 8) * 6;
    let qP: number;
    if (cIdx === 0) {
      qP = this.curQp + qpBdOffset;
    } else {
      const extra = cIdx === 1
        ? pps.picCbQpOffset + this.sliceCbQp + this.cuCbQpOffset
        : pps.picCrQpOffset + this.sliceCrQp + this.cuCrQpOffset;
      const qPi = Math.min(57, Math.max(-qpBdOffset, this.curQp + extra));
      qP = (this.chromaFormat === CHROMA_420 ? CHROMA_QP[Math.max(0, qPi)]! : qPi) + qpBdOffset;
    }

    const plane = this.frame.planes[cIdx]!;
    const scaling = this.scalingFactors;
    const sizeId = nT === 4 ? 0 : nT === 8 ? 1 : nT === 16 ? 2 : 3;
    const matrixId = nT === 32 ? 0 : cIdx;
    const sf = scaling ? scaling[sizeId]![matrixId] : null;
    const rotate = sps.transformSkipRotation && nT === 4;

    if (this.curTqBypass) {
      // lossless: transquant bypass
      const out = this.transformCoeff;
      out.fill(0, 0, nT * nT);
      for (let i = 0; i < nCoeff; i++) {
        const pos = rotate ? nT * nT - 1 - this.posBuf[i]! : this.posBuf[i]!;
        out[pos] = this.coeffBuf[i]!;
      }
      this.addBypass(plane.data, plane.stride, x0, y0, out, nT, bitDepth, rdpcmMode,
        isLuma ? this.lumaResidual : this.componentResidual);
      return;
    }
    if (transformSkip) {
      const coeff = dequant(
        this.coeffBuf, this.posBuf, nCoeff, nT, qP, bitDepth, sf, rotate, this.transformCoeff,
      );
      this.debugCoefficients(x0, y0, cIdx, coeff, nT);
      addTransformSkip(plane.data, plane.stride, x0, y0, coeff, nT, bitDepth, rdpcmMode, sps.extendedPrecision,
        isLuma ? this.lumaResidual : this.componentResidual);
    } else {
      const useDst = isLuma && nT === 4;
      const coeff = dequant(
        this.coeffBuf, this.posBuf, nCoeff, nT, qP, bitDepth, sf, false, this.transformCoeff,
      );
      this.debugCoefficients(x0, y0, cIdx, coeff, nT);
      addInverseTransform(plane.data, plane.stride, x0, y0, coeff, nT, bitDepth, useDst,
        isLuma ? this.lumaResidual : this.componentResidual, this.transformIntermediate);
    }
  }

  private debugCoefficients(x0: number, y0: number, cIdx: number, coeff: Int16Array, nT: number): void {
    if (!this.debug.tuDebug || x0 !== this.debug.tuX ||
      y0 !== this.debug.tuY || cIdx !== this.debug.tuComponent) return;
    debugWrite(`COEFFDBG c=${cIdx} nT=${nT}\n`);
    for (let y = 0; y < nT; y++) {
      debugWrite(`${Array.from(coeff.subarray(y * nT, (y + 1) * nT)).join(' ')}\n`);
    }
  }

  private addBypass(data: SampleArray, stride: number, xT: number, yT: number, coeff: Int16Array, nT: number, bitDepth: number, rdpcmMode = 0, residualOut?: Int32Array) {
    const maxVal = (1 << bitDepth) - 1;
    const residual = residualOut ?? new Int32Array(nT * nT);
    for (let y = 0; y < nT; y++) {
      for (let x = 0; x < nT; x++) {
        let r = coeff[x + y * nT]!;
        if (rdpcmMode === 1 && x > 0) r += residual[x - 1 + y * nT]!;
        else if (rdpcmMode === 2 && y > 0) r += residual[x + (y - 1) * nT]!;
        residual[x + y * nT] = r;
        const v = data[(yT + y) * stride + xT + x]! + r;
        data[(yT + y) * stride + xT + x] = v < 0 ? 0 : v > maxVal ? maxVal : v;
      }
    }
  }

  private decodeCoeffAbsLevelRemaining(riceParam: number): number {
    const cabac = this.cabac;
    let prefix = 0;
    while (cabac.decodeBypass()) {
      prefix++;
      if (prefix > 18) return 0;
    }
    if (prefix <= 3) {
      return (prefix << riceParam) + cabac.readFL(riceParam);
    }
    const codeword = cabac.readFL(prefix - 3 + riceParam);
    return (((1 << (prefix - 3)) + 2) << riceParam) + codeword;
  }
}

/** significant_coeff_flag context increment (Table 9-47) */
function sigCtxInc(xC: number, yC: number, nT: number, cIdx: number, scanIdx: number, prevCsbf: number): number {
  const sbWidth = nT >> 2;
  let sigCtx: number;
  if (sbWidth === 1) {
    sigCtx = SIG_MAP_4x4[(yC << 2) + xC]!;
  } else if (xC + yC === 0) {
    sigCtx = 0;
  } else {
    const xS = xC >> 2, yS = yC >> 2;
    const xP = xC & 3, yP = yC & 3;
    switch (prevCsbf) {
      case 0: sigCtx = xP + yP >= 3 ? 0 : xP + yP > 0 ? 1 : 2; break;
      case 1: sigCtx = yP === 0 ? 2 : yP === 1 ? 1 : 0; break;
      case 2: sigCtx = xP === 0 ? 2 : xP === 1 ? 1 : 0; break;
      default: sigCtx = 2; break;
    }
    if (cIdx === 0) {
      if (xS + yS > 0) sigCtx += 3;
      if (sbWidth === 2) sigCtx += scanIdx === 0 ? 9 : 15;
      else sigCtx += 21;
    } else {
      if (sbWidth === 2) sigCtx += 9;
      else sigCtx += 12;
    }
  }
  return cIdx === 0 ? sigCtx : 27 + sigCtx;
}

const DEFAULT_SCALING_LIST_4X4 = new Uint8Array(16).fill(16);
const DEFAULT_SCALING_LIST_8X8_INTRA = new Uint8Array([
  16, 16, 16, 16, 16, 16, 16, 16,
  16, 16, 17, 16, 17, 16, 17, 18,
  17, 18, 18, 17, 18, 21, 19, 20,
  21, 20, 19, 21, 24, 22, 22, 24,
  24, 22, 22, 24, 25, 25, 27, 30,
  27, 25, 25, 29, 31, 35, 35, 31,
  29, 36, 41, 44, 41, 36, 47, 54,
  54, 47, 65, 70, 65, 88, 88, 115,
]);

/** Build intra ScalingFactor arrays (sizeId 0..3), or null when disabled. */
function buildScalingFactors(sps: Spt, pps: Pps): (Uint8Array | null)[][] | null {
  if (!sps.scalingListEnabled) return null;
  const custom = pps.scalingLists ?? sps.scalingLists;
  const out: (Uint8Array | null)[][] = [[], [], [], []];
  const sizes = [4, 8, 16, 32];
  for (let sizeId = 0; sizeId < 4; sizeId++) {
    const n = sizes[sizeId]!;
    for (let matrixId = 0; matrixId < 6; matrixId++) {
      const sf = new Uint8Array(n * n).fill(16);
      let list = custom?.lists2d[sizeId]?.[matrixId];
      let dc = custom?.dc2d[sizeId]?.[matrixId] || 16;
      if (sizeId === 3 && matrixId !== 0 && matrixId !== 3) {
        list = custom?.lists2d[1]?.[matrixId];
        dc = 16;
      }
      list ??= sizeId === 0 ? DEFAULT_SCALING_LIST_4X4 : DEFAULT_SCALING_LIST_8X8_INTRA;
      if (sizeId <= 1) {
        const scan = SCAN[sizeId + 1]![0]!;
        for (let i = 0; i < n * n; i++) sf[scan[i]!] = list[i]!;
      } else {
        const k = n / 8;
        const scan8 = SCAN[2]![0]!;
        for (let i = 0; i < 64; i++) {
          const p = scan8[i]!;
          const sx = p & 7, sy = p >> 3;
          for (let dy = 0; dy < k; dy++) for (let dx = 0; dx < k; dx++) {
            sf[(sx * k + dx) + (sy * k + dy) * n] = list[i]!;
          }
        }
        sf[0] = dc;
      }
      out[sizeId]![matrixId] = sf;
    }
  }
  return out;
}
