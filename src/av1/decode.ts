import { DecodedFrame, CHROMA_420, CHROMA_422, CHROMA_444, CHROMA_MONO } from '../frame.ts';
import { defaultCdf, defaultCoefCdf } from './cdf_tables.ts';
import { decodeCoefficients } from './coeff.ts';
import type { CoefficientResult } from './coeff.ts';
import { MsacDecoder, mutableCdf } from './msac.ts';
import {
  parseFrameHeader, parseObus, parseSequenceHeader, parseTileGroup,
  OBU_FRAME, OBU_FRAME_HEADER, OBU_SEQUENCE_HEADER, OBU_TILE_GROUP,
} from './obu.ts';
import type { Av1FrameHeader, Av1SequenceHeader } from './obu.ts';
import { reconstructAv1Frame, upscaleAv1Frame } from './reconstruct.ts';
import { applyCdef } from './cdef.ts';
import { applyDeblock } from './deblock.ts';
import { applyRestoration } from './restoration.ts';
import { applyFilmGrain } from './filmgrain.ts';
import { al_part_ctx, block_dimensions, intra_mode_context } from './tables_data.ts';
import {
  cflAllowed, maxTransform420, maxTransform422, maxTransform444, maxTransformLuma,
  partitionBlockSizes, partitionTypeCount, transformSizes,
} from './tables.ts';
import { debugEnabled } from '../debug.ts';
import { assertDimensions, resolveDecodeLimits } from '../limits.ts';
import type { DecodeOptions, ResolvedDecodeLimits } from '../limits.ts';

const PARTITION_NONE = 0, PARTITION_H = 1, PARTITION_V = 2, PARTITION_SPLIT = 3;
const PARTITION_T_TOP = 4, PARTITION_T_BOTTOM = 5, PARTITION_T_LEFT = 6, PARTITION_T_RIGHT = 7;
const PARTITION_H4 = 8, PARTITION_V4 = 9;
const DC_PRED = 0, VERT_PRED = 1, VERT_LEFT_PRED = 8, CFL_PRED = 13;
const FILTER_PRED = 13;
const FILTER_MODE_TO_Y_MODE = [DC_PRED, VERT_PRED, 2, 6, DC_PRED] as const;
const SGR_PARAMETERS = [
  [140, 3236], [112, 2158], [93, 1618], [80, 1438], [70, 1295], [58, 1177],
  [47, 1079], [37, 996], [30, 925], [25, 863], [0, 2589], [0, 1618],
  [0, 1177], [0, 925], [56, 0], [22, 0],
] as const;

export interface Av1DecodedBlock {
  x4: number; y4: number; blockSize: number; skip: boolean;
  segmentId: number; cdefIndex: number; intrabc: boolean; mvX: number; mvY: number;
  deltaLf: [number, number, number, number];
  yMode: number; yAngle: number; uvMode: number; uvAngle: number;
  cflAlpha: [number, number]; qIdx: number; tx: number; uvTx: number;
  yPalette: number[] | null; uvPalette: [number[], number[]] | null;
  yPaletteIndices: Uint8Array | null; uvPaletteIndices: Uint8Array | null;
  yCoefficients: { x4: number; y4: number; tx: number; result: CoefficientResult }[];
  uvCoefficients: { plane: number; x4: number; y4: number; tx: number; result: CoefficientResult }[];
}

export interface Av1DecodeResult {
  frame: DecodedFrame;
  sequence: Av1SequenceHeader;
  header: Av1FrameHeader;
  blocks: Av1DecodedBlock[];
  finalRange: number;
  restorationUnits: Av1RestorationUnit[];
}

export interface Av1RestorationUnit {
  plane: number;
  unitX: number; unitY: number; unitSize: number;
  x: number; y: number; width: number; height: number;
  type: number;
  filterVertical: [number, number, number];
  filterHorizontal: [number, number, number];
  sgrIndex: number;
  sgrWeights: [number, number];
}

export class Av1Decoder {
  private readonly limits: ResolvedDecodeLimits;

  constructor(options: DecodeOptions = {}) {
    this.limits = resolveDecodeLimits(options);
  }

  decode(data: Uint8Array): Av1DecodeResult {
    const obus = parseObus(data, this.limits.maxBoxes);
    const sequenceObus = obus.filter(obu => obu.type === OBU_SEQUENCE_HEADER);
    if (sequenceObus.length !== 1) throw new Error('AV1: image item must contain exactly one sequence header');
    const frameSources = obus.filter(obu => obu.type === OBU_FRAME || obu.type === OBU_FRAME_HEADER);
    if (frameSources.length !== 1) {
      throw new Error('AV1: multi-frame/spatial-layer image items are outside the static single-layer decoder scope');
    }
    const sequenceObu = sequenceObus[0];
    const frameObu = frameSources[0]!.type === OBU_FRAME ? frameSources[0] : undefined;
    const frameHeaderObu = frameSources[0]!.type === OBU_FRAME_HEADER ? frameSources[0] : undefined;
    const frameSource = frameObu ?? frameHeaderObu;
    const framePayload = frameSource?.payload;
    if (!sequenceObu || !framePayload) throw new Error('AV1: sequence or frame OBU is missing');
    const sequence = parseSequenceHeader(sequenceObu.payload);
    const header = parseFrameHeader(framePayload, sequence, frameSource!.temporalId, frameSource!.spatialId);
    assertDimensions(header.width, header.height, this.limits, 'AV1 coded frame');
    assertDimensions(header.upscaledWidth, header.height, this.limits, 'AV1 output frame');
    const tileGroupPayloads = frameObu ? [header.tileData] :
      obus.filter(obu => obu.type === OBU_TILE_GROUP).map(obu => obu.payload);
    if (!tileGroupPayloads.length) throw new Error('AV1: tile group is missing');
    const tilePayloads = tileGroupPayloads.flatMap(payload => parseTileGroup(payload, header));
    const tileCount = header.tileCols * header.tileRows;
    if (tilePayloads.length !== tileCount || tilePayloads.some((tile, index) => tile.index !== index)) {
      throw new Error('AV1: tile groups do not cover the frame in order');
    }

    const chromaFormat = sequence.monochrome ? CHROMA_MONO :
      sequence.subsamplingX ? (sequence.subsamplingY ? CHROMA_420 : CHROMA_422) : CHROMA_444;
    // Super-resolution filters the padded coded picture (to an 8-pixel
    // boundary), while its phase is still derived from the visible coded
    // width. Keep those extra reconstructed samples instead of prematurely
    // clamping the resize filter at the visible right edge.
    const storageWidth = header.width === header.upscaledWidth ? header.width :
      Math.ceil(header.width / 8) * 8;
    const frame = new DecodedFrame(storageWidth, header.height, sequence.bitDepth, chromaFormat);
    const blocks: Av1DecodedBlock[] = [];
    const restorationUnits: Av1RestorationUnit[] = [];
    const sbStep4 = sequence.sb128 ? 32 : 16;
    let finalRange = 0;
    for (const payload of tilePayloads) {
      const tile = new TileDecoder(sequence, header, frame, payload.data, {
        startX: header.tileColStarts[payload.column]! * sbStep4,
        endX: Math.min(Math.ceil(header.width / 4), header.tileColStarts[payload.column + 1]! * sbStep4),
        startY: header.tileRowStarts[payload.row]! * sbStep4,
        endY: Math.min(Math.ceil(header.height / 4), header.tileRowStarts[payload.row + 1]! * sbStep4),
      });
      const tileBlocks = tile.decode();
      reconstructAv1Frame(frame.planes, tileBlocks, sequence, header);
      blocks.push(...tileBlocks);
      restorationUnits.push(...tile.restorationUnits);
      finalRange = tile.msac.range;
    }
    if (!debugEnabled('AV1_DISABLE_DEBLOCK')) applyDeblock(frame, blocks, sequence, header);
    const deblockedFrame = header.restorationTypes.some(Boolean) ? cloneFrame(frame) : null;
    if (!debugEnabled('AV1_DISABLE_CDEF')) applyCdef(frame, blocks, sequence, header);
    const outputFrame = debugEnabled('AV1_DISABLE_SUPERRES') ? frame :
      upscaleAv1Frame(frame, header.upscaledWidth, header.width);
    if (!debugEnabled('AV1_DISABLE_RESTORATION')) {
      const upscaledDeblocked = deblockedFrame ?
        upscaleAv1Frame(deblockedFrame, header.upscaledWidth, header.width) : outputFrame;
      applyRestoration(outputFrame, restorationUnits, sequence, upscaledDeblocked);
    }
    if (!debugEnabled('AV1_DISABLE_FILM_GRAIN')) applyFilmGrain(outputFrame, header.filmGrain, sequence);
    return {
      frame: outputFrame,
      sequence,
      header,
      blocks,
      finalRange,
      restorationUnits,
    };
  }
}

function cloneFrame(source: DecodedFrame): DecodedFrame {
  const output = new DecodedFrame(
    source.width, source.height, source.bitDepth, source.chromaFormat, source.chromaBitDepth,
  );
  for (let plane = 0; plane < source.planes.length; plane++) {
    output.planes[plane]!.data.set(source.planes[plane]!.data);
  }
  return output;
}

type PalettePair = [number[], number[]];
interface TileBounds { startX: number; endX: number; startY: number; endY: number }

class TileDecoder {
  readonly frame: DecodedFrame;
  readonly msac: MsacDecoder;
  private readonly sequence: Av1SequenceHeader;
  private readonly header: Av1FrameHeader;
  private readonly cdf: any;
  private readonly coefCdf: any;
  private readonly width4: number;
  private readonly height4: number;
  private readonly bounds: TileBounds;
  private readonly ssX: number;
  private readonly ssY: number;
  private readonly maxTransformChroma: readonly number[];
  private readonly blocks: Av1DecodedBlock[] = [];
  readonly restorationUnits: Av1RestorationUnit[] = [];
  private readonly restorationReferences = Array.from({ length: 3 }, () => ({
    filterVertical: [3, -7, 15] as [number, number, number],
    filterHorizontal: [3, -7, 15] as [number, number, number],
    sgrWeights: [-32, 31] as [number, number],
  }));

  private readonly abovePartition: Uint8Array;
  private readonly leftPartition = new Uint8Array(16);
  private readonly aboveSkip: Uint8Array;
  private readonly leftSkip = new Uint8Array(32);
  private readonly aboveMode: Uint8Array;
  private readonly leftMode = new Uint8Array(32);
  private readonly aboveUvMode: Uint8Array;
  private readonly leftUvMode: Uint8Array;
  private readonly aboveTxIntra: Int8Array;
  private readonly leftTxIntra = new Int8Array(32).fill(-1);
  private readonly aboveTx: Uint8Array;
  private readonly leftTx = new Uint8Array(32);
  private readonly aboveLcoef: Uint8Array;
  private readonly leftLcoef = new Uint8Array(32).fill(0x40);
  private readonly aboveCcoef: [Uint8Array, Uint8Array];
  private readonly leftCcoef: [Uint8Array, Uint8Array];

  private readonly abovePalSizeY: Uint8Array;
  private readonly leftPalSizeY = new Uint8Array(32);
  private readonly abovePalY: number[][];
  private readonly leftPalY: number[][] = Array.from({ length: 32 }, () => []);
  private readonly abovePalSizeUv: Uint8Array;
  private readonly leftPalSizeUv = new Uint8Array(32);
  private readonly abovePalUv: PalettePair[];
  private readonly leftPalUv: PalettePair[] = Array.from({ length: 32 }, () => [[], []]);

  private readonly intrabcMvValid: Uint8Array;
  private readonly intrabcMvX: Int32Array;
  private readonly intrabcMvY: Int32Array;
  private readonly blockSizeMap: Uint8Array;
  private readonly lumaTxType: Int8Array;
  private readonly segmentMap: Uint8Array;

  private lastQIdx: number;
  private readonly lastDeltaLf = new Int8Array(4);
  private readonly cdefIndices = new Int8Array(4).fill(-1);

  constructor(
    sequence: Av1SequenceHeader, header: Av1FrameHeader, frame: DecodedFrame,
    tileData: Uint8Array, bounds: TileBounds,
  ) {
    this.sequence = sequence;
    this.header = header;
    this.frame = frame;
    this.msac = new MsacDecoder(tileData, header.disableCdfUpdate);
    this.cdf = mutableCdf(defaultCdf) as any;
    // The default table stores one initializer, but vertical and horizontal
    // MV components adapt independently in the normative decoder state.
    this.cdf.mv.comp = [this.cdf.mv.comp, structuredClone(this.cdf.mv.comp)];
    const qCategory = +(header.baseQIdx > 20) + +(header.baseQIdx > 60) + +(header.baseQIdx > 120);
    this.coefCdf = mutableCdf(defaultCoefCdf[qCategory]) as any;
    this.width4 = Math.ceil(header.width / 4);
    this.height4 = Math.ceil(header.height / 4);
    this.bounds = bounds;
    this.ssX = sequence.monochrome ? 0 : sequence.subsamplingX;
    this.ssY = sequence.monochrome ? 0 : sequence.subsamplingY;
    this.maxTransformChroma = this.ssX ? (this.ssY ? maxTransform420 : maxTransform422) : maxTransform444;
    this.abovePartition = new Uint8Array(Math.ceil(this.width4 / 2));
    this.aboveSkip = new Uint8Array(this.width4);
    this.aboveMode = new Uint8Array(this.width4);
    this.aboveUvMode = new Uint8Array(Math.ceil(this.width4 / (1 << this.ssX)));
    this.leftUvMode = new Uint8Array(32 >> this.ssY);
    this.aboveTxIntra = new Int8Array(this.width4).fill(-1);
    this.aboveTx = new Uint8Array(this.width4);
    this.aboveLcoef = new Uint8Array(this.width4).fill(0x40);
    this.aboveCcoef = [
      new Uint8Array(Math.ceil(this.width4 / (1 << this.ssX))).fill(0x40),
      new Uint8Array(Math.ceil(this.width4 / (1 << this.ssX))).fill(0x40),
    ];
    this.leftCcoef = [
      new Uint8Array(32 >> this.ssY).fill(0x40),
      new Uint8Array(32 >> this.ssY).fill(0x40),
    ];
    this.abovePalSizeY = new Uint8Array(this.width4);
    this.abovePalY = Array.from({ length: this.width4 }, () => []);
    this.abovePalSizeUv = new Uint8Array(this.width4);
    this.abovePalUv = Array.from({ length: this.width4 }, () => [[], []]);
    this.intrabcMvValid = new Uint8Array(this.width4 * this.height4);
    this.intrabcMvX = new Int32Array(this.width4 * this.height4);
    this.intrabcMvY = new Int32Array(this.width4 * this.height4);
    this.blockSizeMap = new Uint8Array(this.width4 * this.height4);
    this.lumaTxType = new Int8Array(this.width4 * this.height4);
    this.segmentMap = new Uint8Array(this.width4 * this.height4);
    this.lastQIdx = header.baseQIdx;
  }

  decode(): Av1DecodedBlock[] {
    const sbStep4 = this.sequence.sb128 ? 32 : 16;
    const rootLevel = this.sequence.sb128 ? 0 : 1;
    for (let by = this.bounds.startY; by < this.bounds.endY; by += sbStep4) {
      this.resetLeftContexts();
      for (let bx = this.bounds.startX; bx < this.bounds.endX; bx += sbStep4) {
        this.cdefIndices.fill(-1);
        this.readRestorationUnits(bx, by, sbStep4);
        this.decodePartition(rootLevel, bx, by);
      }
    }
    return this.blocks;
  }

  private decodePartition(level: number, bx: number, by: number): void {
    const halfSize4 = 16 >> level;
    const haveHorizontal = this.bounds.endX > bx + halfSize4;
    const haveVertical = this.bounds.endY > by + halfSize4;
    if (!haveHorizontal && !haveVertical) {
      this.decodePartition(level + 1, bx, by);
      return;
    }

    const bx8 = bx >> 1, by8 = (by & 31) >> 1;
    const context = ((this.abovePartition[bx8]! >> (4 - level)) & 1) +
      (((this.leftPartition[by8]! >> (4 - level)) & 1) << 1);
    const partitionCdf = this.cdf.m.partition[level][context] as number[];
    let partition: number;
    if (haveHorizontal && haveVertical) {
      partition = this.msac.symbol(partitionCdf, partitionTypeCount[level]!);
      this.decodePartitionChildren(level, partition, bx, by, halfSize4);
    } else if (haveHorizontal) {
      const split = this.msac.bool(gatherTopPartitionProbability(partitionCdf, level));
      partition = split ? PARTITION_SPLIT : PARTITION_H;
      if (split) {
        this.decodePartition(level + 1, bx, by);
        this.decodePartition(level + 1, bx + halfSize4, by);
      } else {
        this.decodeBlock(level, partitionBlockSizes[level]![PARTITION_H]![0]!, bx, by);
      }
    } else {
      const split = this.msac.bool(gatherLeftPartitionProbability(partitionCdf, level));
      partition = split ? PARTITION_SPLIT : PARTITION_V;
      if (split) {
        this.decodePartition(level + 1, bx, by);
        this.decodePartition(level + 1, bx, by + halfSize4);
      } else {
        this.decodeBlock(level, partitionBlockSizes[level]![PARTITION_V]![0]!, bx, by);
      }
    }

    if (partition !== PARTITION_SPLIT || level === 4) {
      const count = Math.min(halfSize4, this.abovePartition.length - bx8);
      this.abovePartition.fill(al_part_ctx[0]![level]![partition]!, bx8, bx8 + count);
      this.leftPartition.fill(al_part_ctx[1]![level]![partition]!, by8, Math.min(16, by8 + halfSize4));
    }
  }

  private decodePartitionChildren(level: number, partition: number, bx: number, by: number, half: number): void {
    const sizes = partitionBlockSizes[level]![partition]!;
    const block = (index: number, x: number, y: number) => this.decodeBlock(level, sizes[index]!, x, y);
    switch (partition) {
      case PARTITION_NONE: block(0, bx, by); break;
      case PARTITION_H: block(0, bx, by); block(0, bx, by + half); break;
      case PARTITION_V: block(0, bx, by); block(0, bx + half, by); break;
      case PARTITION_SPLIT:
        if (level === 4) {
          block(0, bx, by); block(0, bx + 1, by); block(0, bx, by + 1); block(0, bx + 1, by + 1);
        } else {
          this.decodePartition(level + 1, bx, by);
          this.decodePartition(level + 1, bx + half, by);
          this.decodePartition(level + 1, bx, by + half);
          this.decodePartition(level + 1, bx + half, by + half);
        }
        break;
      case PARTITION_T_TOP:
        block(0, bx, by); block(0, bx + half, by); block(1, bx, by + half); break;
      case PARTITION_T_BOTTOM:
        block(0, bx, by); block(1, bx, by + half); block(1, bx + half, by + half); break;
      case PARTITION_T_LEFT:
        block(0, bx, by); block(0, bx, by + half); block(1, bx + half, by); break;
      case PARTITION_T_RIGHT:
        block(0, bx, by); block(1, bx + half, by); block(1, bx + half, by + half); break;
      case PARTITION_H4:
        for (let i = 0; i < 4; i++) if (by + i * (half >> 1) < this.bounds.endY) block(0, bx, by + i * (half >> 1));
        break;
      case PARTITION_V4:
        for (let i = 0; i < 4; i++) if (bx + i * (half >> 1) < this.bounds.endX) block(0, bx + i * (half >> 1), by);
        break;
      default: throw new Error(`AV1: invalid partition ${partition}`);
    }
  }

  private decodeBlock(level: number, blockSize: number, bx: number, by: number): void {
    if (bx >= this.bounds.endX || by >= this.bounds.endY) return;
    const dimensions = block_dimensions[blockSize]!;
    const bw4 = dimensions[0]!, bh4 = dimensions[1]!;
    const w4 = Math.min(bw4, this.bounds.endX - bx), h4 = Math.min(bh4, this.bounds.endY - by);
    const byLocal = by & 31;
    const haveTop = by > this.bounds.startY, haveLeft = bx > this.bounds.startX;
    let segmentId = 0;
    if (this.header.segmentationEnabled && this.header.segmentationPreskip) {
      segmentId = this.readSegmentId(bx, by, haveTop, haveLeft, false);
    }
    let segment = this.header.segments[segmentId]!;
    const skipContext = this.aboveSkip[bx]! + this.leftSkip[byLocal]!;
    const skip = segment.skip || !!this.msac.boolAdapt(this.cdf.m.skip[skipContext]);

    if (this.header.segmentationEnabled && !this.header.segmentationPreskip) {
      segmentId = this.readSegmentId(bx, by, haveTop, haveLeft, skip);
      segment = this.header.segments[segmentId]!;
    }

    let cdefIndex = -1;
    if (!skip) {
      const index = this.sequence.sb128 ? ((bx & 16) >> 4) + ((by & 16) >> 3) : 0;
      if (this.cdefIndices[index] < 0) {
        const value = this.msac.bools(this.header.cdefBits);
        this.cdefIndices[index] = value;
        if (bw4 > 16) this.cdefIndices[index + 1] = value;
        if (bh4 > 16) this.cdefIndices[index + 2] = value;
        if (bw4 === 32 && bh4 === 32) this.cdefIndices[index + 3] = value;
      }
      cdefIndex = this.cdefIndices[index]!;
    }
    if (!((bx | by) & (31 >> +!this.sequence.sb128)) && this.header.deltaQPresent &&
      (blockSize !== (this.sequence.sb128 ? 0 : 3) || !skip)) {
      let delta = this.msac.symbol(this.cdf.m.delta_q, 3);
      if (delta === 3) {
        const bits = 1 + this.msac.bools(3);
        delta = this.msac.bools(bits) + 1 + (1 << bits);
      }
      if (delta && this.msac.boolEqui()) delta = -delta;
      this.lastQIdx = clamp(this.lastQIdx + (delta << this.header.deltaQResLog2), 1, 255);
      if (this.header.deltaLfPresent) {
        const count = this.header.deltaLfMulti ? (this.sequence.monochrome ? 2 : 4) : 1;
        for (let index = 0; index < count; index++) {
          let deltaLf = this.msac.symbol(this.cdf.m.delta_lf[index + +this.header.deltaLfMulti], 3);
          if (deltaLf === 3) {
            const bits = 1 + this.msac.bools(3);
            deltaLf = this.msac.bools(bits) + 1 + (1 << bits);
          }
          if (deltaLf && this.msac.boolEqui()) deltaLf = -deltaLf;
          this.lastDeltaLf[index] = clamp(
            this.lastDeltaLf[index]! + (deltaLf << this.header.deltaLfResLog2), -63, 63,
          );
        }
      }
    }
    const blockQIdx = clamp(this.lastQIdx + segment.deltaQ, 0, 255);

    const hasChroma = !this.sequence.monochrome &&
      (bw4 > this.ssX || !!(bx & this.ssX)) && (bh4 > this.ssY || !!(by & this.ssY));
    const intrabc = this.header.allowIntrabc && !!this.msac.boolAdapt(this.cdf.m.intrabc);
    if (intrabc) {
      this.decodeIntrabcBlock(blockSize, bx, by, bw4, bh4, w4, h4,
        skip, hasChroma, segmentId, blockQIdx, cdefIndex);
      return;
    }

    const aboveMode = this.aboveMode[bx]!, leftMode = this.leftMode[byLocal]!;
    const yModeContextA = intra_mode_context[aboveMode]!, yModeContextL = intra_mode_context[leftMode]!;
    let yMode = this.msac.symbol(this.cdf.kfym[yModeContextA][yModeContextL], 12);
    let yAngle = 0;
    if (dimensions[2]! + dimensions[3]! >= 2 && yMode >= VERT_PRED && yMode <= VERT_LEFT_PRED) {
      yAngle = this.msac.symbol(this.cdf.m.angle_delta[yMode - VERT_PRED], 6) - 3;
    }

    let uvMode = 0, uvAngle = 0;
    const cflAlpha: [number, number] = [0, 0];
    if (hasChroma) {
      const cbw4 = (bw4 + this.ssX) >> this.ssX;
      const cbh4 = (bh4 + this.ssY) >> this.ssY;
      const cfl = this.header.segmentLossless[segmentId] ? cbw4 === 1 && cbh4 === 1 : cflAllowed.has(blockSize);
      uvMode = this.msac.symbol(this.cdf.m.uv_mode[+cfl][yMode], 12 + +cfl);
      if (uvMode === CFL_PRED) {
        const sign = this.msac.symbol(this.cdf.m.cfl_sign, 7) + 1;
        const signU = Math.floor(sign * 0x56 / 256), signV = sign - signU * 3;
        if (signU) {
          const context = +(signU === 2) * 3 + signV;
          cflAlpha[0] = this.msac.symbol(this.cdf.m.cfl_alpha[context], 15) + 1;
          if (signU === 1) cflAlpha[0] = -cflAlpha[0];
        }
        if (signV) {
          const context = +(signV === 2) * 3 + signU;
          cflAlpha[1] = this.msac.symbol(this.cdf.m.cfl_alpha[context], 15) + 1;
          if (signV === 1) cflAlpha[1] = -cflAlpha[1];
        }
      } else if (dimensions[2]! + dimensions[3]! >= 2 && uvMode >= VERT_PRED && uvMode <= VERT_LEFT_PRED) {
        uvAngle = this.msac.symbol(this.cdf.m.angle_delta[uvMode - VERT_PRED], 6) - 3;
      }
      void cbw4; void cbh4;
    }

    let yPalette: number[] | null = null;
    let uvPalette: PalettePair | null = null;
    let yPaletteIndices: Uint8Array | null = null;
    let uvPaletteIndices: Uint8Array | null = null;
    if (this.header.allowScreenContentTools && Math.max(bw4, bh4) <= 16 && bw4 + bh4 >= 4) {
      const sizeContext = dimensions[2]! + dimensions[3]! - 2;
      if (yMode === DC_PRED) {
        const paletteContext = +(this.abovePalSizeY[bx]! > 0) + +(this.leftPalSizeY[byLocal]! > 0);
        const usePalette = this.msac.boolAdapt(this.cdf.m.pal_y[sizeContext][paletteContext]);
        if (usePalette) {
          yPalette = this.readPalettePlane(0, sizeContext, bx, byLocal);
        }
      }
      if (hasChroma && uvMode === DC_PRED) {
        const usePalette = this.msac.boolAdapt(this.cdf.m.pal_uv[+(yPalette !== null)]);
        if (usePalette) {
          uvPalette = this.readUvPalette(sizeContext, bx, byLocal);
        }
      }
    }
    if (yMode === DC_PRED && !yPalette && Math.max(dimensions[2]!, dimensions[3]!) <= 3 &&
        this.sequence.filterIntra && this.msac.boolAdapt(this.cdf.m.use_filter_intra[blockSize])) {
      yMode = FILTER_PRED;
      yAngle = this.msac.symbol(this.cdf.m.filter_intra[0], 4);
    }
    if (yPalette) yPaletteIndices = this.readPaletteIndices(yPalette.length, 0, w4, h4, bw4, bh4);
    if (uvPalette) {
      // Palette index maps follow both planes' palette color syntax.
      uvPaletteIndices = this.readPaletteIndices(uvPalette[0].length, 1,
        (w4 + this.ssX) >> this.ssX, (h4 + this.ssY) >> this.ssY,
        (bw4 + this.ssX) >> this.ssX, (bh4 + this.ssY) >> this.ssY);
    }

    let tx = maxTransformLuma[blockSize]!;
    let txInfo = transformSizes[tx]!;
    let uvTx = this.maxTransformChroma[blockSize]!;
    if (this.header.segmentLossless[segmentId]) {
      tx = 0;
      uvTx = 0;
      txInfo = transformSizes[0]!;
    } else if (this.header.txModeSwitchable && txInfo.max > 0) {
      const txContext = +(this.leftTxIntra[byLocal]! >= txInfo.logH) +
        +(this.aboveTxIntra[bx]! >= txInfo.logW);
      let depth = this.msac.symbol(this.cdf.m.txsz[txInfo.max - 1][txContext], Math.min(txInfo.max, 2));
      while (depth-- > 0) { tx = txInfo.sub; txInfo = transformSizes[tx]!; }
    }

    const block: Av1DecodedBlock = {
      x4: bx, y4: by, blockSize, skip, segmentId, cdefIndex,
      intrabc: false, mvX: 0, mvY: 0,
      deltaLf: [this.lastDeltaLf[0]!, this.lastDeltaLf[1]!, this.lastDeltaLf[2]!, this.lastDeltaLf[3]!],
      yMode, yAngle, uvMode, uvAngle, cflAlpha,
      qIdx: blockQIdx, tx, uvTx, yPalette, uvPalette, yPaletteIndices, uvPaletteIndices,
      yCoefficients: [], uvCoefficients: [],
    };
    if (skip) {
      fillSpan(this.aboveLcoef, bx, Math.min(bw4, this.bounds.endX - bx), 0x40);
      fillSpan(this.leftLcoef, byLocal, Math.min(bh4, 32 - byLocal), 0x40);
      if (hasChroma) {
        const cbx = bx >> this.ssX, cby = byLocal >> this.ssY;
        for (let plane = 0; plane < 2; plane++) {
          fillSpan(this.aboveCcoef[plane], cbx, (bw4 + this.ssX) >> this.ssX, 0x40);
          fillSpan(this.leftCcoef[plane], cby, (bh4 + this.ssY) >> this.ssY, 0x40);
        }
      }
    } else {
      const uvInfo = transformSizes[uvTx]!;
      const cw4 = (w4 + this.ssX) >> this.ssX, ch4 = (h4 + this.ssY) >> this.ssY;
      const cbx = bx >> this.ssX, cby = byLocal >> this.ssY;
      for (let initialY = 0; initialY < h4; initialY += 16) {
        const endY = Math.min(h4, initialY + 16);
        for (let initialX = 0; initialX < w4; initialX += 16) {
          const endX = Math.min(w4, initialX + 16);
          for (let y = initialY; y < endY; y += txInfo.h4) {
            for (let x = initialX; x < endX; x += txInfo.w4) {
              const result = decodeCoefficients({
                msac: this.msac, modeCdf: this.cdf.m, coefCdf: this.coefCdf,
                tx, blockSize, plane: 0, intra: true,
                yMode: yMode === FILTER_PRED ? FILTER_MODE_TO_Y_MODE[yAngle]! : yMode, uvMode,
                reducedTransformSet: this.header.reducedTransformSet, qIdx: blockQIdx,
                lossless: this.header.segmentLossless[segmentId],
                subsamplingX: this.ssX, subsamplingY: this.ssY,
                above: this.aboveLcoef.subarray(bx + x), left: this.leftLcoef.subarray(byLocal + y),
              });
              block.yCoefficients.push({ x4: bx + x, y4: by + y, tx, result });
              fillSpan(this.aboveLcoef, bx + x,
                Math.min(txInfo.w4, this.bounds.endX - bx - x), result.context);
              fillSpan(this.leftLcoef, byLocal + y,
                Math.min(txInfo.h4, 32 - byLocal - y), result.context);
            }
          }

          if (!hasChroma) continue;
          const chromaStartY = initialY >> this.ssY;
          const chromaEndY = Math.min(ch4, (initialY + 16) >> this.ssY);
          const chromaStartX = initialX >> this.ssX;
          const chromaEndX = Math.min(cw4, (initialX + 16) >> this.ssX);
          for (let plane = 0; plane < 2; plane++) {
            for (let y = chromaStartY; y < chromaEndY; y += uvInfo.h4) {
              for (let x = chromaStartX; x < chromaEndX; x += uvInfo.w4) {
                const result = decodeCoefficients({
                  msac: this.msac, modeCdf: this.cdf.m, coefCdf: this.coefCdf,
                  tx: uvTx, blockSize, plane: plane + 1, intra: true, yMode, uvMode,
                  reducedTransformSet: this.header.reducedTransformSet, qIdx: blockQIdx,
                  lossless: this.header.segmentLossless[segmentId],
                  subsamplingX: this.ssX, subsamplingY: this.ssY,
                  above: this.aboveCcoef[plane].subarray(cbx + x),
                  left: this.leftCcoef[plane].subarray(cby + y),
                });
                block.uvCoefficients.push({
                  plane: plane + 1, x4: cbx + x, y4: (by >> this.ssY) + y, tx: uvTx, result,
                });
                fillSpan(this.aboveCcoef[plane], cbx + x,
                  Math.min(uvInfo.w4, this.aboveCcoef[plane].length - cbx - x), result.context);
                fillSpan(this.leftCcoef[plane], cby + y,
                  Math.min(uvInfo.h4, this.leftCcoef[plane].length - cby - y), result.context);
              }
            }
          }
        }
      }
    }

    this.updateBlockContexts(block, bw4, bh4, hasChroma);
    this.blocks.push(block);
    void level;
  }

  private readRestorationUnits(bx: number, by: number, sbStep4: number): void {
    for (let plane = 0; plane < (this.sequence.monochrome ? 1 : 3); plane++) {
      const frameType = this.header.restorationTypes[plane]!;
      if (!frameType) continue;
      const ssX = plane ? this.ssX : 0, ssY = plane ? this.ssY : 0;
      const unitLog2 = this.header.restorationUnitSizeLog2[plane ? 1 : 0]!;
      const unitSize = 1 << unitLog2;
      const halfUnit = unitSize >> 1;
      const y = by * 4 >> ssY;
      const planeHeight = (this.header.height + ssY) >> ssY;
      if (y & (unitSize - 1) || y && y + halfUnit > planeHeight) continue;
      const unitY = y >> unitLog2;
      const outputWidth = (this.header.upscaledWidth + ssX) >> ssX;
      const outputHeight = planeHeight;
      const unitCountX = Math.max(1, (outputWidth + halfUnit) >> unitLog2);

      let firstUnitX: number, endUnitX: number;
      if (this.header.width !== this.header.upscaledWidth) {
        const denominator = this.header.superResDenominator;
        const rounding = unitSize * 8 - 1;
        const shift = unitLog2 + 3;
        firstUnitX = ((4 * bx * denominator >> ssX) + rounding) >> shift;
        endUnitX = ((4 * (bx + sbStep4) * denominator >> ssX) + rounding) >> shift;
      } else {
        const x = bx * 4 >> ssX;
        if (x & (unitSize - 1) || x && x + halfUnit > outputWidth) continue;
        firstUnitX = x >> unitLog2;
        endUnitX = firstUnitX + 1;
      }

      for (let unitX = firstUnitX; unitX < Math.min(endUnitX, unitCountX); unitX++) {
        this.readRestorationUnit(plane, unitX, unitY, outputWidth, outputHeight, unitSize, frameType);
      }
    }
  }

  private readRestorationUnit(
    plane: number, unitX: number, unitY: number,
    planeWidth: number, planeHeight: number, unitSize: number, frameType: number,
  ): void {
    let type: number;
    if (frameType === 1) {
      const symbol = this.msac.symbol(this.cdf.m.restore_switchable, 2);
      type = symbol + +!!symbol;
    } else {
      const enabled = this.msac.boolAdapt(
        frameType === 2 ? this.cdf.m.restore_wiener : this.cdf.m.restore_sgrproj,
      );
      type = enabled ? frameType : 0;
    }

    const reference = this.restorationReferences[plane]!;
    const filterVertical = [...reference.filterVertical] as [number, number, number];
    const filterHorizontal = [...reference.filterHorizontal] as [number, number, number];
    const sgrWeights = [...reference.sgrWeights] as [number, number];
    let sgrIndex = 0;
    if (type === 2) {
      filterVertical[0] = plane ? 0 : this.msac.subexp(filterVertical[0] + 5, 16, 1) - 5;
      filterVertical[1] = this.msac.subexp(filterVertical[1] + 23, 32, 2) - 23;
      filterVertical[2] = this.msac.subexp(filterVertical[2] + 17, 64, 3) - 17;
      filterHorizontal[0] = plane ? 0 : this.msac.subexp(filterHorizontal[0] + 5, 16, 1) - 5;
      filterHorizontal[1] = this.msac.subexp(filterHorizontal[1] + 23, 32, 2) - 23;
      filterHorizontal[2] = this.msac.subexp(filterHorizontal[2] + 17, 64, 3) - 17;
      reference.filterVertical = [...filterVertical];
      reference.filterHorizontal = [...filterHorizontal];
    } else if (type === 3) {
      sgrIndex = this.msac.bools(4);
      const parameters = SGR_PARAMETERS[sgrIndex]!;
      sgrWeights[0] = parameters[0] ? this.msac.subexp(sgrWeights[0] + 96, 128, 4) - 96 : 0;
      sgrWeights[1] = parameters[1] ? this.msac.subexp(sgrWeights[1] + 32, 128, 4) - 32 : 95;
      reference.sgrWeights = [...sgrWeights];
    }

    const x = unitX * unitSize, y = unitY * unitSize;
    this.restorationUnits.push({
      plane, unitX, unitY, unitSize, x, y,
      width: Math.max(0, Math.min(unitSize, planeWidth - x)),
      height: Math.max(0, Math.min(unitSize, planeHeight - y)),
      type, filterVertical, filterHorizontal, sgrIndex, sgrWeights,
    });
  }

  private readSegmentId(
    bx: number, by: number, haveTop: boolean, haveLeft: boolean, skip: boolean,
  ): number {
    const index = by * this.width4 + bx;
    let context = 0, predicted = 0;
    if (haveTop && haveLeft) {
      const left = this.segmentMap[index - 1]!;
      const above = this.segmentMap[index - this.width4]!;
      const aboveLeft = this.segmentMap[index - this.width4 - 1]!;
      context = left === above && aboveLeft === left ? 2 :
        left === above || aboveLeft === left || above === aboveLeft ? 1 : 0;
      predicted = above === aboveLeft ? above : left;
    } else if (haveLeft) {
      predicted = this.segmentMap[index - 1]!;
    } else if (haveTop) {
      predicted = this.segmentMap[index - this.width4]!;
    }
    if (skip) return predicted;
    const difference = this.msac.symbol(this.cdf.m.seg_id[context], 7);
    const max = this.header.lastActiveSegmentId + 1;
    const segmentId = max > 0 ? negDeinterleave(difference, predicted, max) : 0;
    return segmentId <= this.header.lastActiveSegmentId && segmentId < 8 ? segmentId : 0;
  }

  private decodeIntrabcBlock(
    blockSize: number, bx: number, by: number, bw4: number, bh4: number,
    w4: number, h4: number, skip: boolean, hasChroma: boolean,
    segmentId: number, blockQIdx: number, cdefIndex: number,
  ): void {
    const predictor = this.findIntrabcPredictor(bx, by, bw4, bh4);
    const decodedMv = this.readMvResidual(predictor.x, predictor.y);
    const mv = this.clipIntrabcMv(decodedMv.x, decodedMv.y, bx, by, bw4, bh4, hasChroma);
    const { maxTx, uvTx, leaves } = this.readVarTxTree(
      blockSize, bx, by, bw4, bh4, w4, h4, skip, this.header.segmentLossless[segmentId]!,
    );

    const block: Av1DecodedBlock = {
      x4: bx, y4: by, blockSize, skip, segmentId, cdefIndex,
      intrabc: true, mvX: mv.x, mvY: mv.y,
      deltaLf: [this.lastDeltaLf[0]!, this.lastDeltaLf[1]!, this.lastDeltaLf[2]!, this.lastDeltaLf[3]!],
      yMode: DC_PRED, yAngle: 0, uvMode: DC_PRED, uvAngle: 0, cflAlpha: [0, 0],
      qIdx: blockQIdx, tx: maxTx, uvTx,
      yPalette: null, uvPalette: null, yPaletteIndices: null, uvPaletteIndices: null,
      yCoefficients: [], uvCoefficients: [],
    };

    const byLocal = by & 31;
    if (skip) {
      fillSpan(this.aboveLcoef, bx, Math.min(bw4, this.bounds.endX - bx), 0x40);
      fillSpan(this.leftLcoef, byLocal, Math.min(bh4, 32 - byLocal), 0x40);
      if (hasChroma) {
        const cbx = bx >> this.ssX, cby = byLocal >> this.ssY;
        for (let plane = 0; plane < 2; plane++) {
          fillSpan(this.aboveCcoef[plane], cbx, (bw4 + this.ssX) >> this.ssX, 0x40);
          fillSpan(this.leftCcoef[plane], cby, (bh4 + this.ssY) >> this.ssY, 0x40);
        }
      }
    } else {
      for (const leaf of leaves) {
        const info = transformSizes[leaf.tx]!;
        const result = decodeCoefficients({
          msac: this.msac, modeCdf: this.cdf.m, coefCdf: this.coefCdf,
          tx: leaf.tx, blockSize, plane: 0, intra: false, yMode: DC_PRED, uvMode: DC_PRED,
          reducedTransformSet: this.header.reducedTransformSet, qIdx: blockQIdx,
          lossless: this.header.segmentLossless[segmentId],
          subsamplingX: this.ssX, subsamplingY: this.ssY,
          above: this.aboveLcoef.subarray(leaf.x4), left: this.leftLcoef.subarray(leaf.y4 & 31),
        });
        block.yCoefficients.push({ x4: leaf.x4, y4: leaf.y4, tx: leaf.tx, result });
        fillSpan(this.aboveLcoef, leaf.x4, Math.min(info.w4, this.bounds.endX - leaf.x4), result.context);
        fillSpan(this.leftLcoef, leaf.y4 & 31, Math.min(info.h4, 32 - (leaf.y4 & 31)), result.context);
        this.fillLumaTxType(leaf.x4, leaf.y4, info.w4, info.h4, result.txType);
      }

      if (hasChroma) {
        const uvInfo = transformSizes[uvTx]!;
        const cw4 = (w4 + this.ssX) >> this.ssX, ch4 = (h4 + this.ssY) >> this.ssY;
        const cbx = bx >> this.ssX, cby = byLocal >> this.ssY;
        for (let plane = 0; plane < 2; plane++) {
          for (let y = 0; y < ch4; y += uvInfo.h4) {
            for (let x = 0; x < cw4; x += uvInfo.w4) {
              const lumaX = Math.min(this.bounds.endX - 1, bx + (x << this.ssX));
              const lumaY = Math.min(this.bounds.endY - 1, by + (y << this.ssY));
              const result = decodeCoefficients({
                msac: this.msac, modeCdf: this.cdf.m, coefCdf: this.coefCdf,
                tx: uvTx, blockSize, plane: plane + 1, intra: false,
                yMode: DC_PRED, uvMode: DC_PRED,
                reducedTransformSet: this.header.reducedTransformSet, qIdx: blockQIdx,
                lossless: this.header.segmentLossless[segmentId],
                subsamplingX: this.ssX, subsamplingY: this.ssY,
                lumaTxType: this.lumaTxType[lumaY * this.width4 + lumaX]!,
                above: this.aboveCcoef[plane].subarray(cbx + x),
                left: this.leftCcoef[plane].subarray(cby + y),
              });
              block.uvCoefficients.push({
                plane: plane + 1, x4: cbx + x, y4: (by >> this.ssY) + y, tx: uvTx, result,
              });
              fillSpan(this.aboveCcoef[plane], cbx + x,
                Math.min(uvInfo.w4, this.aboveCcoef[plane].length - cbx - x), result.context);
              fillSpan(this.leftCcoef[plane], cby + y,
                Math.min(uvInfo.h4, this.leftCcoef[plane].length - cby - y), result.context);
            }
          }
        }
      }
    }

    this.updateBlockContexts(block, bw4, bh4, hasChroma);
    this.blocks.push(block);
  }

  private readMvResidual(x: number, y: number): { x: number; y: number } {
    const joint = this.msac.symbol(this.cdf.mv.joint, 3);
    if (joint & 2) y += this.readMvComponent(this.cdf.mv.comp[0]);
    if (joint & 1) x += this.readMvComponent(this.cdf.mv.comp[1]);
    return { x, y };
  }

  private readMvComponent(component: any): number {
    const sign = this.msac.boolAdapt(component.sign);
    const mvClass = this.msac.symbol(component.classes, 10);
    let up: number;
    if (!mvClass) {
      up = this.msac.boolAdapt(component.class0);
    } else {
      up = 1 << mvClass;
      for (let bit = 0; bit < mvClass; bit++) {
        up |= this.msac.boolAdapt(component.classN[bit]) << bit;
      }
    }
    // IntraBC always uses integer-pixel precision, so the fractional syntax
    // is absent and the implied fractional fields are fp=3, hp=1.
    const difference = (up + 1) << 3;
    return sign ? -difference : difference;
  }

  private findIntrabcPredictor(bx: number, by: number, bw4: number, bh4: number): { x: number; y: number } {
    const candidates: { x: number; y: number; weight: number }[] = [];
    const w4 = Math.min(bw4, 16, this.bounds.endX - bx);
    const h4 = Math.min(bh4, 16, this.bounds.endY - by);
    let maxRows = 0, maxCols = 0;
    let rowsScanned = -1, colsScanned = -1;

    if (by > this.bounds.startY) {
      maxRows = Math.min((by - this.bounds.startY + 1) >> 1, 2 + +(bh4 > 1));
      rowsScanned = this.scanMvRow(candidates, by - 1, bx, bw4, w4,
        maxRows, bw4 >= 16 ? 4 : 1);
    }
    if (bx > this.bounds.startX) {
      maxCols = Math.min((bx - this.bounds.startX + 1) >> 1, 2 + +(bw4 > 1));
      colsScanned = this.scanMvColumn(candidates, by, bx - 1, bh4, h4,
        maxCols, bh4 >= 16 ? 4 : 1);
    }

    // Top-right is only eligible if that block has already been decoded. The
    // decoded-map test is the practical equivalent of the partition edge flag.
    if (by > this.bounds.startY && Math.max(bw4, bh4) <= 16 && bx + bw4 < this.bounds.endX &&
        this.intrabcMvValid[(by - 1) * this.width4 + bx + bw4]) {
      this.addMvCandidate(candidates, bx + bw4, by - 1, 4);
    }

    const nearestCount = candidates.length;
    for (let index = 0; index < nearestCount; index++) candidates[index]!.weight += 640;

    if (by > this.bounds.startY && bx > this.bounds.startX) this.addMvCandidate(candidates, bx - 1, by - 1, 4);
    const byLocal = by & 31;
    for (let distance = 2; distance <= 3; distance++) {
      if (distance > rowsScanned && distance <= maxRows) {
        const targetLocal = (byLocal - 2 * distance + 1) | 1;
        const row = by + targetLocal - byLocal;
        rowsScanned += this.scanMvRow(candidates, row, bx | 1, bw4, w4,
          1 + maxRows - distance, bw4 >= 16 ? 4 : 2);
      }
      if (distance > colsScanned && distance <= maxCols) {
        const column = (bx - distance * 2 + 1) | 1;
        const row = by + ((byLocal | 1) - byLocal);
        colsScanned += this.scanMvColumn(candidates, row, column, bh4, h4,
          1 + maxCols - distance, bh4 >= 16 ? 4 : 2);
      }
    }

    const nearest = candidates.slice(0, nearestCount).sort((a, b) => b.weight - a.weight);
    const secondary = candidates.slice(nearestCount).sort((a, b) => b.weight - a.weight);
    const candidate = nearest[0] ?? secondary[0];
    if (candidate) {
      const left = -(bx - this.bounds.startX + bw4 + 4) * 32;
      const right = (this.bounds.endX - bx + 4) * 32;
      const top = -(by - this.bounds.startY + bh4 + 4) * 32;
      const bottom = (this.bounds.endY - by + 4) * 32;
      return { x: clamp(candidate.x, left, right), y: clamp(candidate.y, top, bottom) };
    }

    const sb4 = 16 << +this.sequence.sb128;
    return by - sb4 < this.bounds.startY
      ? { x: -(512 << +this.sequence.sb128) - 2048, y: 0 }
      : { x: 0, y: -(512 << +this.sequence.sb128) };
  }

  private addMvCandidate(
    candidates: { x: number; y: number; weight: number }[], x4: number, y4: number, weight: number,
  ): boolean {
    if (x4 < this.bounds.startX || y4 < this.bounds.startY ||
        x4 >= this.bounds.endX || y4 >= this.bounds.endY) return false;
    const index = y4 * this.width4 + x4;
    if (!this.intrabcMvValid[index]) return false;
    const x = this.intrabcMvX[index]!, y = this.intrabcMvY[index]!;
    const existing = candidates.find(candidate => candidate.x === x && candidate.y === y);
    if (existing) existing.weight += weight;
    else if (candidates.length < 8) candidates.push({ x, y, weight });
    return true;
  }

  private scanMvRow(
    candidates: { x: number; y: number; weight: number }[],
    y4: number, x4: number, bw4: number, w4: number, maxRows: number, step: number,
  ): number {
    if (y4 < this.bounds.startY || x4 < this.bounds.startX ||
        y4 >= this.bounds.endY || x4 >= this.bounds.endX) return 1;
    let candidateX = x4;
    let dimensions = block_dimensions[this.blockSizeMap[y4 * this.width4 + candidateX]!]!;
    let candidateWidth = dimensions[0]!;
    let length = Math.max(step, Math.min(bw4, candidateWidth));
    if (bw4 <= candidateWidth) {
      const weight = bw4 === 1 ? 2 : Math.max(2, Math.min(2 * maxRows, dimensions[1]!));
      this.addMvCandidate(candidates, candidateX, y4, length * weight);
      return weight >> 1;
    }
    for (let x = 0; ;) {
      this.addMvCandidate(candidates, candidateX, y4, length * 2);
      x += length;
      if (x >= w4) return 1;
      candidateX = x4 + x;
      dimensions = block_dimensions[this.blockSizeMap[y4 * this.width4 + candidateX]!]!;
      candidateWidth = dimensions[0]!;
      length = Math.max(step, candidateWidth);
    }
  }

  private scanMvColumn(
    candidates: { x: number; y: number; weight: number }[],
    y4: number, x4: number, bh4: number, h4: number, maxCols: number, step: number,
  ): number {
    if (y4 < this.bounds.startY || x4 < this.bounds.startX ||
        y4 >= this.bounds.endY || x4 >= this.bounds.endX) return 1;
    let candidateY = y4;
    let dimensions = block_dimensions[this.blockSizeMap[candidateY * this.width4 + x4]!]!;
    let candidateHeight = dimensions[1]!;
    let length = Math.max(step, Math.min(bh4, candidateHeight));
    if (bh4 <= candidateHeight) {
      const weight = bh4 === 1 ? 2 : Math.max(2, Math.min(2 * maxCols, dimensions[0]!));
      this.addMvCandidate(candidates, x4, candidateY, length * weight);
      return weight >> 1;
    }
    for (let y = 0; ;) {
      this.addMvCandidate(candidates, x4, candidateY, length * 2);
      y += length;
      if (y >= h4) return 1;
      candidateY = y4 + y;
      dimensions = block_dimensions[this.blockSizeMap[candidateY * this.width4 + x4]!]!;
      candidateHeight = dimensions[1]!;
      length = Math.max(step, candidateHeight);
    }
  }

  private clipIntrabcMv(
    mvX: number, mvY: number, bx: number, by: number,
    bw4: number, bh4: number, hasChroma: boolean,
  ): { x: number; y: number } {
    let borderLeft = this.bounds.startX * 4, borderTop = this.bounds.startY * 4;
    if (hasChroma && this.ssX && bw4 < 2) borderLeft += 4;
    if (hasChroma && this.ssY && bh4 < 2) borderTop += 4;
    let srcLeft = bx * 4 + (mvX >> 3);
    let srcTop = by * 4 + (mvY >> 3);
    let srcRight = srcLeft + bw4 * 4;
    let srcBottom = srcTop + bh4 * 4;
    const borderRight = Math.ceil(this.bounds.endX / bw4) * bw4 * 4;

    if (srcLeft < borderLeft) {
      srcRight += borderLeft - srcLeft;
      srcLeft = borderLeft;
    } else if (srcRight > borderRight) {
      srcLeft -= srcRight - borderRight;
      srcRight = borderRight;
    }
    if (srcTop < borderTop) {
      srcBottom += borderTop - srcTop;
      srcTop = borderTop;
    }

    const sbShift = 6 + +this.sequence.sb128;
    const sbx = (bx >> (4 + +this.sequence.sb128)) << sbShift;
    const sby = (by >> (4 + +this.sequence.sb128)) << sbShift;
    const sbSize = 1 << sbShift;
    if (srcBottom > sby && srcRight > sbx) {
      if (srcTop - borderTop >= srcBottom - sby) {
        srcTop -= srcBottom - sby;
        srcBottom = sby;
      } else if (srcLeft - borderLeft >= srcRight - sbx) {
        srcLeft -= srcRight - sbx;
        srcRight = sbx;
      }
    }
    if (srcBottom > sby + sbSize) {
      srcTop -= srcBottom - (sby + sbSize);
      srcBottom = sby + sbSize;
    }
    if (srcBottom > sby && srcRight > sbx) {
      throw new Error('AV1: invalid IntraBC motion vector overlaps the current superblock');
    }
    return { x: (srcLeft - bx * 4) * 8, y: (srcTop - by * 4) * 8 };
  }

  private readVarTxTree(
    blockSize: number, bx: number, by: number, bw4: number, bh4: number,
    w4: number, h4: number, skip: boolean, lossless: boolean,
  ): { maxTx: number; uvTx: number; leaves: { x4: number; y4: number; tx: number }[] } {
    let maxTx = maxTransformLuma[blockSize]!;
    let uvTx = this.maxTransformChroma[blockSize]!;
    const leaves: { x4: number; y4: number; tx: number }[] = [];
    if (!skip && (lossless || maxTx === 0)) {
      maxTx = 0;
      uvTx = 0;
      if (this.header.txModeSwitchable) this.updateTxContexts(bx, by, bw4, bh4, 0, 0);
      this.appendFixedTxLeaves(leaves, bx, by, w4, h4, maxTx);
    } else if (!this.header.txModeSwitchable || skip) {
      if (this.header.txModeSwitchable) {
        const dimensions = block_dimensions[blockSize]!;
        this.updateTxContexts(bx, by, bw4, bh4, dimensions[2]!, dimensions[3]!);
      }
      if (!skip) this.appendFixedTxLeaves(leaves, bx, by, w4, h4, maxTx);
    } else {
      const info = transformSizes[maxTx]!;
      for (let y = 0; y < h4; y += info.h4) {
        for (let x = 0; x < w4; x += info.w4) {
          this.readTxTree(leaves, maxTx, 0, bx + x, by + y);
        }
      }
    }
    return { maxTx, uvTx, leaves };
  }

  private readTxTree(
    leaves: { x4: number; y4: number; tx: number }[],
    tx: number, depth: number, bx: number, by: number,
  ): void {
    const info = transformSizes[tx]!;
    let split = false;
    if (depth < 2 && tx > 0) {
      const category = 2 * (4 - info.max) - depth;
      const aboveSmall = +(this.aboveTx[bx]! < info.logW);
      const leftSmall = +(this.leftTx[by & 31]! < info.logH);
      split = !!this.msac.boolAdapt(this.cdf.m.txpart[category][aboveSmall + leftSmall]);
    }

    if (split && info.max > 1) {
      this.visitTxChildren(info, bx, by, (childX, childY) => {
        this.readTxTree(leaves, info.sub, depth + 1, childX, childY);
      });
    } else if (split) {
      this.updateTxContexts(bx, by, info.w4, info.h4, 0, 0);
      this.visitTxChildren(info, bx, by, (childX, childY) => {
        leaves.push({ x4: childX, y4: childY, tx: info.sub });
      });
    } else {
      this.updateTxContexts(bx, by, info.w4, info.h4, info.logW, info.logH);
      leaves.push({ x4: bx, y4: by, tx });
    }
  }

  private visitTxChildren(
    info: (typeof transformSizes)[number], bx: number, by: number,
    visit: (x: number, y: number) => void,
  ): void {
    const sub = transformSizes[info.sub]!;
    visit(bx, by);
    if (info.w4 >= info.h4 && bx + sub.w4 < this.bounds.endX) visit(bx + sub.w4, by);
    if (info.h4 >= info.w4 && by + sub.h4 < this.bounds.endY) {
      visit(bx, by + sub.h4);
      if (info.w4 >= info.h4 && bx + sub.w4 < this.bounds.endX) visit(bx + sub.w4, by + sub.h4);
    }
  }

  private appendFixedTxLeaves(
    leaves: { x4: number; y4: number; tx: number }[],
    bx: number, by: number, w4: number, h4: number, tx: number,
  ): void {
    const info = transformSizes[tx]!;
    for (let y = 0; y < h4; y += info.h4) {
      for (let x = 0; x < w4; x += info.w4) leaves.push({ x4: bx + x, y4: by + y, tx });
    }
  }

  private updateTxContexts(
    bx: number, by: number, width4: number, height4: number, logW: number, logH: number,
  ): void {
    fillSpan(this.aboveTx, bx, Math.min(width4, this.bounds.endX - bx), logW);
    fillSpan(this.leftTx, by & 31, Math.min(height4, 32 - (by & 31)), logH);
  }

  private fillLumaTxType(x4: number, y4: number, width4: number, height4: number, txType: number): void {
    for (let y = y4; y < Math.min(this.bounds.endY, y4 + height4); y++) {
      this.lumaTxType.fill(txType, y * this.width4 + x4,
        y * this.width4 + Math.min(this.bounds.endX, x4 + width4));
    }
  }

  private readPalettePlane(plane: number, sizeContext: number, bx: number, byLocal: number): number[] {
    const paletteSize = this.msac.symbol(this.cdf.m.pal_sz[plane][sizeContext], 6) + 2;
    const isUv = plane !== 0;
    const leftSize = isUv ? this.leftPalSizeUv[byLocal]! : this.leftPalSizeY[byLocal]!;
    const aboveSize = (byLocal & 15) ?
      (isUv ? this.abovePalSizeUv[bx]! : this.abovePalSizeY[bx]!) : 0;
    const leftPalette = isUv ? this.leftPalUv[byLocal]![0] : this.leftPalY[byLocal]!;
    const abovePalette = isUv ? this.abovePalUv[bx]![0] : this.abovePalY[bx]!;
    const cache = mergeUniqueSorted(leftPalette.slice(0, leftSize), abovePalette.slice(0, aboveSize));
    const used: number[] = [];
    for (let index = 0; index < cache.length && used.length < paletteSize; index++) {
      if (this.msac.boolEqui()) used.push(cache[index]!);
    }

    const fresh: number[] = [];
    const max = (1 << this.sequence.bitDepth) - 1;
    if (used.length < paletteSize) {
      let previous = this.msac.bools(this.sequence.bitDepth);
      fresh.push(previous);
      if (used.length + fresh.length < paletteSize) {
        let bits = this.sequence.bitDepth - 3 + this.msac.bools(2);
        while (used.length + fresh.length < paletteSize) {
          previous = Math.min(previous + this.msac.bools(bits) + +!isUv, max);
          fresh.push(previous);
          if (previous + +!isUv >= max) {
            while (used.length + fresh.length < paletteSize) fresh.push(max);
            break;
          }
          bits = Math.min(bits, 1 + Math.floor(Math.log2(max - previous - +!isUv)));
        }
      }
    }
    return mergeSorted(used, fresh).slice(0, paletteSize);
  }

  private readUvPalette(sizeContext: number, bx: number, byLocal: number): PalettePair {
    const u = this.readPalettePlane(1, sizeContext, bx, byLocal);
    const v: number[] = [];
    const max = (1 << this.sequence.bitDepth) - 1;
    if (this.msac.boolEqui()) {
      const bits = this.sequence.bitDepth - 4 + this.msac.bools(2);
      let previous = this.msac.bools(this.sequence.bitDepth);
      v.push(previous);
      for (let i = 1; i < u.length; i++) {
        let delta = this.msac.bools(bits);
        if (delta && this.msac.boolEqui()) delta = -delta;
        previous = (previous + delta) & max;
        v.push(previous);
      }
    } else {
      for (let i = 0; i < u.length; i++) v.push(this.msac.bools(this.sequence.bitDepth));
    }
    return [u, v];
  }

  private readPaletteIndices(paletteSize: number, plane: number,
    w4: number, h4: number, bw4: number, bh4: number): Uint8Array {
    const width = bw4 * 4, height = bh4 * 4;
    const actualWidth = w4 * 4, actualHeight = h4 * 4;
    const indices = new Uint8Array(width * height);
    indices[0] = this.msac.uniform(paletteSize);
    const colorMap = this.cdf.m.color_map[plane][paletteSize - 2];
    for (let diagonal = 1; diagonal < actualWidth + actualHeight - 1; diagonal++) {
      const first = Math.min(diagonal, actualWidth - 1);
      const last = Math.max(0, diagonal - actualHeight + 1);
      for (let x = first; x >= last; x--) {
        const y = diagonal - x;
        const { context, order } = paletteOrder(indices, width, x, y);
        const coded = this.msac.symbol(colorMap[context], paletteSize - 1);
        indices[y * width + x] = order[coded]!;
      }
    }
    return indices;
  }

  private updateBlockContexts(block: Av1DecodedBlock, bw4: number, bh4: number, hasChroma: boolean): void {
    const { x4: bx, y4: by, yMode, uvMode, skip, tx, yPalette, uvPalette, intrabc } = block;
    const byLocal = by & 31;
    const txInfo = transformSizes[tx]!;
    const contextYMode = yMode === FILTER_PRED ? DC_PRED : yMode;
    fillSpan(this.aboveMode, bx, bw4, contextYMode);
    fillSpan(this.leftMode, byLocal, bh4, contextYMode);
    fillSpan(this.aboveSkip, bx, bw4, +skip);
    fillSpan(this.leftSkip, byLocal, bh4, +skip);
    if (intrabc) {
      const dimensions = block_dimensions[block.blockSize]!;
      fillSpan(this.aboveTxIntra, bx, bw4, dimensions[2]!);
      fillSpan(this.leftTxIntra, byLocal, bh4, dimensions[3]!);
    } else {
      fillSpan(this.aboveTxIntra, bx, bw4, txInfo.logW);
      fillSpan(this.leftTxIntra, byLocal, bh4, txInfo.logH);
      fillSpan(this.aboveTx, bx, bw4, txInfo.logW);
      fillSpan(this.leftTx, byLocal, bh4, txInfo.logH);
    }

    for (let y = 0; y < bh4 && by + y < this.bounds.endY; y++) {
      for (let x = 0; x < bw4 && bx + x < this.bounds.endX; x++) {
        const index = (by + y) * this.width4 + bx + x;
        this.blockSizeMap[index] = block.blockSize;
        this.segmentMap[index] = block.segmentId;
        this.intrabcMvValid[index] = +intrabc;
        if (intrabc) {
          this.intrabcMvX[index] = block.mvX;
          this.intrabcMvY[index] = block.mvY;
        }
      }
    }

    const yPaletteSize = yPalette?.length ?? 0;
    for (let x = 0; x < bw4 && bx + x < this.bounds.endX; x++) {
      this.abovePalSizeY[bx + x] = yPaletteSize;
      this.abovePalY[bx + x] = yPalette?.slice() ?? [];
    }
    for (let y = 0; y < bh4 && byLocal + y < 32; y++) {
      this.leftPalSizeY[byLocal + y] = yPaletteSize;
      this.leftPalY[byLocal + y] = yPalette?.slice() ?? [];
    }

    const uvPaletteSize = uvPalette?.[0].length ?? 0;
    // Palette UV contexts deliberately use luma coordinates in AV1.
    for (let x = 0; x < bw4 && bx + x < this.bounds.endX; x++) {
      this.abovePalSizeUv[bx + x] = uvPaletteSize;
      this.abovePalUv[bx + x] = uvPalette ? [uvPalette[0].slice(), uvPalette[1].slice()] : [[], []];
    }
    for (let y = 0; y < bh4 && byLocal + y < 32; y++) {
      this.leftPalSizeUv[byLocal + y] = uvPaletteSize;
      this.leftPalUv[byLocal + y] = uvPalette ? [uvPalette[0].slice(), uvPalette[1].slice()] : [[], []];
    }

    if (hasChroma) {
      const cbx = bx >> this.ssX, cby = byLocal >> this.ssY;
      fillSpan(this.aboveUvMode, cbx, (bw4 + this.ssX) >> this.ssX, uvMode);
      fillSpan(this.leftUvMode, cby, (bh4 + this.ssY) >> this.ssY, uvMode);
    }
  }

  private resetLeftContexts(): void {
    this.leftPartition.fill(0);
    this.leftSkip.fill(0);
    this.leftMode.fill(DC_PRED);
    this.leftUvMode.fill(DC_PRED);
    this.leftTxIntra.fill(-1);
    this.leftTx.fill(0);
    this.leftLcoef.fill(0x40);
    this.leftCcoef[0].fill(0x40); this.leftCcoef[1].fill(0x40);
    this.leftPalSizeY.fill(0); this.leftPalSizeUv.fill(0);
    for (let i = 0; i < 32; i++) {
      this.leftPalY[i] = [];
      this.leftPalUv[i] = [[], []];
    }
  }

}

function gatherLeftPartitionProbability(cdf: number[], level: number): number {
  let probability = cdf[PARTITION_H - 1]! - cdf[PARTITION_H]!;
  probability += cdf[PARTITION_SPLIT - 1]! - cdf[PARTITION_T_LEFT]!;
  if (level !== 0) probability += cdf[PARTITION_H4 - 1]! - cdf[PARTITION_H4]!;
  return probability;
}

function gatherTopPartitionProbability(cdf: number[], level: number): number {
  let probability = cdf[PARTITION_V - 1]! - cdf[PARTITION_T_TOP]!;
  probability += cdf[PARTITION_T_LEFT - 1]!;
  if (level !== 0) probability += cdf[PARTITION_V4 - 1]! - cdf[PARTITION_T_RIGHT]!;
  return probability;
}

function fillSpan(array: Uint8Array | Int8Array, start: number, length: number, value: number): void {
  if (length > 0) array.fill(value, start, Math.min(array.length, start + length));
}

function negDeinterleave(difference: number, reference: number, maximum: number): number {
  if (!reference) return difference;
  if (reference >= maximum - 1) return maximum - difference - 1;
  if (2 * reference < maximum) {
    if (difference <= 2 * reference) {
      return difference & 1 ? reference + ((difference + 1) >> 1) : reference - (difference >> 1);
    }
    return difference;
  }
  if (difference <= 2 * (maximum - reference - 1)) {
    return difference & 1 ? reference + ((difference + 1) >> 1) : reference - (difference >> 1);
  }
  return maximum - difference - 1;
}

function mergeUniqueSorted(first: number[], second: number[]): number[] {
  return [...new Set([...first, ...second])].sort((a, b) => a - b);
}

function mergeSorted(first: number[], second: number[]): number[] {
  const output: number[] = [];
  let i = 0, j = 0;
  while (i < first.length || j < second.length) {
    if (j >= second.length || i < first.length && first[i]! <= second[j]!) output.push(first[i++]!);
    else output.push(second[j++]!);
  }
  return output;
}

function paletteOrder(indices: Uint8Array, stride: number, x: number, y: number): { context: number; order: number[] } {
  const order: number[] = [];
  let context = 0;
  const add = (value: number) => { if (!order.includes(value)) order.push(value); };
  if (x === 0) add(indices[(y - 1) * stride + x]!);
  else if (y === 0) add(indices[y * stride + x - 1]!);
  else {
    const left = indices[y * stride + x - 1]!;
    const top = indices[(y - 1) * stride + x]!;
    const topLeft = indices[(y - 1) * stride + x - 1]!;
    if (top === left && top === topLeft) { context = 4; add(top); }
    else if (top === left) { context = 3; add(top); add(topLeft); }
    else if (top === topLeft || left === topLeft) {
      context = 2; add(topLeft); add(top === topLeft ? left : top);
    } else {
      context = 1; add(Math.min(top, left)); add(Math.max(top, left)); add(topLeft);
    }
  }
  for (let value = 0; value < 8; value++) add(value);
  return { context, order };
}

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}
