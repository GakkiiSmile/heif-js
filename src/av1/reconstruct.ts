import { DecodedFrame } from '../frame.ts';
import type { Plane } from '../frame.ts';
import type { Av1FrameHeader, Av1SequenceHeader } from './obu.ts';
import type { Av1DecodedBlock } from './decode.ts';
import type { CoefficientResult } from './coeff.ts';
import { dequantizers } from './dequant_data.ts';
import { quantMatrixValue } from './qm_data.ts';
import { resizeFilter } from './resize_data.ts';
import { block_dimensions, dr_intra_derivative, sm_weights } from './tables_data.ts';
import { transformSizes } from './tables.ts';
import { inverseTransform2d } from './itx.ts';

const DC_PRED = 0;
const VERT_PRED = 1;
const HOR_PRED = 2;
const SMOOTH_PRED = 9;
const SMOOTH_V_PRED = 10;
const SMOOTH_H_PRED = 11;
const PAETH_PRED = 12;
const CFL_PRED = 13;
const FILTER_PRED = 13;

const FILTER_INTRA_TAPS = [
  [[-6, 10, 0, 0, 0, 12, 0], [-5, 2, 10, 0, 0, 9, 0], [-3, 1, 1, 10, 0, 7, 0], [-3, 1, 1, 2, 10, 5, 0],
   [-4, 6, 0, 0, 0, 2, 12], [-3, 2, 6, 0, 0, 2, 9], [-3, 2, 2, 6, 0, 2, 7], [-3, 1, 2, 2, 6, 3, 5]],
  [[-10, 16, 0, 0, 0, 10, 0], [-6, 0, 16, 0, 0, 6, 0], [-4, 0, 0, 16, 0, 4, 0], [-2, 0, 0, 0, 16, 2, 0],
   [-10, 16, 0, 0, 0, 0, 10], [-6, 0, 16, 0, 0, 0, 6], [-4, 0, 0, 16, 0, 0, 4], [-2, 0, 0, 0, 16, 0, 2]],
  [[-8, 8, 0, 0, 0, 16, 0], [-8, 0, 8, 0, 0, 16, 0], [-8, 0, 0, 8, 0, 16, 0], [-8, 0, 0, 0, 8, 16, 0],
   [-4, 4, 0, 0, 0, 0, 16], [-4, 0, 4, 0, 0, 0, 16], [-4, 0, 0, 4, 0, 0, 16], [-4, 0, 0, 0, 4, 0, 16]],
  [[-2, 8, 0, 0, 0, 10, 0], [-1, 3, 8, 0, 0, 6, 0], [-1, 2, 3, 8, 0, 4, 0], [0, 1, 2, 3, 8, 2, 0],
   [-1, 4, 0, 0, 0, 3, 10], [-1, 3, 4, 0, 0, 4, 6], [-1, 2, 3, 4, 0, 4, 4], [-1, 2, 2, 3, 4, 3, 3]],
  [[-12, 14, 0, 0, 0, 14, 0], [-10, 0, 14, 0, 0, 12, 0], [-9, 0, 0, 14, 0, 11, 0], [-8, 0, 0, 0, 14, 10, 0],
   [-10, 12, 0, 0, 0, 0, 14], [-9, 1, 12, 0, 0, 0, 12], [-8, 0, 0, 12, 0, 1, 11], [-7, 0, 0, 1, 12, 1, 9]],
] as const;

const MODE_ANGLES = [90, 180, 45, 135, 113, 157, 203, 67] as const;

interface ReconstructionContext {
  sequence: Av1SequenceHeader;
  header: Av1FrameHeader;
  planes: Plane[];
  masks: ReconstructionMask[];
  smoothMasks: ReconstructionMask[];
  max: number;
}

interface ReconstructionMask {
  data: Uint8Array;
  width: number;
  height: number;
  originX: number;
  originY: number;
}

export interface ReconstructionBounds {
  startX: number; endX: number; startY: number; endY: number;
}

/** Reconstruct an intra-only AV1 still picture from its already decoded block syntax. */
export function reconstructAv1Frame(
  planes: Plane[], blocks: readonly Av1DecodedBlock[],
  sequence: Av1SequenceHeader, header: Av1FrameHeader,
  bounds?: ReconstructionBounds,
): void {
  const context: ReconstructionContext = {
    sequence, header, planes,
    masks: planes.map((plane, index) => createReconstructionMask(plane, sequence, index, bounds)),
    smoothMasks: planes.map((plane, index) => createReconstructionMask(plane, sequence, index, bounds)),
    max: (1 << sequence.bitDepth) - 1,
  };
  for (const block of blocks) reconstructBlock(context, block);
}

function createReconstructionMask(
  plane: Plane, sequence: Av1SequenceHeader, planeIndex: number,
  bounds?: ReconstructionBounds,
): ReconstructionMask {
  if (!bounds) {
    return { data: new Uint8Array(plane.width * plane.height), width: plane.width, height: plane.height,
      originX: 0, originY: 0 };
  }
  const shiftX = planeIndex ? sequence.subsamplingX : 0;
  const shiftY = planeIndex ? sequence.subsamplingY : 0;
  const originX = (bounds.startX >> shiftX) * 4;
  const originY = (bounds.startY >> shiftY) * 4;
  const endX = Math.min(plane.width, ((bounds.endX + (1 << shiftX) - 1) >> shiftX) * 4);
  const endY = Math.min(plane.height, ((bounds.endY + (1 << shiftY) - 1) >> shiftY) * 4);
  const width = Math.max(0, endX - originX), height = Math.max(0, endY - originY);
  return { data: new Uint8Array(width * height), width, height, originX, originY };
}

/** Apply the normative horizontal AV1 frame super-resolution filter. */
export function upscaleAv1Frame(
  frame: DecodedFrame, outputWidth: number, visibleInputWidth = frame.width,
): DecodedFrame {
  if (outputWidth === visibleInputWidth && frame.width === visibleInputWidth) return frame;
  const output = new DecodedFrame(
    outputWidth, frame.height, frame.bitDepth, frame.chromaFormat, frame.chromaBitDepth,
  );
  const maximum = (1 << frame.bitDepth) - 1;
  for (let planeIndex = 0; planeIndex < frame.planes.length; planeIndex++) {
    const source = frame.planes[planeIndex]!;
    const destination = output.planes[planeIndex]!;
    const subsamplingX = planeIndex &&
      (frame.chromaFormat === 1 || frame.chromaFormat === 2) ? 1 : 0;
    const inputWidth = (visibleInputWidth + subsamplingX) >> subsamplingX;
    const step = Math.floor(((inputWidth << 14) + (destination.width >> 1)) / destination.width);
    const error = destination.width * step - (inputWidth << 14);
    const start = (Math.trunc((-(destination.width - inputWidth) * (1 << 13) +
      (destination.width >> 1)) / destination.width) + 128 - Math.trunc(error / 2)) & 0x3fff;
    for (let y = 0; y < source.height; y++) {
      let phase = start, sourceX = -1;
      const sourceRow = y * source.stride, destinationRow = y * destination.stride;
      for (let x = 0; x < destination.width; x++) {
        const filterOffset = (phase >> 8) * 8;
        let sum = 0;
        for (let tap = 0; tap < 8; tap++) {
          const sampleX = clip(sourceX + tap - 3, 0, source.width - 1);
          sum -= resizeFilter[filterOffset + tap]! * source.data[sourceRow + sampleX]!;
        }
        destination.data[destinationRow + x] = clip((sum + 64) >> 7, 0, maximum);
        phase += step;
        sourceX += phase >> 14;
        phase &= 0x3fff;
      }
    }
  }
  return output;
}

function reconstructBlock(context: ReconstructionContext, block: Av1DecodedBlock): void {
  if (block.intrabc) {
    reconstructIntrabcBlock(context, block);
    return;
  }
  const dimensions = block_dimensions[block.blockSize]!;
  const bw = dimensions[0]! * 4;
  const bh = dimensions[1]! * 4;
  const x = block.x4 * 4;
  const y = block.y4 * 4;
  const luma = context.planes[0]!;
  const width = Math.min(bw, luma.width - x);
  const height = Math.min(bh, luma.height - y);
  const lumaFilterType = filterType(context, 0, x, y);

  if (block.yPalette && block.yPaletteIndices) {
    palettePredict(luma, x, y, width, height, bw, block.yPalette, block.yPaletteIndices);
    if (!block.skip) {
      for (const unit of block.yCoefficients) {
        addResidual(context, 0, unit.x4 * 4, unit.y4 * 4, block.qIdx, unit.result, unit.tx);
      }
    }
    mark(context, 0, x, y, width, height);
  } else {
    const txInfo = transformSizes[block.tx]!;
    for (let oy = 0; oy < height; oy += txInfo.h4 * 4) {
      for (let ox = 0; ox < width; ox += txInfo.w4 * 4) {
        const unitWidth = Math.min(txInfo.w4 * 4, width - ox);
        const unitHeight = Math.min(txInfo.h4 * 4, height - oy);
        intraPredict(context, 0, x + ox, y + oy, unitWidth, unitHeight,
          block.yMode, block.yAngle, lumaFilterType, txInfo.w4 * 4, txInfo.h4 * 4);
        if (!block.skip) {
          const unit = block.yCoefficients.find(candidate =>
            candidate.x4 * 4 === x + ox && candidate.y4 * 4 === y + oy);
          if (unit) addResidual(context, 0, x + ox, y + oy, block.qIdx, unit.result, unit.tx);
        }
        mark(context, 0, x + ox, y + oy, unitWidth, unitHeight);
      }
    }
  }

  if (context.planes.length === 1) return;
  const ssX = context.sequence.subsamplingX, ssY = context.sequence.subsamplingY;
  const cx = (block.x4 >> ssX) * 4;
  const cy = (block.y4 >> ssY) * 4;
  const cwNominal = ((dimensions[0]! + ssX) >> ssX) * 4;
  const chNominal = ((dimensions[1]! + ssY) >> ssY) * 4;
  const chroma = context.planes[1]!;
  const cw = Math.min(cwNominal, chroma.width - cx);
  const ch = Math.min(chNominal, chroma.height - cy);
  if (cw <= 0 || ch <= 0) return;

  for (let plane = 1; plane <= 2; plane++) {
    const chromaFilterType = filterType(context, plane, cx, cy);
    if (block.uvPalette && block.uvPaletteIndices) {
      palettePredict(context.planes[plane]!, cx, cy, cw, ch, cwNominal,
        block.uvPalette[plane - 1]!, block.uvPaletteIndices);
      if (!block.skip) addChromaResiduals(context, block, plane);
      mark(context, plane, cx, cy, cw, ch);
    } else {
      const txInfo = transformSizes[block.uvTx]!;
      const units = block.uvCoefficients.filter(unit => unit.plane === plane);
      for (let oy = 0; oy < ch; oy += txInfo.h4 * 4) {
        for (let ox = 0; ox < cw; ox += txInfo.w4 * 4) {
          const unitWidth = Math.min(txInfo.w4 * 4, cw - ox);
          const unitHeight = Math.min(txInfo.h4 * 4, ch - oy);
          if (block.uvMode === CFL_PRED && block.cflAlpha[plane - 1] !== 0) {
            // CfL, like every other intra predictor, operates at transform-block
            // granularity; its luma average must not span the whole coding block.
            cflPredict(context, plane, cx + ox, cy + oy, unitWidth, unitHeight,
              block.cflAlpha[plane - 1]!, txInfo.w4 * 4, txInfo.h4 * 4);
          } else {
            intraPredict(context, plane, cx + ox, cy + oy, unitWidth, unitHeight,
              block.uvMode === CFL_PRED ? DC_PRED : block.uvMode, block.uvAngle, chromaFilterType,
              txInfo.w4 * 4, txInfo.h4 * 4);
          }
          if (!block.skip) {
            const unit = units.find(candidate =>
              candidate.x4 * 4 === cx + ox && candidate.y4 * 4 === cy + oy);
            if (unit) addResidual(context, plane, cx + ox, cy + oy, block.qIdx, unit.result, unit.tx);
          }
          mark(context, plane, cx + ox, cy + oy, unitWidth, unitHeight);
        }
      }
    }
  }
}

function addChromaResiduals(context: ReconstructionContext, block: Av1DecodedBlock, plane: number): void {
  for (const unit of block.uvCoefficients) {
    if (unit.plane === plane) {
      addResidual(context, plane, unit.x4 * 4, unit.y4 * 4,
        block.qIdx, unit.result, unit.tx);
    }
  }
}

function reconstructIntrabcBlock(context: ReconstructionContext, block: Av1DecodedBlock): void {
  const dimensions = block_dimensions[block.blockSize]!;
  const bw4 = dimensions[0]!, bh4 = dimensions[1]!;
  const x = block.x4 * 4, y = block.y4 * 4;
  const luma = context.planes[0]!;
  const width = Math.min(bw4 * 4, luma.width - x);
  const height = Math.min(bh4 * 4, luma.height - y);
  copyPlaneRegion(luma, x, y, x + (block.mvX >> 3), y + (block.mvY >> 3), width, height);
  if (!block.skip) {
    for (const unit of block.yCoefficients) {
      addResidual(context, 0, unit.x4 * 4, unit.y4 * 4, block.qIdx, unit.result, unit.tx);
    }
  }
  mark(context, 0, x, y, width, height);

  const hasChroma = context.planes.length > 1 &&
    (bw4 > context.sequence.subsamplingX || !!(block.x4 & context.sequence.subsamplingX)) &&
    (bh4 > context.sequence.subsamplingY || !!(block.y4 & context.sequence.subsamplingY));
  if (!hasChroma) return;
  const ssX = context.sequence.subsamplingX, ssY = context.sequence.subsamplingY;
  const cx = (block.x4 >> ssX) * 4, cy = (block.y4 >> ssY) * 4;
  for (let planeIndex = 1; planeIndex <= 2; planeIndex++) {
    const plane = context.planes[planeIndex]!;
    const widthC = Math.min(((bw4 + ssX) >> ssX) * 4, plane.width - cx);
    const heightC = Math.min(((bh4 + ssY) >> ssY) * 4, plane.height - cy);
    copyPlaneRegion(plane, cx, cy,
      cx + (block.mvX >> (3 + ssX)), cy + (block.mvY >> (3 + ssY)), widthC, heightC);
    if (!block.skip) addChromaResiduals(context, block, planeIndex);
    mark(context, planeIndex, cx, cy, widthC, heightC);
  }
}

function copyPlaneRegion(
  plane: Plane, dstX: number, dstY: number, srcX: number, srcY: number,
  width: number, height: number,
): void {
  for (let y = 0; y < height; y++) {
    const sourceY = clip(srcY + y, 0, plane.height - 1);
    const destination = (dstY + y) * plane.stride + dstX;
    for (let x = 0; x < width; x++) {
      const sourceX = clip(srcX + x, 0, plane.width - 1);
      plane.data[destination + x] = plane.data[sourceY * plane.stride + sourceX]!;
    }
  }
}

function palettePredict(
  plane: Plane, x0: number, y0: number, width: number, height: number,
  indexStride: number, palette: readonly number[], indices: Uint8Array,
): void {
  for (let y = 0; y < height; y++) {
    const dst = (y0 + y) * plane.stride + x0;
    const src = y * indexStride;
    for (let x = 0; x < width; x++) plane.data[dst + x] = palette[indices[src + x]!]!;
  }
}

function intraPredict(
  context: ReconstructionContext, planeIndex: number,
  x0: number, y0: number, width: number, height: number,
  mode: number, angleDelta: number, edgeFilterType = 0,
  predictionWidth = width, predictionHeight = height,
): void {
  const plane = context.planes[planeIndex]!;
  const { top, left, topLeft, haveTop, haveLeft } = collectEdges(
    context, planeIndex, x0, y0, predictionWidth, predictionHeight,
  );
  const dst = plane.data;
  const stride = plane.stride;
  markSmooth(context, planeIndex, x0, y0, width, height,
    mode === SMOOTH_PRED || mode === SMOOTH_V_PRED || mode === SMOOTH_H_PRED);

  if (mode === FILTER_PRED) {
    filterIntraPredict(plane, x0, y0, width, height, predictionWidth, predictionHeight,
      top, left, topLeft, angleDelta, context.max);
    return;
  }

  if (mode === DC_PRED) {
    let sum = 0, count = 0;
    if (haveTop) {
      for (let x = 0; x < predictionWidth; x++) sum += top[x]!;
      count += predictionWidth;
    }
    if (haveLeft) {
      for (let y = 0; y < predictionHeight; y++) sum += left[y]!;
      count += predictionHeight;
    }
    const dc = count ? Math.floor((sum + (count >> 1)) / count) : (context.max + 1) >> 1;
    fillBlock(plane, x0, y0, width, height, dc);
    return;
  }
  if (mode === VERT_PRED && angleDelta === 0) {
    for (let y = 0; y < height; y++) {
      const row = (y0 + y) * stride + x0;
      for (let x = 0; x < width; x++) dst[row + x] = top[x]!;
    }
    return;
  }
  if (mode === HOR_PRED && angleDelta === 0) {
    for (let y = 0; y < height; y++) dst.fill(left[y]!, (y0 + y) * stride + x0, (y0 + y) * stride + x0 + width);
    return;
  }
  if (mode >= VERT_PRED && mode <= 8) {
    directionalPredict(plane, x0, y0, width, height, top, left, topLeft,
      MODE_ANGLES[mode - VERT_PRED]! + 3 * angleDelta, context.sequence.intraEdgeFilter,
      haveTop, haveLeft, edgeFilterType, context.max, predictionWidth, predictionHeight);
    return;
  }
  if (mode === PAETH_PRED) {
    for (let y = 0; y < height; y++) {
      const row = (y0 + y) * stride + x0;
      for (let x = 0; x < width; x++) {
        const base = left[y]! + top[x]! - topLeft;
        const ld = Math.abs(left[y]! - base), td = Math.abs(top[x]! - base), tld = Math.abs(topLeft - base);
        dst[row + x] = ld <= td && ld <= tld ? left[y]! : td <= tld ? top[x]! : topLeft;
      }
    }
    return;
  }
  if (mode === SMOOTH_PRED || mode === SMOOTH_V_PRED || mode === SMOOTH_H_PRED) {
    const right = top[predictionWidth - 1]!;
    const bottom = left[predictionHeight - 1]!;
    for (let y = 0; y < height; y++) {
      const wy = smoothWeight(predictionHeight, y);
      const row = (y0 + y) * stride + x0;
      for (let x = 0; x < width; x++) {
        const wx = smoothWeight(predictionWidth, x);
        let value: number;
        if (mode === SMOOTH_V_PRED) value = (wy * top[x]! + (256 - wy) * bottom + 128) >> 8;
        else if (mode === SMOOTH_H_PRED) value = (wx * left[y]! + (256 - wx) * right + 128) >> 8;
        else value = (wy * top[x]! + (256 - wy) * bottom + wx * left[y]! + (256 - wx) * right + 256) >> 9;
        dst[row + x] = value;
      }
    }
    return;
  }
  // Corrupt/unknown modes are concealed with DC rather than leaking zeroes.
  fillBlock(plane, x0, y0, width, height, (context.max + 1) >> 1);
}

function filterIntraPredict(
  plane: Plane, x0: number, y0: number, width: number, height: number,
  predictionWidth: number, predictionHeight: number,
  top: Uint16Array, left: Uint16Array, topLeft: number, filterIndex: number, max: number,
): void {
  const taps = FILTER_INTRA_TAPS[Math.max(0, Math.min(4, filterIndex))]!;
  const data = plane.data, stride = plane.stride;
  for (let y = 0; y < height; y += 2) {
    for (let x = 0; x < width; x += 4) {
      // AV1 7.11.2.3: the first five inputs come from the row above
      // this 4x2 group (or AboveRow for the first group row); the last
      // two come from its left (or LeftCol for the first group column).
      // In particular, x>0 on y==0 still reads AboveRow[x-1], not a
      // non-existent reconstructed row at y0-1.
      const inputs = new Int32Array(7);
      for (let input = 0; input < 5; input++) {
        if (y === 0) {
          const topIndex = x + input - 1;
          inputs[input] = topIndex < 0 ? topLeft : top[Math.min(topIndex, top.length - 1)]!;
        } else if (x === 0 && input === 0) {
          inputs[input] = left[y - 1]!;
        } else {
          inputs[input] = data[(y0 + y - 1) * stride + x0 + x + input - 1]!;
        }
      }
      for (let input = 5; input < 7; input++) {
        const row = Math.min(y + input - 5, predictionHeight - 1);
        inputs[input] = x === 0 ? left[row]! : data[(y0 + row) * stride + x0 + x - 1]!;
      }
      for (let yy = 0; yy < 2 && y + yy < height; yy++) {
        for (let xx = 0; xx < 4 && x + xx < width; xx++) {
          const filter = taps[yy * 4 + xx]!;
          let sum = 0;
          for (let i = 0; i < 7; i++) sum += filter[i]! * inputs[i]!;
          data[(y0 + y + yy) * stride + x0 + x + xx] = clip((sum + 8) >> 4, 0, max);
        }
      }
    }
  }
}

function directionalPredict(
  plane: Plane, x0: number, y0: number, width: number, height: number,
  top: Uint16Array, left: Uint16Array, topLeft: number, angle: number,
  enableEdgeFilter: boolean, haveTop: boolean, haveLeft: boolean, smoothFilterType: number,
  maximum: number, predictionWidth = width, predictionHeight = height,
): void {
  const dst = plane.data, stride = plane.stride;
  if (angle === 90) {
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) dst[(y0 + y) * stride + x0 + x] = top[x]!;
    return;
  }
  if (angle === 180) {
    for (let y = 0; y < height; y++) dst.fill(left[y]!, (y0 + y) * stride + x0, (y0 + y) * stride + x0 + width);
    return;
  }
  const topEdge = makeDirectionalEdge(top, topLeft);
  const leftEdge = makeDirectionalEdge(left, topLeft);
  let upsampleTop = 0, upsampleLeft = 0;
  if (enableEdgeFilter) {
    if (angle !== 90 && angle !== 180) {
      if (angle > 90 && angle < 180 && predictionWidth + predictionHeight >= 24) {
        topLeft = round2(5 * leftEdge.get(0) + 6 * topLeft + 5 * topEdge.get(0), 4);
        topEdge.set(-1, topLeft); leftEdge.set(-1, topLeft);
      }
      if (haveTop) {
        const strength = intraEdgeFilterStrength(
          predictionWidth, predictionHeight, smoothFilterType, angle - 90,
        );
        const sampleCount = predictionWidth + (angle < 90 ? predictionHeight : 0) + 1;
        filterIntraEdge(topEdge, sampleCount, strength);
      }
      if (haveLeft) {
        const strength = intraEdgeFilterStrength(
          predictionWidth, predictionHeight, smoothFilterType, angle - 180,
        );
        const sampleCount = predictionHeight + (angle > 180 ? predictionWidth : 0) + 1;
        filterIntraEdge(leftEdge, sampleCount, strength);
      }
    }
    upsampleTop = +shouldUpsampleIntraEdge(
      predictionWidth, predictionHeight, smoothFilterType, angle - 90,
    );
    if (upsampleTop) {
      upsampleIntraEdge(topEdge, predictionWidth + (angle < 90 ? predictionHeight : 0), maximum);
    }
    upsampleLeft = +shouldUpsampleIntraEdge(
      predictionWidth, predictionHeight, smoothFilterType, angle - 180,
    );
    if (upsampleLeft) {
      upsampleIntraEdge(leftEdge, predictionHeight + (angle > 180 ? predictionWidth : 0), maximum);
    }
  }

  if (angle < 90) {
    const dx = derivative(angle);
    for (let y = 0; y < height; y++) {
      const index = (y + 1) * dx;
      for (let x = 0; x < width; x++) {
        const base = floorShift(index, 6 - upsampleTop) + (x << upsampleTop);
        const shift = lowFiveBits(floorShift(index * (1 << upsampleTop), 1));
        const maximumBase = (predictionWidth + predictionHeight - 1) << upsampleTop;
        dst[(y0 + y) * stride + x0 + x] = base < maximumBase ?
          round2(topEdge.get(base) * (32 - shift) + topEdge.get(base + 1) * shift, 5) :
          topEdge.get(maximumBase);
      }
    }
    return;
  }
  if (angle > 180) {
    const dy = derivative(270 - angle);
    for (let x = 0; x < width; x++) {
      const index = (x + 1) * dy;
      for (let y = 0; y < height; y++) {
        const base = floorShift(index, 6 - upsampleLeft) + (y << upsampleLeft);
        const shift = lowFiveBits(floorShift(index * (1 << upsampleLeft), 1));
        dst[(y0 + y) * stride + x0 + x] = round2(
          leftEdge.get(base) * (32 - shift) + leftEdge.get(base + 1) * shift, 5,
        );
      }
    }
    return;
  }

  const dx = derivative(180 - angle);
  const dy = derivative(angle - 90);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let index = (x << 6) - (y + 1) * dx;
      let base = floorShift(index, 6 - upsampleTop);
      let shift: number;
      let value: number;
      if (base >= -(1 << upsampleTop)) {
        shift = lowFiveBits(floorShift(index * (1 << upsampleTop), 1));
        value = topEdge.get(base) * (32 - shift) + topEdge.get(base + 1) * shift;
      } else {
        index = (y << 6) - (x + 1) * dy;
        base = floorShift(index, 6 - upsampleLeft);
        shift = lowFiveBits(floorShift(index * (1 << upsampleLeft), 1));
        value = leftEdge.get(base) * (32 - shift) + leftEdge.get(base + 1) * shift;
      }
      dst[(y0 + y) * stride + x0 + x] = round2(value, 5);
    }
  }
}

function cflPredict(
  context: ReconstructionContext, planeIndex: number,
  x0: number, y0: number, width: number, height: number, alpha: number,
  predictionWidth = width, predictionHeight = height,
): void {
  const plane = context.planes[planeIndex]!;
  const edges = collectEdges(context, planeIndex, x0, y0, predictionWidth, predictionHeight);
  let sum = 0, count = 0;
  if (edges.haveTop) {
    for (let x = 0; x < predictionWidth; x++) sum += edges.top[x]!;
    count += predictionWidth;
  }
  if (edges.haveLeft) {
    for (let y = 0; y < predictionHeight; y++) sum += edges.left[y]!;
    count += predictionHeight;
  }
  const dc = count ? Math.floor((sum + (count >> 1)) / count) : (context.max + 1) >> 1;

  const luma = context.planes[0]!;
  const ssX = context.sequence.subsamplingX, ssY = context.sequence.subsamplingY;
  const ac = new Int32Array(predictionWidth * predictionHeight);
  let average = 0;
  for (let y = 0; y < predictionHeight; y++) {
    for (let x = 0; x < predictionWidth; x++) {
      const lx = Math.min(luma.width - 1, (x0 + x) << ssX);
      const ly = Math.min(luma.height - 1, (y0 + y) << ssY);
      let lumaSum = 0;
      for (let sy = 0; sy < 1 << ssY; sy++) {
        const sampleY = Math.min(luma.height - 1, ly + sy);
        for (let sx = 0; sx < 1 << ssX; sx++) {
          lumaSum += luma.data[sampleY * luma.stride + Math.min(luma.width - 1, lx + sx)]!;
        }
      }
      const value = lumaSum << (3 - ssX - ssY);
      ac[y * predictionWidth + x] = value;
      average += value;
    }
  }
  const predictionArea = predictionWidth * predictionHeight;
  average = Math.floor((average + (predictionArea >> 1)) / predictionArea);
  for (let y = 0; y < height; y++) {
    const row = (y0 + y) * plane.stride + x0;
    for (let x = 0; x < width; x++) {
      const diff = alpha * (ac[y * predictionWidth + x]! - average);
      const delta = Math.sign(diff) * ((Math.abs(diff) + 32) >> 6);
      plane.data[row + x] = clip(dc + delta, 0, context.max);
    }
  }
}

function addResidual(
  context: ReconstructionContext, planeIndex: number, x0: number, y0: number,
  qIdx: number, result: CoefficientResult, tx: number,
): void {
  if (result.eob < 0) return;
  const info = transformSizes[tx]!;
  const width = info.w4 * 4, height = info.h4 * 4;
  const storedWidth = Math.min(width, 32), storedHeight = Math.min(height, 32);
  const coefficients = new Int32Array(width * height);
  const q = quantizers(context, planeIndex, qIdx);
  const shift = Math.max(0, info.ctx - 2);
  const qmLevel = planeIndex === 0 ? context.header.qmY : planeIndex === 1 ? context.header.qmU : context.header.qmV;
  const useQmatrix = context.header.usingQmatrix && result.txType < 9;
  for (let x = 0; x < storedWidth; x++) {
    for (let y = 0; y < storedHeight; y++) {
      const coefficient = x * storedHeight + y;
      const token = result.coefficients[coefficient]!;
      if (token) {
        let step = x === 0 && y === 0 ? q.dc : q.ac;
        if (useQmatrix) step = (step * quantMatrixValue(qmLevel, planeIndex !== 0, tx, coefficient) + 16) >> 5;
        coefficients[y * width + x] = signedDequant(token, step, shift);
      }
    }
  }
  const residual = inverseTransform2d(
    coefficients, width, height, tx, result.txType, context.sequence.bitDepth,
  );
  const plane = context.planes[planeIndex]!;
  const outWidth = Math.min(width, plane.width - x0), outHeight = Math.min(height, plane.height - y0);
  for (let y = 0; y < outHeight; y++) {
    const row = (y0 + y) * plane.stride + x0;
    for (let x = 0; x < outWidth; x++) plane.data[row + x] = clip(plane.data[row + x]! + residual[y * width + x]!, 0, context.max);
  }
}

function quantizers(context: ReconstructionContext, plane: number, qIdx: number): { dc: number; ac: number } {
  const hbd = context.sequence.bitDepth === 8 ? 0 : context.sequence.bitDepth === 10 ? 1 : 2;
  let dcDelta: number, acDelta: number;
  if (plane === 0) { dcDelta = context.header.yDcDelta; acDelta = 0; }
  else if (plane === 1) { dcDelta = context.header.uDcDelta; acDelta = context.header.uAcDelta; }
  else { dcDelta = context.header.vDcDelta; acDelta = context.header.vAcDelta; }
  const table = dequantizers[hbd]!;
  return {
    dc: table[clip(qIdx + dcDelta, 0, 255)]![0],
    ac: table[clip(qIdx + acDelta, 0, 255)]![1],
  };
}

function signedDequant(token: number, step: number, shift: number): number {
  const magnitude = (Math.abs(token) * step) >> shift;
  return token < 0 ? -magnitude : magnitude;
}

function collectEdges(
  context: ReconstructionContext, planeIndex: number,
  x0: number, y0: number, width: number, height: number,
): { top: Uint16Array; left: Uint16Array; topLeft: number; haveTop: boolean; haveLeft: boolean } {
  const plane = context.planes[planeIndex]!;
  const mask = context.masks[planeIndex]!;
  const extension = width + height;
  const top = new Uint16Array(extension);
  const left = new Uint16Array(extension);
  const haveTop = y0 > 0 && maskValue(mask, Math.min(x0, plane.width - 1), y0 - 1) !== 0;
  const haveLeft = x0 > 0 && maskValue(mask, x0 - 1, Math.min(y0, plane.height - 1)) !== 0;
  const midpoint = (context.max + 1) >> 1;

  let lastTop = haveTop ? plane.data[(y0 - 1) * plane.stride + x0]! :
    haveLeft ? plane.data[y0 * plane.stride + x0 - 1]! : midpoint - 1;
  const topSampleLimit = width + Math.min(width, height);
  let topEnded = false;
  for (let i = 0; i < extension; i++) {
    const x = Math.min(x0 + i, plane.width - 1);
    if (i < topSampleLimit && haveTop && !topEnded && maskValue(mask, x, y0 - 1)) {
      lastTop = plane.data[(y0 - 1) * plane.stride + x]!;
    } else if (haveTop) topEnded = true;
    top[i] = lastTop;
  }
  let lastLeft = haveLeft ? plane.data[y0 * plane.stride + x0 - 1]! :
    haveTop ? plane.data[(y0 - 1) * plane.stride + x0]! : midpoint + 1;
  const leftSampleLimit = height + Math.min(width, height);
  let leftEnded = false;
  for (let i = 0; i < extension; i++) {
    const y = Math.min(y0 + i, plane.height - 1);
    if (i < leftSampleLimit && haveLeft && !leftEnded && maskValue(mask, x0 - 1, y)) {
      lastLeft = plane.data[y * plane.stride + x0 - 1]!;
    } else if (haveLeft) leftEnded = true;
    left[i] = lastLeft;
  }
  let topLeft = midpoint;
  if (haveTop && haveLeft && maskValue(mask, x0 - 1, y0 - 1)) topLeft = plane.data[(y0 - 1) * plane.stride + x0 - 1]!;
  else if (haveTop) topLeft = top[0]!;
  else if (haveLeft) topLeft = left[0]!;
  return { top, left, topLeft, haveTop, haveLeft };
}

function mark(context: ReconstructionContext, plane: number, x0: number, y0: number, width: number, height: number): void {
  const target = context.masks[plane]!;
  fillMask(target, x0, y0, width, height, 1);
}

function markSmooth(
  context: ReconstructionContext, plane: number,
  x0: number, y0: number, width: number, height: number, smooth: boolean,
): void {
  const target = context.smoothMasks[plane]!;
  fillMask(target, x0, y0, width, height, +smooth);
}

function filterType(
  context: ReconstructionContext, plane: number, x0: number, y0: number,
): number {
  const smooth = context.smoothMasks[plane]!;
  return +((y0 > 0 && maskValue(smooth, x0, y0 - 1) !== 0) ||
    (x0 > 0 && maskValue(smooth, x0 - 1, y0) !== 0));
}

function maskValue(mask: ReconstructionMask, x: number, y: number): number {
  const localX = x - mask.originX, localY = y - mask.originY;
  return localX < 0 || localY < 0 || localX >= mask.width || localY >= mask.height ? 0 :
    mask.data[localY * mask.width + localX]!;
}

function fillMask(
  mask: ReconstructionMask, x: number, y: number, width: number, height: number, value: number,
): void {
  const startX = Math.max(0, x - mask.originX);
  const endX = Math.min(mask.width, x - mask.originX + width);
  const startY = Math.max(0, y - mask.originY);
  const endY = Math.min(mask.height, y - mask.originY + height);
  if (startX >= endX || startY >= endY) return;
  for (let localY = startY; localY < endY; localY++) {
    mask.data.fill(value, localY * mask.width + startX, localY * mask.width + endX);
  }
}

function fillBlock(plane: Plane, x0: number, y0: number, width: number, height: number, value: number): void {
  for (let y = 0; y < height; y++) plane.data.fill(value, (y0 + y) * plane.stride + x0, (y0 + y) * plane.stride + x0 + width);
}

function smoothWeight(size: number, position: number): number {
  return sm_weights[Math.min(127, size + position)] ?? Math.max(0, 255 - Math.floor(position * 256 / size));
}

function derivative(angle: number): number {
  return dr_intra_derivative[Math.max(0, Math.min(dr_intra_derivative.length - 1, angle >> 1))] ?? 0;
}

interface DirectionalEdge {
  values: Int32Array;
  offset: number;
  get(index: number): number;
  set(index: number, value: number): void;
}

function makeDirectionalEdge(input: Uint16Array, topLeft: number): DirectionalEdge {
  const offset = 2;
  const values = new Int32Array(input.length * 2 + 8);
  values.fill(input[input.length - 1] ?? topLeft);
  values[offset - 2] = topLeft;
  values[offset - 1] = topLeft;
  values.set(input, offset);
  return {
    values, offset,
    get(index: number): number {
      return values[clip(index + offset, 0, values.length - 1)]!;
    },
    set(index: number, value: number): void {
      values[clip(index + offset, 0, values.length - 1)] = value;
    },
  };
}

function intraEdgeFilterStrength(width: number, height: number, type: number, delta: number): number {
  const difference = Math.abs(delta), size = width + height;
  if (type === 0) {
    if (size <= 8) return difference >= 56 ? 1 : 0;
    if (size <= 12) return difference >= 40 ? 1 : 0;
    if (size <= 16) return difference >= 40 ? 1 : 0;
    if (size <= 24) return difference >= 32 ? 3 : difference >= 16 ? 2 : difference >= 8 ? 1 : 0;
    if (size <= 32) return difference >= 32 ? 3 : difference >= 4 ? 2 : 1;
    return 3;
  }
  if (size <= 8) return difference >= 64 ? 2 : difference >= 40 ? 1 : 0;
  if (size <= 16) return difference >= 48 ? 2 : difference >= 20 ? 1 : 0;
  if (size <= 24) return difference >= 4 ? 3 : 0;
  return 3;
}

function shouldUpsampleIntraEdge(width: number, height: number, type: number, delta: number): boolean {
  const difference = Math.abs(delta);
  if (difference <= 0 || difference >= 40) return false;
  return type === 0 ? width + height <= 16 : width + height <= 8;
}

function filterIntraEdge(edge: DirectionalEdge, size: number, strength: number): void {
  if (!strength) return;
  const kernels = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]] as const;
  const kernel = kernels[strength - 1]!;
  const copy = new Int32Array(size);
  for (let i = 0; i < size; i++) copy[i] = edge.get(i - 1);
  for (let i = 1; i < size; i++) {
    let sum = 0;
    for (let tap = 0; tap < 5; tap++) {
      sum += kernel[tap]! * copy[clip(i - 2 + tap, 0, size - 1)]!;
    }
    edge.set(i - 1, round2(sum, 4));
  }
}

function upsampleIntraEdge(edge: DirectionalEdge, sampleCount: number, maximum: number): void {
  const duplicate = new Int32Array(sampleCount + 3);
  duplicate[0] = edge.get(-1);
  for (let i = -1; i < sampleCount; i++) duplicate[i + 2] = edge.get(i);
  duplicate[sampleCount + 2] = edge.get(sampleCount - 1);
  edge.set(-2, duplicate[0]!);
  for (let i = 0; i < sampleCount; i++) {
    const sum = -duplicate[i]! + 9 * duplicate[i + 1]! +
      9 * duplicate[i + 2]! - duplicate[i + 3]!;
    edge.set(2 * i - 1, clip(round2(sum, 4), 0, maximum));
    edge.set(2 * i, duplicate[i + 2]!);
  }
}

function round2(value: number, bits: number): number {
  return bits ? Math.floor((value + 2 ** (bits - 1)) / 2 ** bits) : value;
}

function floorShift(value: number, bits: number): number {
  return bits ? Math.floor(value / 2 ** bits) : value;
}

function lowFiveBits(value: number): number {
  return ((value % 32) + 32) % 32;
}

function clip(value: number, minimum: number, maximum: number): number {
  return value < minimum ? minimum : value > maximum ? maximum : value;
}
