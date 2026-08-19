import type { DecodedFrame, SampleArray } from '../frame.ts';
import type { Av1FilmGrain, Av1SequenceHeader } from './obu.ts';
import { gaussianSequence } from './grain_data.ts';

const GRAIN_WIDTH = 82, GRAIN_HEIGHT = 73, BLOCK_SIZE = 32;
const FILM_GRAIN_SCRATCH = {
  lumaGrain: new Int16Array(GRAIN_WIDTH * GRAIN_HEIGHT),
  chromaGrain: new Int16Array(GRAIN_WIDTH * GRAIN_HEIGHT),
  yScaling: new Uint8Array(1 << 12),
  uvScaling: new Uint8Array(1 << 12),
  currentOffsets: new Uint8Array(0),
  previousOffsets: new Uint8Array(0),
};

/** Synthesize AV1 film grain over the fully filtered output frame. */
export function applyFilmGrain(
  frame: DecodedFrame, data: Av1FilmGrain | null, sequence: Av1SequenceHeader,
): void {
  if (!data) return;
  const activeY = data.yPoints.length > 0;
  const activeChroma: [boolean, boolean] = frame.planes.length > 1 ? [
    data.uvPoints[0].length > 0 || data.chromaScalingFromLuma,
    data.uvPoints[1].length > 0 || data.chromaScalingFromLuma,
  ] : [false, false];
  if (!activeY && !activeChroma[0] && !activeChroma[1]) return;
  const source = frame.planes.map((plane, index) => {
    if (index === 0) return activeY ? plane.data.slice() : plane.data;
    return activeChroma[index - 1] ? plane.data.slice() : null;
  });
  const yScalingNeeded = activeY || data.chromaScalingFromLuma && (activeChroma[0] || activeChroma[1]);
  const yScaling = yScalingNeeded ?
    generateScaling(sequence.bitDepth, data.yPoints, FILM_GRAIN_SCRATCH.yScaling) : null;
  const lumaGrain = activeY ?
    generateLumaGrain(data, sequence.bitDepth, FILM_GRAIN_SCRATCH.lumaGrain) : FILM_GRAIN_SCRATCH.lumaGrain;

  if (activeY) {
    applyPlaneGrain(frame, source, 0, lumaGrain, yScaling!, data, sequence, 0, 0);
  }
  for (let plane = 0; plane < 2; plane++) {
    if (!activeChroma[plane]) continue;
    const chromaGrain = generateChromaGrain(
      data, lumaGrain, plane, sequence.subsamplingX, sequence.subsamplingY, sequence.bitDepth,
      FILM_GRAIN_SCRATCH.chromaGrain,
    );
    const scaling = data.chromaScalingFromLuma ? yScaling! :
      generateScaling(sequence.bitDepth, data.uvPoints[plane], FILM_GRAIN_SCRATCH.uvScaling);
    applyPlaneGrain(frame, source, plane + 1, chromaGrain, scaling,
      data, sequence, sequence.subsamplingX, sequence.subsamplingY);
  }
}

function generateLumaGrain(data: Av1FilmGrain, bitDepth: number, grain: Int16Array): Int16Array {
  const state = { value: data.seed };
  const bitDepthShift = bitDepth - 8;
  const shift = 4 - bitDepthShift + data.grainScaleShift;
  const center = 128 << bitDepthShift;
  for (let y = 0; y < GRAIN_HEIGHT; y++) {
    for (let x = 0; x < GRAIN_WIDTH; x++) {
      grain[y * GRAIN_WIDTH + x] = round2(gaussianSequence[randomNumber(11, state)]!, shift);
    }
  }
  const lag = data.arCoeffLag;
  for (let y = 3; y < GRAIN_HEIGHT; y++) {
    for (let x = 3; x < GRAIN_WIDTH - 3; x++) {
      let coefficient = 0, sum = 0;
      for (let dy = -lag; dy <= 0; dy++) {
        for (let dx = -lag; dx <= lag; dx++) {
          if (!dx && !dy) break;
          sum += data.arCoeffsY[coefficient++]! * grain[(y + dy) * GRAIN_WIDTH + x + dx]!;
        }
      }
      grain[y * GRAIN_WIDTH + x] = clamp(
        grain[y * GRAIN_WIDTH + x]! + round2(sum, data.arCoeffShift), -center, center - 1,
      );
    }
  }
  return grain;
}

function generateChromaGrain(
  data: Av1FilmGrain, luma: Int16Array, plane: number,
  subX: number, subY: number, bitDepth: number,
  grain: Int16Array,
): Int16Array {
  const state = { value: data.seed ^ (plane ? 0x49d8 : 0xb524) };
  const bitDepthShift = bitDepth - 8;
  const shift = 4 - bitDepthShift + data.grainScaleShift;
  const center = 128 << bitDepthShift;
  const width = subX ? 44 : GRAIN_WIDTH, height = subY ? 38 : GRAIN_HEIGHT;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    grain[y * GRAIN_WIDTH + x] = round2(gaussianSequence[randomNumber(11, state)]!, shift);
  }
  const lag = data.arCoeffLag;
  for (let y = 3; y < height; y++) {
    for (let x = 3; x < width - 3; x++) {
      let coefficient = 0, sum = 0;
      for (let dy = -lag; dy <= 0; dy++) {
        for (let dx = -lag; dx <= lag; dx++) {
          if (!dx && !dy) {
            if (!data.yPoints.length) break;
            const lumaX = ((x - 3) << subX) + 3;
            const lumaY = ((y - 3) << subY) + 3;
            let lumaGrain = 0;
            for (let yy = 0; yy <= subY; yy++) for (let xx = 0; xx <= subX; xx++) {
              lumaGrain += luma[(lumaY + yy) * GRAIN_WIDTH + lumaX + xx]!;
            }
            sum += round2(lumaGrain, subX + subY) * data.arCoeffsUv[plane][coefficient]!;
            break;
          }
          sum += data.arCoeffsUv[plane][coefficient++]! *
            grain[(y + dy) * GRAIN_WIDTH + x + dx]!;
        }
      }
      grain[y * GRAIN_WIDTH + x] = clamp(
        grain[y * GRAIN_WIDTH + x]! + round2(sum, data.arCoeffShift), -center, center - 1,
      );
    }
  }
  return grain;
}

function applyPlaneGrain(
  frame: DecodedFrame, source: (SampleArray | null)[], planeIndex: number,
  grain: Int16Array, scaling: Uint8Array, data: Av1FilmGrain,
  sequence: Av1SequenceHeader, subX: number, subY: number,
): void {
  const plane = frame.planes[planeIndex]!;
  const sourcePlane = source[planeIndex]!;
  const luma = frame.planes[0]!, sourceLuma = source[0]!;
  const blockWidth = BLOCK_SIZE >> subX, blockHeight = BLOCK_SIZE >> subY;
  const blockColumns = Math.ceil(plane.width / blockWidth);
  const blockRows = Math.ceil(plane.height / blockHeight);
  const bitDepthShift = sequence.bitDepth - 8;
  const grainCenter = 128 << bitDepthShift;
  const minimum = data.clipToRestrictedRange ? 16 << bitDepthShift : 0;
  const identity = sequence.matrixCoefficients === 0;
  const maximum = data.clipToRestrictedRange
    ? (planeIndex === 0 || identity ? 235 : 240) << bitDepthShift
    : (1 << sequence.bitDepth) - 1;
  if (FILM_GRAIN_SCRATCH.currentOffsets.length < blockColumns) {
    FILM_GRAIN_SCRATCH.currentOffsets = new Uint8Array(blockColumns);
    FILM_GRAIN_SCRATCH.previousOffsets = new Uint8Array(blockColumns);
  }
  let currentOffsets = FILM_GRAIN_SCRATCH.currentOffsets;
  let previousOffsets = FILM_GRAIN_SCRATCH.previousOffsets;
  const grainStepX = 2 >> subX, grainStepY = 2 >> subY;

  for (let blockRow = 0; blockRow < blockRows; blockRow++) {
    const swap = previousOffsets; previousOffsets = currentOffsets; currentOffsets = swap;
    let offsetState = data.seed ^ ((((blockRow * 37 + 178) & 0xff) << 8)) ^
      ((blockRow * 173 + 105) & 0xff);
    for (let column = 0; column < blockColumns; column++) {
      const bit = ((offsetState >> 0) ^ (offsetState >> 1) ^ (offsetState >> 3) ^ (offsetState >> 12)) & 1;
      offsetState = ((offsetState >> 1) | (bit << 15)) & 0xffff;
      currentOffsets[column] = offsetState >> 8;
    }
    const y0 = blockRow * blockHeight;
    const height = Math.min(blockHeight, plane.height - y0);
    for (let blockColumn = 0; blockColumn < blockColumns; blockColumn++) {
      const x0 = blockColumn * blockWidth;
      const width = Math.min(blockWidth, plane.width - x0);
      const overlapX = data.overlap && blockColumn ? Math.min(2 >> subX, width) : 0;
      const overlapY = data.overlap && blockRow ? Math.min(2 >> subY, height) : 0;
      const random = currentOffsets[blockColumn]!;
      const currentBase = (3 + grainStepY * (3 + (random & 15))) * GRAIN_WIDTH +
        3 + grainStepX * (3 + (random >> 4));
      let leftBase = 0, topBase = 0, topLeftBase = 0;
      if (overlapX) {
        const leftRandom = currentOffsets[blockColumn - 1]!;
        leftBase = (3 + grainStepY * (3 + (leftRandom & 15))) * GRAIN_WIDTH +
          3 + grainStepX * (3 + (leftRandom >> 4)) + blockWidth;
      }
      if (overlapY) {
        const topRandom = previousOffsets[blockColumn]!;
        topBase = (3 + grainStepY * (3 + (topRandom & 15)) + blockHeight) * GRAIN_WIDTH +
          3 + grainStepX * (3 + (topRandom >> 4));
        if (overlapX) {
          const topLeftRandom = previousOffsets[blockColumn - 1]!;
          topLeftBase = (3 + grainStepY * (3 + (topLeftRandom & 15)) + blockHeight) * GRAIN_WIDTH +
            3 + grainStepX * (3 + (topLeftRandom >> 4)) + blockWidth;
        }
      }
      for (let y = 0; y < height; y++) {
        const grainRow = y * GRAIN_WIDTH;
        const planeRow = (y0 + y) * plane.stride + x0;
        const lumaY = (y0 + y) << subY;
        const lumaRow = lumaY * luma.stride;
        for (let x = 0; x < width; x++) {
          let value = grain[currentBase + grainRow + x]!;
          if (x < overlapX) {
            const old = grain[leftBase + grainRow + x]!;
            value = blend(old, value, subX, x);
          }
          if (y < overlapY) {
            let top = grain[topBase + grainRow + x]!;
            if (x < overlapX) {
              const topLeft = grain[topLeftBase + grainRow + x]!;
              top = blend(topLeft, top, subX, x);
            }
            value = blend(top, value, subY, y);
          }
          value = value < -grainCenter ? -grainCenter : value >= grainCenter ? grainCenter - 1 : value;

          const globalX = x0 + x;
          const sourceValue = sourcePlane[planeRow + x]!;
          let scalingIndex = sourceValue;
          if (planeIndex) {
            const lumaX = globalX << subX;
            let average = sourceLuma[lumaRow + lumaX]!;
            if (subX) {
              const adjacentX = lumaX + 1 < luma.width ? lumaX + 1 : lumaX;
              average = (average + sourceLuma[lumaRow + adjacentX]! + 1) >> 1;
            }
            if (data.chromaScalingFromLuma) scalingIndex = average;
            else {
              const uv = planeIndex - 1;
              const combined = ((average * data.uvLumaMult[uv] + sourceValue * data.uvMult[uv]) >> 6) +
                (data.uvOffset[uv] << bitDepthShift);
              const maxIndex = (1 << sequence.bitDepth) - 1;
              scalingIndex = combined < 0 ? 0 : combined > maxIndex ? maxIndex : combined;
            }
          }
          const noise = round2(scaling[scalingIndex]! * value, data.scalingShift);
          const output = sourceValue + noise;
          plane.data[planeRow + x] = output < minimum ? minimum : output > maximum ? maximum : output;
        }
      }
    }
  }
}

function blend(oldValue: number, newValue: number, subsampling: number, offset: number): number {
  if (subsampling) return round2(oldValue * 23 + newValue * 22, 5);
  return offset ? round2(oldValue * 17 + newValue * 27, 5) :
    round2(oldValue * 27 + newValue * 17, 5);
}

function generateScaling(
  bitDepth: number, points: readonly [number, number][], scaling: Uint8Array,
): Uint8Array {
  const size = 1 << bitDepth;
  if (!points.length) {
    scaling.fill(0, 0, size);
    return scaling;
  }
  const shift = bitDepth - 8;
  scaling.fill(points[0]![1], 0, points[0]![0] << shift);
  for (let index = 0; index < points.length - 1; index++) {
    const [beginX, beginY] = points[index]!, [endX, endY] = points[index + 1]!;
    const distance = endX - beginX;
    const delta = (endY - beginY) * Math.floor((0x10000 + (distance >> 1)) / distance);
    for (let x = 0, accumulator = 0x8000; x < distance; x++, accumulator += delta) {
      scaling[(beginX + x) << shift] = beginY + (accumulator >> 16);
    }
  }
  scaling.fill(points[points.length - 1]![1], points[points.length - 1]![0] << shift);
  if (shift) {
    const step = 1 << shift, rounding = step >> 1;
    for (let index = 0; index < points.length - 1; index++) {
      const begin = points[index]![0] << shift, end = points[index + 1]![0] << shift;
      for (let x = begin; x < end; x += step) {
        const range = scaling[x + step]! - scaling[x]!;
        for (let offset = 1, accumulator = rounding + range; offset < step; offset++, accumulator += range) {
          scaling[x + offset] = scaling[x]! + (accumulator >> shift);
        }
      }
    }
  }
  return scaling;
}

function randomNumber(bits: number, state: { value: number }): number {
  const bit = ((state.value >> 0) ^ (state.value >> 1) ^ (state.value >> 3) ^ (state.value >> 12)) & 1;
  state.value = ((state.value >> 1) | (bit << 15)) & 0xffff;
  return (state.value >> (16 - bits)) & ((1 << bits) - 1);
}

function round2(value: number, shift: number): number {
  return shift ? (value + ((1 << shift) >> 1)) >> shift : value;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
