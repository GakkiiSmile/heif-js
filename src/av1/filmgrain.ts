import type { DecodedFrame } from '../frame.ts';
import type { Av1FilmGrain, Av1SequenceHeader } from './obu.ts';
import { gaussianSequence } from './grain_data.ts';

const GRAIN_WIDTH = 82, GRAIN_HEIGHT = 73, BLOCK_SIZE = 32;

/** Synthesize AV1 film grain over the fully filtered output frame. */
export function applyFilmGrain(
  frame: DecodedFrame, data: Av1FilmGrain | null, sequence: Av1SequenceHeader,
): void {
  if (!data) return;
  const source = frame.planes.map(plane => new Uint16Array(plane.data));
  const scaling = [
    generateScaling(sequence.bitDepth, data.yPoints),
    generateScaling(sequence.bitDepth, data.uvPoints[0]),
    generateScaling(sequence.bitDepth, data.uvPoints[1]),
  ];
  const lumaGrain = generateLumaGrain(data, sequence.bitDepth);
  const chromaGrain: [Int16Array | null, Int16Array | null] = [null, null];
  if (frame.planes.length > 1) {
    for (let plane = 0; plane < 2; plane++) {
      if (data.uvPoints[plane].length || data.chromaScalingFromLuma) {
        chromaGrain[plane] = generateChromaGrain(
          data, lumaGrain, plane, sequence.subsamplingX, sequence.subsamplingY, sequence.bitDepth,
        );
      }
    }
  }

  if (data.yPoints.length) {
    applyPlaneGrain(frame, source, 0, lumaGrain, scaling[0]!, data, sequence, 0, 0);
  }
  for (let plane = 0; plane < 2; plane++) {
    if (chromaGrain[plane]) {
      applyPlaneGrain(frame, source, plane + 1, chromaGrain[plane]!,
        data.chromaScalingFromLuma ? scaling[0]! : scaling[plane + 1]!,
        data, sequence, sequence.subsamplingX, sequence.subsamplingY);
    }
  }
}

function generateLumaGrain(data: Av1FilmGrain, bitDepth: number): Int16Array {
  const grain = new Int16Array(GRAIN_WIDTH * GRAIN_HEIGHT);
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
): Int16Array {
  const grain = new Int16Array(GRAIN_WIDTH * GRAIN_HEIGHT);
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
  frame: DecodedFrame, source: Uint16Array[], planeIndex: number,
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

  const offsets: number[][] = Array.from({ length: blockRows }, (_, row) => {
    const state = {
      value: data.seed ^ ((((row * 37 + 178) & 0xff) << 8)) ^ ((row * 173 + 105) & 0xff),
    };
    return Array.from({ length: blockColumns }, () => randomNumber(8, state));
  });

  for (let blockRow = 0; blockRow < blockRows; blockRow++) {
    const y0 = blockRow * blockHeight;
    const height = Math.min(blockHeight, plane.height - y0);
    for (let blockColumn = 0; blockColumn < blockColumns; blockColumn++) {
      const x0 = blockColumn * blockWidth;
      const width = Math.min(blockWidth, plane.width - x0);
      const overlapX = data.overlap && blockColumn ? Math.min(2 >> subX, width) : 0;
      const overlapY = data.overlap && blockRow ? Math.min(2 >> subY, height) : 0;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          let value = sampleGrain(grain, offsets, blockColumn, blockRow, subX, subY, 0, 0, x, y);
          if (x < overlapX) {
            const old = sampleGrain(grain, offsets, blockColumn, blockRow, subX, subY, 1, 0, x, y);
            value = blend(old, value, subX, x);
          }
          if (y < overlapY) {
            let top = sampleGrain(grain, offsets, blockColumn, blockRow, subX, subY, 0, 1, x, y);
            if (x < overlapX) {
              const topLeft = sampleGrain(grain, offsets, blockColumn, blockRow, subX, subY, 1, 1, x, y);
              top = blend(topLeft, top, subX, x);
            }
            value = blend(top, value, subY, y);
          }
          value = clamp(value, -grainCenter, grainCenter - 1);

          const globalX = x0 + x, globalY = y0 + y;
          const sourceValue = sourcePlane[globalY * plane.stride + globalX]!;
          let scalingIndex = sourceValue;
          if (planeIndex) {
            const lumaX = globalX << subX, lumaY = globalY << subY;
            let average = sourceLuma[lumaY * luma.stride + lumaX]!;
            if (subX) average = (average + sourceLuma[lumaY * luma.stride + Math.min(luma.width - 1, lumaX + 1)]! + 1) >> 1;
            if (data.chromaScalingFromLuma) scalingIndex = average;
            else {
              const uv = planeIndex - 1;
              scalingIndex = clamp(((average * data.uvLumaMult[uv] +
                sourceValue * data.uvMult[uv]) >> 6) + (data.uvOffset[uv] << bitDepthShift),
              0, (1 << sequence.bitDepth) - 1);
            }
          }
          const noise = round2(scaling[scalingIndex]! * value, data.scalingShift);
          plane.data[globalY * plane.stride + globalX] = clamp(sourceValue + noise, minimum, maximum);
        }
      }
    }
  }
}

function sampleGrain(
  grain: Int16Array, offsets: number[][], blockX: number, blockY: number,
  subX: number, subY: number, previousX: number, previousY: number, x: number, y: number,
): number {
  const random = offsets[blockY - previousY]![blockX - previousX]!;
  const offsetX = 3 + (2 >> subX) * (3 + (random >> 4));
  const offsetY = 3 + (2 >> subY) * (3 + (random & 15));
  const lutX = offsetX + x + (BLOCK_SIZE >> subX) * previousX;
  const lutY = offsetY + y + (BLOCK_SIZE >> subY) * previousY;
  return grain[lutY * GRAIN_WIDTH + lutX]!;
}

function blend(oldValue: number, newValue: number, subsampling: number, offset: number): number {
  if (subsampling) return round2(oldValue * 23 + newValue * 22, 5);
  return offset ? round2(oldValue * 17 + newValue * 27, 5) :
    round2(oldValue * 27 + newValue * 17, 5);
}

function generateScaling(bitDepth: number, points: readonly [number, number][]): Uint8Array {
  const size = 1 << bitDepth;
  const scaling = new Uint8Array(size);
  if (!points.length) return scaling;
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
