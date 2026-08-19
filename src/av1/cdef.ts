import type { DecodedFrame, Plane } from '../frame.ts';
import type { Av1DecodedBlock } from './decode.ts';
import type { Av1FrameHeader, Av1SequenceHeader } from './obu.ts';
import { block_dimensions } from './tables_data.ts';

const DIRECTIONS: readonly (readonly [readonly [number, number], readonly [number, number]])[] = [
  [[0, 1], [0, 2]], [[0, 1], [-1, 2]], [[1, -1], [2, -2]], [[1, 0], [2, -1]],
  [[1, 0], [2, 0]], [[1, 0], [2, 1]], [[1, 1], [2, 2]], [[0, 1], [1, 2]],
  [[0, 1], [0, 2]], [[0, 1], [-1, 2]], [[1, -1], [2, -2]], [[1, 0], [2, -1]],
] as const;
const UV_DIRECTIONS_422 = [7, 0, 2, 4, 5, 6, 6, 6] as const;

interface DirectionScratch {
  horizontalVertical: Int32Array;
  diagonal: Int32Array;
  alternate: Int32Array;
  cost: Float64Array;
  direction: number;
  variance: number;
}

function createDirectionScratch(): DirectionScratch {
  return {
    horizontalVertical: new Int32Array(16),
    diagonal: new Int32Array(30),
    alternate: new Int32Array(44),
    cost: new Float64Array(8),
    direction: 0,
    variance: 0,
  };
}

/** Apply AV1's constrained directional enhancement filter to a reconstructed frame. */
export function applyCdef(
  frame: DecodedFrame, blocks: readonly Av1DecodedBlock[],
  sequence: Av1SequenceHeader, header: Av1FrameHeader, sourceFrame: DecodedFrame | null = null,
): void {
  if (!header.cdefYStrength.length ||
      !header.cdefYStrength.some(Boolean) && !header.cdefUvStrength.some(Boolean)) return;

  const width8 = Math.ceil(frame.width / 8), height8 = Math.ceil(frame.height / 8);
  const nonSkip = new Uint8Array(width8 * height8);
  const indices = new Int8Array(width8 * height8).fill(-1);
  for (const block of blocks) {
    if (block.skip || block.cdefIndex < 0) continue;
    const dimensions = block_dimensions[block.blockSize]!;
    const startX = block.x4 >> 1, startY = block.y4 >> 1;
    const endX = Math.min(width8, (block.x4 + dimensions[0]! + 1) >> 1);
    const endY = Math.min(height8, (block.y4 + dimensions[1]! + 1) >> 1);
    for (let y = startY; y < endY; y++) {
      for (let x = startX; x < endX; x++) {
        nonSkip[y * width8 + x] = 1;
        indices[y * width8 + x] = block.cdefIndex;
      }
    }
  }

  const filterLuma = header.cdefYStrength.some(Boolean);
  const filterChroma = header.cdefUvStrength.some(Boolean);
  // Restoration already keeps an immutable pre-CDEF/deblocked frame. Reuse it
  // instead of copying every active plane a second time.
  const sources = sourceFrame && sourceFrame !== frame ? sourceFrame.planes :
    frame.planes.map((plane, index) =>
      (index === 0 ? filterLuma : filterChroma) ? copyPlane(plane) : plane);
  const directionScratch = createDirectionScratch();
  const bitDepthShift = sequence.bitDepth - 8;
  const damping = header.cdefDamping + bitDepthShift;
  const ssX = sequence.subsamplingX, ssY = sequence.subsamplingY;
  for (let by8 = 0; by8 < height8; by8++) {
    for (let bx8 = 0; bx8 < width8; bx8++) {
      const mapIndex = by8 * width8 + bx8;
      if (!nonSkip[mapIndex]) continue;
      const cdefIndex = indices[mapIndex]!;
      if (cdefIndex < 0) continue;
      const yLevel = header.cdefYStrength[cdefIndex] ?? 0;
      const uvLevel = header.cdefUvStrength[cdefIndex] ?? 0;
      if (!yLevel && !uvLevel) continue;

      const x = bx8 * 8, y = by8 * 8;
      findDirection(sources[0]!, x, y, bitDepthShift, directionScratch);
      const direction = directionScratch.direction, variance = directionScratch.variance;
      const yPrimary = adjustStrength((yLevel >> 2) << bitDepthShift, variance);
      const ySecondaryCode = yLevel & 3;
      const ySecondary = (ySecondaryCode + +(ySecondaryCode === 3)) << bitDepthShift;
      if (yPrimary || ySecondary) {
        filterBlock(sources[0]!, frame.planes[0]!, x, y, 8, 8,
          yPrimary, ySecondary, direction, damping, bitDepthShift);
      }

      if (!uvLevel || frame.planes.length === 1) continue;
      const uvPrimary = (uvLevel >> 2) << bitDepthShift;
      const uvSecondaryCode = uvLevel & 3;
      const uvSecondary = (uvSecondaryCode + +(uvSecondaryCode === 3)) << bitDepthShift;
      const uvDirection = ssX && !ssY ? UV_DIRECTIONS_422[direction]! : direction;
      const uvX = x >> ssX, uvY = y >> ssY;
      for (let plane = 1; plane <= 2; plane++) {
        filterBlock(sources[plane]!, frame.planes[plane]!, uvX, uvY,
          8 >> ssX, 8 >> ssY, uvPrimary, uvSecondary, uvDirection, damping - 1, bitDepthShift);
      }
    }
  }
}

function copyPlane(plane: Plane): Plane {
  return { width: plane.width, height: plane.height, stride: plane.stride, data: plane.data.slice() };
}

function findDirection(
  plane: Plane, x0: number, y0: number, bitDepthShift: number, scratch: DirectionScratch,
): void {
  const horizontalVertical = scratch.horizontalVertical;
  const diagonal = scratch.diagonal;
  const alternate = scratch.alternate;
  const cost = scratch.cost;
  horizontalVertical.fill(0);
  diagonal.fill(0);
  alternate.fill(0);
  cost.fill(0);
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      const sampleX = Math.min(plane.width - 1, x0 + x);
      const sampleY = Math.min(plane.height - 1, y0 + y);
      const pixel = (plane.data[sampleY * plane.stride + sampleX]! >> bitDepthShift) - 128;
      diagonal[y + x] += pixel;
      alternate[y + (x >> 1)] += pixel;
      horizontalVertical[y] += pixel;
      alternate[11 + 3 + y - (x >> 1)] += pixel;
      diagonal[15 + 7 + y - x] += pixel;
      alternate[22 + 3 - (y >> 1) + x] += pixel;
      horizontalVertical[8 + x] += pixel;
      alternate[33 + (y >> 1) + x] += pixel;
    }
  }
  for (let index = 0; index < 8; index++) {
    cost[2] += horizontalVertical[index]! ** 2;
    cost[6] += horizontalVertical[8 + index]! ** 2;
  }
  cost[2] *= 105;
  cost[6] *= 105;
  const divisors = [840, 420, 280, 210, 168, 140, 120] as const;
  for (let index = 0; index < 7; index++) {
    cost[0] += (diagonal[index]! ** 2 + diagonal[14 - index]! ** 2) * divisors[index]!;
    cost[4] += (diagonal[15 + index]! ** 2 + diagonal[15 + 14 - index]! ** 2) * divisors[index]!;
  }
  cost[0] += diagonal[7]! ** 2 * 105;
  cost[4] += diagonal[15 + 7]! ** 2 * 105;
  for (let direction = 0; direction < 4; direction++) {
    const output = direction * 2 + 1;
    const alternateOffset = direction * 11;
    for (let index = 0; index < 5; index++) cost[output] += alternate[alternateOffset + 3 + index]! ** 2;
    cost[output] *= 105;
    for (let index = 0; index < 3; index++) {
      cost[output] += (alternate[alternateOffset + index]! ** 2 +
        alternate[alternateOffset + 10 - index]! ** 2) * divisors[index * 2 + 1]!;
    }
  }
  let direction = 0;
  for (let index = 1; index < 8; index++) if (cost[index]! > cost[direction]!) direction = index;
  scratch.direction = direction;
  scratch.variance = Math.floor((cost[direction]! - cost[direction ^ 4]!) / 1024);
}

function adjustStrength(strength: number, variance: number): number {
  if (!variance) return 0;
  const scaled = Math.floor(variance / 64);
  const adjustment = scaled ? Math.min(Math.floor(Math.log2(scaled)), 12) : 0;
  return (strength * (4 + adjustment) + 8) >> 4;
}

function filterBlock(
  source: Plane, destination: Plane, x0: number, y0: number, nominalWidth: number, nominalHeight: number,
  primaryStrength: number, secondaryStrength: number, direction: number, damping: number, bitDepthShift: number,
): void {
  const width = Math.min(nominalWidth, destination.width - x0);
  const height = Math.min(nominalHeight, destination.height - y0);
  if (width <= 0 || height <= 0) return;
  const primaryShift = primaryStrength ? Math.max(0, damping - floorLog2(primaryStrength)) : 0;
  const secondaryShift = secondaryStrength ? Math.max(0, damping - floorLog2(secondaryStrength)) : 0;
  const primaryTap0 = 4 - ((primaryStrength >> bitDepthShift) & 1);
  // Every CDEF displacement is at most two samples. Most 8x8 blocks have that
  // margin, so their inner loop can use precomputed linear offsets and avoid
  // twelve bounds-checked helper calls per pixel.
  if (x0 >= 2 && y0 >= 2 && x0 + width + 1 < source.width && y0 + height + 1 < source.height) {
    filterInteriorBlock(source, destination, x0, y0, width, height,
      primaryStrength, secondaryStrength, direction, primaryShift, secondaryShift, primaryTap0);
    return;
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const center = source.data[(y0 + y) * source.stride + x0 + x]!;
      let sum = 0, minimum = center, maximum = center;
      if (primaryStrength) {
        let tap = primaryTap0;
        for (let pass = 0; pass < 2; pass++) {
          const [dx, dy] = DIRECTIONS[direction + 2]![pass]!;
          const negative = cdefSample(source, x0 + x - dx, y0 + y - dy);
          if (negative !== null) {
            sum += tap * constrain(negative - center, primaryStrength, primaryShift);
            minimum = Math.min(minimum, negative);
            maximum = Math.max(maximum, negative);
          }
          const positive = cdefSample(source, x0 + x + dx, y0 + y + dy);
          if (positive !== null) {
            sum += tap * constrain(positive - center, primaryStrength, primaryShift);
            minimum = Math.min(minimum, positive);
            maximum = Math.max(maximum, positive);
          }
          tap = (tap & 3) | 2;
        }
      }
      if (secondaryStrength) {
        for (let pass = 0; pass < 2; pass++) {
          const tap = 2 - pass;
          for (let directionOffset = 0; directionOffset <= 4; directionOffset += 4) {
            const [dx, dy] = DIRECTIONS[direction + directionOffset]![pass]!;
            const negative = cdefSample(source, x0 + x - dx, y0 + y - dy);
            if (negative !== null) {
              sum += tap * constrain(negative - center, secondaryStrength, secondaryShift);
              minimum = Math.min(minimum, negative);
              maximum = Math.max(maximum, negative);
            }
            const positive = cdefSample(source, x0 + x + dx, y0 + y + dy);
            if (positive !== null) {
              sum += tap * constrain(positive - center, secondaryStrength, secondaryShift);
              minimum = Math.min(minimum, positive);
              maximum = Math.max(maximum, positive);
            }
          }
        }
      }
      let value = center + ((sum - +(sum < 0) + 8) >> 4);
      if (primaryStrength && secondaryStrength) value = Math.max(minimum, Math.min(maximum, value));
      destination.data[(y0 + y) * destination.stride + x0 + x] = value;
    }
  }
}

function filterInteriorBlock(
  source: Plane, destination: Plane, x0: number, y0: number, width: number, height: number,
  primaryStrength: number, secondaryStrength: number, direction: number,
  primaryShift: number, secondaryShift: number, primaryTap0: number,
): void {
  const sourceData = source.data, destinationData = destination.data;
  const sourceStride = source.stride, destinationStride = destination.stride;
  const primary = DIRECTIONS[direction + 2]!;
  const primaryOffset0 = primary[0]![1] * sourceStride + primary[0]![0];
  const primaryOffset1 = primary[1]![1] * sourceStride + primary[1]![0];
  const primaryTap1 = (primaryTap0 & 3) | 2;
  const secondary0 = DIRECTIONS[direction]!;
  const secondary1 = DIRECTIONS[direction + 4]!;
  const secondaryOffset00 = secondary0[0]![1] * sourceStride + secondary0[0]![0];
  const secondaryOffset01 = secondary1[0]![1] * sourceStride + secondary1[0]![0];
  const secondaryOffset10 = secondary0[1]![1] * sourceStride + secondary0[1]![0];
  const secondaryOffset11 = secondary1[1]![1] * sourceStride + secondary1[1]![0];
  if (!secondaryStrength) {
    filterInteriorPrimary(
      sourceData, destinationData, sourceStride, destinationStride, x0, y0, width, height,
      primaryStrength, primaryShift, primaryTap0, primaryTap1, primaryOffset0, primaryOffset1,
    );
    return;
  }
  if (!primaryStrength) {
    filterInteriorSecondary(
      sourceData, destinationData, sourceStride, destinationStride, x0, y0, width, height,
      secondaryStrength, secondaryShift,
      secondaryOffset00, secondaryOffset01, secondaryOffset10, secondaryOffset11,
    );
    return;
  }

  for (let y = 0; y < height; y++) {
    let sourceIndex = (y0 + y) * sourceStride + x0;
    let destinationIndex = (y0 + y) * destinationStride + x0;
    for (let x = 0; x < width; x++, sourceIndex++, destinationIndex++) {
      const center = sourceData[sourceIndex]!;
      let sum = 0, minimum = center, maximum = center;
      let negative: number, positive: number;
      negative = sourceData[sourceIndex - primaryOffset0]!;
      positive = sourceData[sourceIndex + primaryOffset0]!;
      sum += primaryTap0 * constrain(negative - center, primaryStrength, primaryShift);
      sum += primaryTap0 * constrain(positive - center, primaryStrength, primaryShift);
      minimum = Math.min(minimum, negative, positive);
      maximum = Math.max(maximum, negative, positive);

      negative = sourceData[sourceIndex - primaryOffset1]!;
      positive = sourceData[sourceIndex + primaryOffset1]!;
      sum += primaryTap1 * constrain(negative - center, primaryStrength, primaryShift);
      sum += primaryTap1 * constrain(positive - center, primaryStrength, primaryShift);
      minimum = Math.min(minimum, negative, positive);
      maximum = Math.max(maximum, negative, positive);

      negative = sourceData[sourceIndex - secondaryOffset00]!;
      positive = sourceData[sourceIndex + secondaryOffset00]!;
      sum += 2 * constrain(negative - center, secondaryStrength, secondaryShift);
      sum += 2 * constrain(positive - center, secondaryStrength, secondaryShift);
      minimum = Math.min(minimum, negative, positive);
      maximum = Math.max(maximum, negative, positive);

      negative = sourceData[sourceIndex - secondaryOffset01]!;
      positive = sourceData[sourceIndex + secondaryOffset01]!;
      sum += 2 * constrain(negative - center, secondaryStrength, secondaryShift);
      sum += 2 * constrain(positive - center, secondaryStrength, secondaryShift);
      minimum = Math.min(minimum, negative, positive);
      maximum = Math.max(maximum, negative, positive);

      negative = sourceData[sourceIndex - secondaryOffset10]!;
      positive = sourceData[sourceIndex + secondaryOffset10]!;
      sum += constrain(negative - center, secondaryStrength, secondaryShift);
      sum += constrain(positive - center, secondaryStrength, secondaryShift);
      minimum = Math.min(minimum, negative, positive);
      maximum = Math.max(maximum, negative, positive);

      negative = sourceData[sourceIndex - secondaryOffset11]!;
      positive = sourceData[sourceIndex + secondaryOffset11]!;
      sum += constrain(negative - center, secondaryStrength, secondaryShift);
      sum += constrain(positive - center, secondaryStrength, secondaryShift);
      minimum = Math.min(minimum, negative, positive);
      maximum = Math.max(maximum, negative, positive);
      let value = center + ((sum - +(sum < 0) + 8) >> 4);
      value = Math.max(minimum, Math.min(maximum, value));
      destinationData[destinationIndex] = value;
    }
  }
}

function filterInteriorPrimary(
  source: Plane['data'], destination: Plane['data'], sourceStride: number, destinationStride: number,
  x0: number, y0: number, width: number, height: number,
  strength: number, shift: number, tap0: number, tap1: number, offset0: number, offset1: number,
): void {
  for (let y = 0; y < height; y++) {
    let sourceIndex = (y0 + y) * sourceStride + x0;
    let destinationIndex = (y0 + y) * destinationStride + x0;
    for (let x = 0; x < width; x++, sourceIndex++, destinationIndex++) {
      const center = source[sourceIndex]!;
      let sum = 0;
      sum += tap0 * constrain(source[sourceIndex - offset0]! - center, strength, shift);
      sum += tap0 * constrain(source[sourceIndex + offset0]! - center, strength, shift);
      sum += tap1 * constrain(source[sourceIndex - offset1]! - center, strength, shift);
      sum += tap1 * constrain(source[sourceIndex + offset1]! - center, strength, shift);
      destination[destinationIndex] = center + ((sum - +(sum < 0) + 8) >> 4);
    }
  }
}

function filterInteriorSecondary(
  source: Plane['data'], destination: Plane['data'], sourceStride: number, destinationStride: number,
  x0: number, y0: number, width: number, height: number, strength: number, shift: number,
  offset00: number, offset01: number, offset10: number, offset11: number,
): void {
  for (let y = 0; y < height; y++) {
    let sourceIndex = (y0 + y) * sourceStride + x0;
    let destinationIndex = (y0 + y) * destinationStride + x0;
    for (let x = 0; x < width; x++, sourceIndex++, destinationIndex++) {
      const center = source[sourceIndex]!;
      let sum = 0;
      sum += 2 * constrain(source[sourceIndex - offset00]! - center, strength, shift);
      sum += 2 * constrain(source[sourceIndex + offset00]! - center, strength, shift);
      sum += 2 * constrain(source[sourceIndex - offset01]! - center, strength, shift);
      sum += 2 * constrain(source[sourceIndex + offset01]! - center, strength, shift);
      sum += constrain(source[sourceIndex - offset10]! - center, strength, shift);
      sum += constrain(source[sourceIndex + offset10]! - center, strength, shift);
      sum += constrain(source[sourceIndex - offset11]! - center, strength, shift);
      sum += constrain(source[sourceIndex + offset11]! - center, strength, shift);
      destination[destinationIndex] = center + ((sum - +(sum < 0) + 8) >> 4);
    }
  }
}

function cdefSample(plane: Plane, x: number, y: number): number | null {
  return x < 0 || y < 0 || x >= plane.width || y >= plane.height ? null :
    plane.data[y * plane.stride + x]!;
}

function constrain(difference: number, threshold: number, shift: number): number {
  const magnitude = Math.abs(difference);
  const constrained = Math.min(magnitude, Math.max(0, threshold - (magnitude >> shift)));
  return difference < 0 ? -constrained : constrained;
}

function floorLog2(value: number): number {
  return Math.floor(Math.log2(value));
}
