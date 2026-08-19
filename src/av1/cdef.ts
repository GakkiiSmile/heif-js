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

/** Apply AV1's constrained directional enhancement filter to a reconstructed frame. */
export function applyCdef(
  frame: DecodedFrame, blocks: readonly Av1DecodedBlock[],
  sequence: Av1SequenceHeader, header: Av1FrameHeader,
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

  const sources = frame.planes.map(copyPlane);
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
      const { direction, variance } = findDirection(sources[0]!, x, y, bitDepthShift);
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
  return { width: plane.width, height: plane.height, stride: plane.stride, data: new Uint16Array(plane.data) };
}

function findDirection(
  plane: Plane, x0: number, y0: number, bitDepthShift: number,
): { direction: number; variance: number } {
  const horizontalVertical = [new Int32Array(8), new Int32Array(8)];
  const diagonal = [new Int32Array(15), new Int32Array(15)];
  const alternate = Array.from({ length: 4 }, () => new Int32Array(11));
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      const sampleX = Math.min(plane.width - 1, x0 + x);
      const sampleY = Math.min(plane.height - 1, y0 + y);
      const pixel = (plane.data[sampleY * plane.stride + sampleX]! >> bitDepthShift) - 128;
      diagonal[0]![y + x] += pixel;
      alternate[0]![y + (x >> 1)] += pixel;
      horizontalVertical[0]![y] += pixel;
      alternate[1]![3 + y - (x >> 1)] += pixel;
      diagonal[1]![7 + y - x] += pixel;
      alternate[2]![3 - (y >> 1) + x] += pixel;
      horizontalVertical[1]![x] += pixel;
      alternate[3]![(y >> 1) + x] += pixel;
    }
  }
  const cost = new Float64Array(8);
  for (let index = 0; index < 8; index++) {
    cost[2] += horizontalVertical[0]![index]! ** 2;
    cost[6] += horizontalVertical[1]![index]! ** 2;
  }
  cost[2] *= 105;
  cost[6] *= 105;
  const divisors = [840, 420, 280, 210, 168, 140, 120] as const;
  for (let index = 0; index < 7; index++) {
    cost[0] += (diagonal[0]![index]! ** 2 + diagonal[0]![14 - index]! ** 2) * divisors[index]!;
    cost[4] += (diagonal[1]![index]! ** 2 + diagonal[1]![14 - index]! ** 2) * divisors[index]!;
  }
  cost[0] += diagonal[0]![7]! ** 2 * 105;
  cost[4] += diagonal[1]![7]! ** 2 * 105;
  for (let direction = 0; direction < 4; direction++) {
    const output = direction * 2 + 1;
    for (let index = 0; index < 5; index++) cost[output] += alternate[direction]![3 + index]! ** 2;
    cost[output] *= 105;
    for (let index = 0; index < 3; index++) {
      cost[output] += (alternate[direction]![index]! ** 2 +
        alternate[direction]![10 - index]! ** 2) * divisors[index * 2 + 1]!;
    }
  }
  let direction = 0;
  for (let index = 1; index < 8; index++) if (cost[index]! > cost[direction]!) direction = index;
  return { direction, variance: Math.floor((cost[direction]! - cost[direction ^ 4]!) / 1024) };
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
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const center = source.data[(y0 + y) * source.stride + x0 + x]!;
      let sum = 0, minimum = center, maximum = center;
      if (primaryStrength) {
        let tap = primaryTap0;
        for (let pass = 0; pass < 2; pass++) {
          const [dx, dy] = DIRECTIONS[direction + 2]![pass]!;
          for (const sign of [-1, 1]) {
            const sample = cdefSample(source, x0 + x + sign * dx, y0 + y + sign * dy);
            if (sample !== null) {
              sum += tap * constrain(sample - center, primaryStrength, primaryShift);
              minimum = Math.min(minimum, sample);
              maximum = Math.max(maximum, sample);
            }
          }
          tap = (tap & 3) | 2;
        }
      }
      if (secondaryStrength) {
        for (let pass = 0; pass < 2; pass++) {
          const tap = 2 - pass;
          for (const directionOffset of [0, 4]) {
            const [dx, dy] = DIRECTIONS[direction + directionOffset]![pass]!;
            for (const sign of [-1, 1]) {
              const sample = cdefSample(source, x0 + x + sign * dx, y0 + y + sign * dy);
              if (sample !== null) {
                sum += tap * constrain(sample - center, secondaryStrength, secondaryShift);
                minimum = Math.min(minimum, sample);
                maximum = Math.max(maximum, sample);
              }
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
