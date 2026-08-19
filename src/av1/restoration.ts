import type { DecodedFrame, Plane } from '../frame.ts';
import type { Av1RestorationUnit } from './decode.ts';
import type { Av1SequenceHeader } from './obu.ts';
import { sgrXByX } from './sgr_data.ts';

const SGR_PARAMETERS = [
  [140, 3236], [112, 2158], [93, 1618], [80, 1438], [70, 1295], [58, 1177],
  [47, 1079], [37, 996], [30, 925], [25, 863], [0, 2589], [0, 1618],
  [0, 1177], [0, 925], [56, 0], [22, 0],
] as const;

/** Apply decoded Wiener and self-guided loop-restoration units. */
export function applyRestoration(
  frame: DecodedFrame, units: readonly Av1RestorationUnit[], sequence: Av1SequenceHeader,
  deblockedFrame: DecodedFrame = frame,
): void {
  if (!units.some(unit => unit.type)) return;
  const sources = frame.planes.map(copyPlane);
  const deblockedSources = deblockedFrame.planes.map(copyPlane);
  for (const unit of units) {
    if (!unit.type) continue;
    const source = sources[unit.plane]!, deblocked = deblockedSources[unit.plane]!;
    const destination = frame.planes[unit.plane]!;
    const subsamplingY = unit.plane ? sequence.subsamplingY : 0;
    const halfUnit = unit.unitSize >> 1;
    const unitColumns = Math.max(1, (destination.width + halfUnit) >> Math.log2(unit.unitSize));
    const unitRows = Math.max(1, (destination.height + halfUnit) >> Math.log2(unit.unitSize));
    const stripeOffset = 8 >> subsamplingY;
    const x = unit.unitX * unit.unitSize;
    const y = unit.unitY ? unit.unitY * unit.unitSize - stripeOffset : 0;
    const unitEndX = unit.unitX + 1 === unitColumns ? destination.width :
      Math.min(destination.width, x + unit.unitSize);
    const unitEndY = unit.unitY + 1 === unitRows ? destination.height :
      Math.min(destination.height, (unit.unitY + 1) * unit.unitSize - stripeOffset);
    if (x >= unitEndX || y >= unitEndY) continue;
    for (let stripeY = y; stripeY < unitEndY;) {
      const stripeNumber = Math.floor(((stripeY << subsamplingY) + 8) / 64);
      const stripeStart = (-8 + stripeNumber * 64) >> subsamplingY;
      const stripeEnd = stripeStart + (64 >> subsamplingY) - 1;
      const end = Math.min(unitEndY, stripeEnd + 1);
      const stripeUnit = { ...unit, x, y: stripeY, width: unitEndX - x, height: end - stripeY };
      if (unit.type === 2) {
        applyWiener(source, deblocked, destination, stripeUnit, sequence.bitDepth, stripeStart, stripeEnd);
      } else if (unit.type === 3) {
        applySelfGuided(source, deblocked, destination, stripeUnit,
          sequence.bitDepth, stripeStart, stripeEnd);
      }
      stripeY = end;
    }
  }
}

function applyWiener(
  source: Plane, deblocked: Plane, destination: Plane,
  unit: Av1RestorationUnit, bitDepth: number, stripeStart: number, stripeEnd: number,
): void {
  const horizontal = fullFilter(unit.filterHorizontal);
  const vertical = fullFilter(unit.filterVertical);
  const roundBitsHorizontal = 3 + +(bitDepth === 12) * 2;
  const roundBitsVertical = 11 - +(bitDepth === 12) * 2;
  const horizontalOffset = 1 << (bitDepth + 6);
  const horizontalRound = 1 << (roundBitsHorizontal - 1);
  const clipLimit = 1 << (bitDepth + 8 - roundBitsHorizontal);
  const verticalOffset = 1 << (bitDepth + roundBitsVertical - 1);
  const verticalRound = 1 << (roundBitsVertical - 1);
  const temporary = new Int32Array((unit.height + 6) * unit.width);

  for (let row = -3; row < unit.height + 3; row++) {
    const temporaryRow = (row + 3) * unit.width;
    for (let x = 0; x < unit.width; x++) {
      let sum = horizontalOffset;
      for (let tap = 0; tap < 7; tap++) {
        sum += sourceSample(source, deblocked, unit.x + x + tap - 3, unit.y + row,
          stripeStart, stripeEnd) * horizontal[tap]!;
      }
      temporary[temporaryRow + x] = clamp((sum + horizontalRound) >> roundBitsHorizontal, 0, clipLimit - 1);
    }
  }

  const maximum = (1 << bitDepth) - 1;
  for (let y = 0; y < unit.height; y++) {
    for (let x = 0; x < unit.width; x++) {
      let sum = -verticalOffset;
      for (let tap = 0; tap < 7; tap++) {
        sum += temporary[(y + tap) * unit.width + x]! * vertical[tap]!;
      }
      destination.data[(unit.y + y) * destination.stride + unit.x + x] =
        clamp((sum + verticalRound) >> roundBitsVertical, 0, maximum);
    }
  }
}

function fullFilter(outer: readonly number[]): Int16Array {
  const filter = new Int16Array(7);
  filter[0] = filter[6] = outer[0]!;
  filter[1] = filter[5] = outer[1]!;
  filter[2] = filter[4] = outer[2]!;
  filter[3] = 128 - 2 * (outer[0]! + outer[1]! + outer[2]!);
  return filter;
}

function applySelfGuided(
  source: Plane, deblocked: Plane, destination: Plane,
  unit: Av1RestorationUnit, bitDepth: number, stripeStart: number, stripeEnd: number,
): void {
  const [strength5, strength3] = SGR_PARAMETERS[unit.sgrIndex]!;
  const radius5 = strength5 ? buildGuidedCoefficients(
    source, deblocked, unit, 2, strength5, bitDepth, stripeStart, stripeEnd,
  ) : null;
  const radius3 = strength3 ? buildGuidedCoefficients(
    source, deblocked, unit, 1, strength3, bitDepth, stripeStart, stripeEnd,
  ) : null;
  const weight0 = unit.sgrWeights[0];
  const weight1 = 128 - unit.sgrWeights[0] - unit.sgrWeights[1];
  const maximum = (1 << bitDepth) - 1;

  for (let y = 0; y < unit.height; y++) {
    for (let x = 0; x < unit.width; x++) {
      const sourcePixel = source.data[(unit.y + y) * source.stride + unit.x + x]!;
      const residual5 = radius5 ? guidedResidual5(radius5, x, y, unit.width, sourcePixel) : 0;
      const residual3 = radius3 ? guidedResidual3(radius3, x, y, unit.width, sourcePixel) : 0;
      const correction = (weight0 * residual5 + weight1 * residual3 + 1024) >> 11;
      destination.data[(unit.y + y) * destination.stride + unit.x + x] =
        clamp(sourcePixel + correction, 0, maximum);
    }
  }
}

interface GuidedCoefficients {
  a: Int32Array;
  b: Int32Array;
  stride: number;
  radius: number;
}

function buildGuidedCoefficients(
  source: Plane, deblocked: Plane, unit: Av1RestorationUnit,
  radius: number, strength: number, bitDepth: number,
  stripeStart: number, stripeEnd: number,
): GuidedCoefficients {
  // One coefficient border is needed by the final weighted-neighbour stage.
  const stride = unit.width + 2;
  const rows = unit.height + 2;
  const a = new Int32Array(stride * rows);
  const b = new Int32Array(stride * rows);
  const window = radius * 2 + 1;
  const count = window * window;
  const reciprocal = radius === 1 ? 455 : 164;
  const bitDepthShift = bitDepth - 8;
  const squareRound = (1 << (2 * bitDepthShift)) >> 1;
  const sumRound = (1 << bitDepthShift) >> 1;
  for (let y = -1; y <= unit.height; y++) {
    for (let x = -1; x <= unit.width; x++) {
      let sum = 0, sumSquares = 0;
      for (let wy = -radius; wy <= radius; wy++) {
        for (let wx = -radius; wx <= radius; wx++) {
          const value = sourceSample(source, deblocked, unit.x + x + wx, unit.y + y + wy,
            stripeStart, stripeEnd);
          sum += value;
          sumSquares += value * value;
        }
      }
      const normalizedSquares = Math.floor((sumSquares + squareRound) / 2 ** (2 * bitDepthShift));
      const normalizedSum = Math.floor((sum + sumRound) / 2 ** bitDepthShift);
      const variance = Math.max(normalizedSquares * count - normalizedSum * normalizedSum, 0);
      const z = Math.min(255, Math.floor((variance * strength + (1 << 19)) / 2 ** 20));
      const factor = sgrXByX[z]!;
      const index = (y + 1) * stride + x + 1;
      a[index] = Math.floor((factor * sum * reciprocal + (1 << 11)) / 2 ** 12);
      b[index] = factor;
    }
  }
  return { a, b, stride, radius };
}

function guidedResidual3(
  coefficients: GuidedCoefficients, x: number, y: number, width: number, source: number,
): number {
  const index = (y + 1) * coefficients.stride + x + 1;
  const weighted = (values: Int32Array): number =>
    (values[index]! + values[index - 1]! + values[index + 1]! +
      values[index - coefficients.stride]! + values[index + coefficients.stride]!) * 4 +
    (values[index - coefficients.stride - 1]! + values[index - coefficients.stride + 1]! +
      values[index + coefficients.stride - 1]! + values[index + coefficients.stride + 1]!) * 3;
  const factor = weighted(coefficients.b);
  const offset = weighted(coefficients.a);
  void width;
  return (offset - factor * source + 256) >> 9;
}

function guidedResidual5(
  coefficients: GuidedCoefficients, x: number, y: number, width: number, source: number,
): number {
  const stride = coefficients.stride;
  // The normative radius-2 implementation alternates one- and two-row
  // coefficient blends; this direct form is equivalent to those finish stages.
  if (y & 1) {
    const index = (y + 1) * stride + x + 1;
    const factor = coefficients.b[index]! * 6 +
      (coefficients.b[index - 1]! + coefficients.b[index + 1]!) * 5;
    const offset = coefficients.a[index]! * 6 +
      (coefficients.a[index - 1]! + coefficients.a[index + 1]!) * 5;
    return (offset - factor * source + 128) >> 8;
  }
  const upper = y * stride + x + 1;
  const lower = (y + 2) * stride + x + 1;
  const blend = (values: Int32Array): number =>
    (values[upper]! + values[lower]!) * 6 +
    (values[upper - 1]! + values[upper + 1]! + values[lower - 1]! + values[lower + 1]!) * 5;
  void width;
  return (blend(coefficients.a) - blend(coefficients.b) * source + 256) >> 9;
}

function sourceSample(
  cdef: Plane, deblocked: Plane, x: number, y: number,
  stripeStart: number, stripeEnd: number,
): number {
  x = clamp(x, 0, cdef.width - 1);
  y = clamp(y, 0, cdef.height - 1);
  if (y < stripeStart) {
    y = Math.max(stripeStart - 2, y);
    return deblocked.data[y * deblocked.stride + x]!;
  }
  if (y > stripeEnd) {
    y = Math.min(stripeEnd + 2, y);
    return deblocked.data[y * deblocked.stride + x]!;
  }
  return cdef.data[y * cdef.stride + x]!;
}

function copyPlane(plane: Plane): Plane {
  return { width: plane.width, height: plane.height, stride: plane.stride, data: new Uint16Array(plane.data) };
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
