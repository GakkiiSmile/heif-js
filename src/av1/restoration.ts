import type { DecodedFrame, Plane } from '../frame.ts';
import type { Av1RestorationUnit } from './decode.ts';
import type { Av1SequenceHeader } from './obu.ts';
import { sgrXByX } from './sgr_data.ts';

const SGR_PARAMETERS = [
  [140, 3236], [112, 2158], [93, 1618], [80, 1438], [70, 1295], [58, 1177],
  [47, 1079], [37, 996], [30, 925], [25, 863], [0, 2589], [0, 1618],
  [0, 1177], [0, 925], [56, 0], [22, 0],
] as const;

interface RestorationScratch {
  wiener: Uint16Array;
  horizontalFilter: Int16Array;
  verticalFilter: Int16Array;
  sgrA5: Int32Array;
  sgrB5: Uint8Array;
  sgrA3: Int32Array;
  sgrB3: Uint8Array;
  rollingHorizontal: Int32Array;
  rollingHorizontalSquares: Int32Array;
  rollingVertical: Int32Array;
  rollingVerticalSquares: Int32Array;
}

function createScratch(): RestorationScratch {
  return {
    wiener: new Uint16Array(0),
    horizontalFilter: new Int16Array(7),
    verticalFilter: new Int16Array(7),
    sgrA5: new Int32Array(0), sgrB5: new Uint8Array(0),
    sgrA3: new Int32Array(0), sgrB3: new Uint8Array(0),
    rollingHorizontal: new Int32Array(0), rollingHorizontalSquares: new Int32Array(0),
    rollingVertical: new Int32Array(0), rollingVerticalSquares: new Int32Array(0),
  };
}

function ensureInt32(buffer: Int32Array, length: number): Int32Array {
  return buffer.length >= length ? buffer : new Int32Array(length);
}

function ensureUint16(buffer: Uint16Array, length: number): Uint16Array {
  return buffer.length >= length ? buffer : new Uint16Array(length);
}

function ensureUint8(buffer: Uint8Array, length: number): Uint8Array {
  return buffer.length >= length ? buffer : new Uint8Array(length);
}

/** Apply decoded Wiener and self-guided loop-restoration units. */
export function applyRestoration(
  frame: DecodedFrame, units: readonly Av1RestorationUnit[], sequence: Av1SequenceHeader,
  deblockedFrame: DecodedFrame = frame,
): void {
  if (!units.some(unit => unit.type)) return;
  const activePlanes = frame.planes.map((_, plane) => units.some(unit => unit.plane === plane && unit.type));
  const sources = frame.planes.map((plane, index) => activePlanes[index] ? copyPlane(plane) : plane);
  // A distinct deblocked frame is already an immutable snapshot. Copying it
  // again doubled restoration's full-frame working set without protecting any
  // data that this function writes.
  const deblockedSources = deblockedFrame === frame ? sources : deblockedFrame.planes;
  const scratch = createScratch();
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
        applyWiener(source, deblocked, destination, stripeUnit, sequence.bitDepth, stripeStart, stripeEnd, scratch);
      } else if (unit.type === 3) {
        applySelfGuided(source, deblocked, destination, stripeUnit,
          sequence.bitDepth, stripeStart, stripeEnd, scratch);
      }
      stripeY = end;
    }
  }
}

function applyWiener(
  source: Plane, deblocked: Plane, destination: Plane,
  unit: Av1RestorationUnit, bitDepth: number, stripeStart: number, stripeEnd: number,
  scratch: RestorationScratch,
): void {
  const horizontal = fullFilter(unit.filterHorizontal, scratch.horizontalFilter);
  const vertical = fullFilter(unit.filterVertical, scratch.verticalFilter);
  const roundBitsHorizontal = 3 + +(bitDepth === 12) * 2;
  const roundBitsVertical = 11 - +(bitDepth === 12) * 2;
  const horizontalOffset = 1 << (bitDepth + 6);
  const horizontalRound = 1 << (roundBitsHorizontal - 1);
  const clipLimit = 1 << (bitDepth + 8 - roundBitsHorizontal);
  const verticalOffset = 1 << (bitDepth + roundBitsVertical - 1);
  const verticalRound = 1 << (roundBitsVertical - 1);
  const temporaryLength = (unit.height + 6) * unit.width;
  // The normative horizontal clip limit is at most 32768 for every supported
  // bit depth, so the intermediate is unsigned 16-bit data, not Int32.
  scratch.wiener = ensureUint16(scratch.wiener, temporaryLength);
  const temporary = scratch.wiener;
  const h0 = horizontal[0]!, h1 = horizontal[1]!, h2 = horizontal[2]!, h3 = horizontal[3]!;
  const h4 = horizontal[4]!, h5 = horizontal[5]!, h6 = horizontal[6]!;
  const interiorStart = Math.min(unit.width, Math.max(0, 3 - unit.x));
  const interiorEnd = Math.max(interiorStart, Math.min(unit.width, source.width - 3 - unit.x));

  for (let row = -3; row < unit.height + 3; row++) {
    const temporaryRow = (row + 3) * unit.width;
    // All seven horizontal taps read the same source row. Resolve frame and
    // restoration-stripe boundaries once per row instead of once per tap.
    let sampleY = clamp(unit.y + row, 0, source.height - 1);
    let sampleData = source.data;
    let sampleStride = source.stride;
    if (sampleY < stripeStart) {
      sampleY = Math.max(stripeStart - 2, sampleY);
      sampleData = deblocked.data;
      sampleStride = deblocked.stride;
    } else if (sampleY > stripeEnd) {
      sampleY = Math.min(stripeEnd + 2, sampleY);
      sampleData = deblocked.data;
      sampleStride = deblocked.stride;
    }
    const sampleRow = sampleY * sampleStride;
    let x = 0;
    for (; x < interiorStart; x++) {
      const center = unit.x + x;
      let sum = horizontalOffset;
      for (let tap = 0; tap < 7; tap++) {
        const sampleX = clamp(center + tap - 3, 0, source.width - 1);
        sum += sampleData[sampleRow + sampleX]! * horizontal[tap]!;
      }
      temporary[temporaryRow + x] = clamp((sum + horizontalRound) >> roundBitsHorizontal, 0, clipLimit - 1);
    }
    let sourceIndex = sampleRow + unit.x + x;
    for (; x < interiorEnd; x++, sourceIndex++) {
      let sum = horizontalOffset;
      sum += sampleData[sourceIndex - 3]! * h0;
      sum += sampleData[sourceIndex - 2]! * h1;
      sum += sampleData[sourceIndex - 1]! * h2;
      sum += sampleData[sourceIndex]! * h3;
      sum += sampleData[sourceIndex + 1]! * h4;
      sum += sampleData[sourceIndex + 2]! * h5;
      sum += sampleData[sourceIndex + 3]! * h6;
      const value = (sum + horizontalRound) >> roundBitsHorizontal;
      temporary[temporaryRow + x] = value < 0 ? 0 : value >= clipLimit ? clipLimit - 1 : value;
    }
    for (; x < unit.width; x++) {
      const center = unit.x + x;
      let sum = horizontalOffset;
      for (let tap = 0; tap < 7; tap++) {
        const sampleX = clamp(center + tap - 3, 0, source.width - 1);
        sum += sampleData[sampleRow + sampleX]! * horizontal[tap]!;
      }
      temporary[temporaryRow + x] = clamp((sum + horizontalRound) >> roundBitsHorizontal, 0, clipLimit - 1);
    }
  }

  const maximum = (1 << bitDepth) - 1;
  const v0 = vertical[0]!, v1 = vertical[1]!, v2 = vertical[2]!, v3 = vertical[3]!;
  const v4 = vertical[4]!, v5 = vertical[5]!, v6 = vertical[6]!;
  for (let y = 0; y < unit.height; y++) {
    let index = y * unit.width;
    let destinationIndex = (unit.y + y) * destination.stride + unit.x;
    for (let x = 0; x < unit.width; x++) {
      let sum = -verticalOffset;
      sum += temporary[index]! * v0;
      sum += temporary[index + unit.width]! * v1;
      sum += temporary[index + 2 * unit.width]! * v2;
      sum += temporary[index + 3 * unit.width]! * v3;
      sum += temporary[index + 4 * unit.width]! * v4;
      sum += temporary[index + 5 * unit.width]! * v5;
      sum += temporary[index + 6 * unit.width]! * v6;
      const value = (sum + verticalRound) >> roundBitsVertical;
      destination.data[destinationIndex] = value < 0 ? 0 : value > maximum ? maximum : value;
      index++;
      destinationIndex++;
    }
  }
}

function fullFilter(outer: readonly number[], filter: Int16Array): Int16Array {
  filter[0] = filter[6] = outer[0]!;
  filter[1] = filter[5] = outer[1]!;
  filter[2] = filter[4] = outer[2]!;
  filter[3] = 128 - 2 * (outer[0]! + outer[1]! + outer[2]!);
  return filter;
}

function applySelfGuided(
  source: Plane, deblocked: Plane, destination: Plane,
  unit: Av1RestorationUnit, bitDepth: number, stripeStart: number, stripeEnd: number,
  scratch: RestorationScratch,
): void {
  const [strength5, strength3] = SGR_PARAMETERS[unit.sgrIndex]!;
  const coefficientLength = (unit.width + 2) * (unit.height + 2);
  let radius5: GuidedCoefficients | null = null;
  let radius3: GuidedCoefficients | null = null;
  if (strength5) {
    scratch.sgrA5 = ensureInt32(scratch.sgrA5, coefficientLength);
    scratch.sgrB5 = ensureUint8(scratch.sgrB5, coefficientLength);
    radius5 = buildGuidedCoefficients(
      source, deblocked, unit, 2, strength5, bitDepth, stripeStart, stripeEnd,
      scratch.sgrA5, scratch.sgrB5, scratch,
    );
  }
  if (strength3) {
    scratch.sgrA3 = ensureInt32(scratch.sgrA3, coefficientLength);
    scratch.sgrB3 = ensureUint8(scratch.sgrB3, coefficientLength);
    radius3 = buildGuidedCoefficients(
      source, deblocked, unit, 1, strength3, bitDepth, stripeStart, stripeEnd,
      scratch.sgrA3, scratch.sgrB3, scratch,
    );
  }
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
  b: Uint8Array;
  stride: number;
  radius: number;
}

function buildGuidedCoefficients(
  source: Plane, deblocked: Plane, unit: Av1RestorationUnit,
  radius: number, strength: number, bitDepth: number,
  stripeStart: number, stripeEnd: number,
  a: Int32Array, b: Uint8Array, scratch: RestorationScratch,
): GuidedCoefficients {
  // One coefficient border is needed by the final weighted-neighbour stage.
  const stride = unit.width + 2;
  const rows = unit.height + 2;
  const window = radius * 2 + 1;
  const count = window * window;
  const reciprocal = radius === 1 ? 455 : 164;
  const bitDepthShift = bitDepth - 8;
  const squareRound = (1 << (2 * bitDepthShift)) >> 1;
  const sumRound = (1 << bitDepthShift) >> 1;
  const ringLength = stride * window;
  scratch.rollingHorizontal = ensureInt32(scratch.rollingHorizontal, ringLength);
  scratch.rollingHorizontalSquares = ensureInt32(scratch.rollingHorizontalSquares, ringLength);
  scratch.rollingVertical = ensureInt32(scratch.rollingVertical, stride);
  scratch.rollingVerticalSquares = ensureInt32(scratch.rollingVerticalSquares, stride);
  const horizontal = scratch.rollingHorizontal;
  const horizontalSquares = scratch.rollingHorizontalSquares;
  const vertical = scratch.rollingVertical;
  const verticalSquares = scratch.rollingVerticalSquares;
  vertical.fill(0, 0, stride);
  verticalSquares.fill(0, 0, stride);

  // Form horizontal window sums once per source row, then maintain vertical
  // rolling sums over exactly `window` rows. This is bit-equivalent to the
  // direct square loop because all intermediates are bounded exact integers.
  const sourceRows = rows + 2 * radius;
  for (let sourceRow = 0; sourceRow < sourceRows; sourceRow++) {
    const slot = sourceRow % window;
    const ringOffset = slot * stride;
    if (sourceRow >= window) {
      for (let column = 0; column < stride; column++) {
        vertical[column] -= horizontal[ringOffset + column]!;
        verticalSquares[column] -= horizontalSquares[ringOffset + column]!;
      }
    }

    let sampleY = clamp(unit.y - 1 - radius + sourceRow, 0, source.height - 1);
    let sampleData = source.data;
    let sampleStride = source.stride;
    if (sampleY < stripeStart) {
      sampleY = Math.max(stripeStart - 2, sampleY);
      sampleData = deblocked.data;
      sampleStride = deblocked.stride;
    } else if (sampleY > stripeEnd) {
      sampleY = Math.min(stripeEnd + 2, sampleY);
      sampleData = deblocked.data;
      sampleStride = deblocked.stride;
    }
    const sampleRow = sampleY * sampleStride;
    let sum = 0, sumSquares = 0;
    for (let wx = -1 - radius; wx <= -1 + radius; wx++) {
      const sampleX = clamp(unit.x + wx, 0, source.width - 1);
      const value = sampleData[sampleRow + sampleX]!;
      sum += value;
      sumSquares += value * value;
    }
    horizontal[ringOffset] = sum;
    horizontalSquares[ringOffset] = sumSquares;
    vertical[0] += sum;
    verticalSquares[0] += sumSquares;
    for (let column = 1; column < stride; column++) {
      const removedX = clamp(unit.x + column - 2 - radius, 0, source.width - 1);
      const addedX = clamp(unit.x + column - 1 + radius, 0, source.width - 1);
      const removed = sampleData[sampleRow + removedX]!;
      const added = sampleData[sampleRow + addedX]!;
      sum += added - removed;
      sumSquares += added * added - removed * removed;
      horizontal[ringOffset + column] = sum;
      horizontalSquares[ringOffset + column] = sumSquares;
      vertical[column] += sum;
      verticalSquares[column] += sumSquares;
    }

    if (sourceRow < window - 1) continue;
    const outputRow = sourceRow - window + 1;
    for (let column = 0; column < stride; column++) {
      sum = vertical[column]!;
      sumSquares = verticalSquares[column]!;
      const normalizedSquares = Math.floor((sumSquares + squareRound) / 2 ** (2 * bitDepthShift));
      const normalizedSum = Math.floor((sum + sumRound) / 2 ** bitDepthShift);
      const variance = Math.max(normalizedSquares * count - normalizedSum * normalizedSum, 0);
      const z = Math.min(255, Math.floor((variance * strength + (1 << 19)) / 2 ** 20));
      const factor = sgrXByX[z]!;
      const index = outputRow * stride + column;
      a[index] = Math.floor((factor * sum * reciprocal + (1 << 11)) / 2 ** 12);
      b[index] = factor;
    }
  }
  return { a, b, stride, radius };
}

function guidedResidual3(
  coefficients: GuidedCoefficients, x: number, y: number, width: number, source: number,
): number {
  const stride = coefficients.stride;
  const index = (y + 1) * stride + x + 1;
  const a = coefficients.a, b = coefficients.b;
  const factor = (b[index]! + b[index - 1]! + b[index + 1]! +
    b[index - stride]! + b[index + stride]!) * 4 +
    (b[index - stride - 1]! + b[index - stride + 1]! +
      b[index + stride - 1]! + b[index + stride + 1]!) * 3;
  const offset = (a[index]! + a[index - 1]! + a[index + 1]! +
    a[index - stride]! + a[index + stride]!) * 4 +
    (a[index - stride - 1]! + a[index - stride + 1]! +
      a[index + stride - 1]! + a[index + stride + 1]!) * 3;
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
  const a = coefficients.a, b = coefficients.b;
  const offset = (a[upper]! + a[lower]!) * 6 +
    (a[upper - 1]! + a[upper + 1]! + a[lower - 1]! + a[lower + 1]!) * 5;
  const factor = (b[upper]! + b[lower]!) * 6 +
    (b[upper - 1]! + b[upper + 1]! + b[lower - 1]! + b[lower + 1]!) * 5;
  void width;
  return (offset - factor * source + 256) >> 9;
}

function copyPlane(plane: Plane): Plane {
  return { width: plane.width, height: plane.height, stride: plane.stride, data: plane.data.slice() };
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
