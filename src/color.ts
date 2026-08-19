/** H.273 YCbCr/derived-matrix to nonlinear RGB conversion. */
import { CHROMA_MONO, CHROMA_420, CHROMA_422 } from './frame.ts';
import type { DecodedFrame } from './frame.ts';

type Vec3 = readonly [number, number, number];
type MutableVec3 = [number, number, number];
type Mat3 = readonly [Vec3, Vec3, Vec3];

const MATRICES: Record<number, readonly [number, number]> = {
  1: [0.2126, 0.0722], // BT.709
  2: [0.299, 0.114],   // unspecified: conventional BT.601 fallback
  4: [0.300, 0.110],   // FCC
  5: [0.299, 0.114],   // BT.470BG
  6: [0.299, 0.114],   // SMPTE 170M
  7: [0.212, 0.087],   // SMPTE 240M
  9: [0.2627, 0.0593], // BT.2020 non-constant luminance
  10: [0.2627, 0.0593], // BT.2020 constant luminance
};

interface Chromaticities {
  r: readonly [number, number];
  g: readonly [number, number];
  b: readonly [number, number];
  w: readonly [number, number];
}

const D65 = [0.3127, 0.3290] as const;
const C = [0.310, 0.316] as const;
const PRIMARIES: Record<number, Chromaticities> = {
  1: { r: [0.640, 0.330], g: [0.300, 0.600], b: [0.150, 0.060], w: D65 },
  4: { r: [0.670, 0.330], g: [0.210, 0.710], b: [0.140, 0.080], w: C },
  5: { r: [0.640, 0.330], g: [0.290, 0.600], b: [0.150, 0.060], w: D65 },
  6: { r: [0.630, 0.340], g: [0.310, 0.595], b: [0.155, 0.070], w: D65 },
  7: { r: [0.630, 0.340], g: [0.310, 0.595], b: [0.155, 0.070], w: D65 },
  8: { r: [0.681, 0.319], g: [0.243, 0.692], b: [0.145, 0.049], w: C },
  9: { r: [0.708, 0.292], g: [0.170, 0.797], b: [0.131, 0.046], w: D65 },
  11: { r: [0.680, 0.320], g: [0.265, 0.690], b: [0.150, 0.060], w: [0.314, 0.351] },
  12: { r: [0.680, 0.320], g: [0.265, 0.690], b: [0.150, 0.060], w: D65 },
  22: { r: [0.630, 0.340], g: [0.295, 0.605], b: [0.155, 0.077], w: D65 },
};

const RGB_TO_LMS_ICTCP: Mat3 = [
  [1688 / 4096, 2146 / 4096, 262 / 4096],
  [683 / 4096, 2951 / 4096, 462 / 4096],
  [99 / 4096, 309 / 4096, 3688 / 4096],
];
const RGB_TO_LMS_IPT: Mat3 = [
  [1747 / 4096, 2169 / 4096, 180 / 4096],
  [673 / 4096, 3029 / 4096, 394 / 4096],
  [50 / 4096, 207 / 4096, 3839 / 4096],
];
const LMS_TO_ICTCP_PQ: Mat3 = [
  [0.5, 0.5, 0],
  [6610 / 4096, -13613 / 4096, 7003 / 4096],
  [17933 / 4096, -17390 / 4096, -543 / 4096],
];
const LMS_TO_ICTCP_HLG: Mat3 = [
  [0.5, 0.5, 0],
  [3625 / 4096, -7465 / 4096, 3840 / 4096],
  [9500 / 4096, -9212 / 4096, -288 / 4096],
];
const LMS_TO_IPT: Mat3 = [
  [1638 / 4096, 1638 / 4096, 820 / 4096],
  [18248 / 4096, -19870 / 4096, 1622 / 4096],
  [3300 / 4096, 1463 / 4096, -4763 / 4096],
];
const LMS_ICTCP_TO_RGB = invert3(RGB_TO_LMS_ICTCP);
const LMS_IPT_TO_RGB = invert3(RGB_TO_LMS_IPT);
const ICTCP_PQ_TO_LMS = invert3(LMS_TO_ICTCP_PQ);
const ICTCP_HLG_TO_LMS = invert3(LMS_TO_ICTCP_HLG);
const IPT_TO_LMS = invert3(LMS_TO_IPT);
const BRADFORD: Mat3 = [
  [0.8951, 0.2664, -0.1614],
  [-0.7502, 1.7135, 0.0367],
  [0.0389, -0.0685, 1.0296],
];
const BRADFORD_INVERSE = invert3(BRADFORD);
const SRGB_TO_XYZ_D65 = rgbToXyzMatrix(PRIMARIES[1]!);
const XYZ_D65_TO_SRGB = invert3(SRGB_TO_XYZ_D65);
const D50 = [0.34567, 0.35850] as const;
const D50_TO_D65 = chromaticAdaptation(D50, D65);
const NCLX_TO_SRGB = new Map<number, Mat3>();
const ICC_TRANSFORMS = new WeakMap<Uint8Array, IccTransform | null>();
const STANDARD_8_TABLES = new Map<string, Standard8Tables>();

type IccTransform = (red: number, green: number, blue: number, output: MutableVec3) => void;

interface Standard8Tables {
  y: Float64Array;
  red: Float64Array;
  greenRed: Float64Array;
  greenBlue: Float64Array;
  blue: Float64Array;
}

export interface FrameCrop {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** Convert a decoded frame to RGBA8 (row-major, straight alpha). */
export function frameToRgba(
  frame: DecodedFrame,
  matrixCoefficients: number,
  fullRange: boolean,
  colourPrimaries = 2,
  transferCharacteristics = 2,
  chromaSamplePosition = 0,
  iccProfile: Uint8Array | null = null,
  colorManagement = true,
  crop: FrameCrop | null = null,
  output: Uint8ClampedArray | null = null,
): Uint8ClampedArray {
  return frameToPixels(
    frame, matrixCoefficients, fullRange, colourPrimaries, transferCharacteristics,
    chromaSamplePosition, iccProfile, colorManagement, false, crop, output,
  );
}

/** Convert a decoded frame to the single grayscale channel used by an auxiliary alpha item. */
export function frameToAlpha(
  frame: DecodedFrame,
  matrixCoefficients: number,
  fullRange: boolean,
  colourPrimaries = 2,
  transferCharacteristics = 2,
  chromaSamplePosition = 0,
  crop: FrameCrop | null = null,
): Uint8ClampedArray {
  return frameToPixels(
    frame, matrixCoefficients, fullRange, colourPrimaries, transferCharacteristics,
    chromaSamplePosition, null, false, true, crop, null,
  );
}

function frameToPixels(
  frame: DecodedFrame,
  matrixCoefficients: number,
  fullRange: boolean,
  colourPrimaries: number,
  transferCharacteristics: number,
  chromaSamplePosition: number,
  iccProfile: Uint8Array | null,
  colorManagement: boolean,
  alphaOnly: boolean,
  crop: FrameCrop | null,
  output: Uint8ClampedArray | null,
): Uint8ClampedArray {
  const { bitDepth, chromaBitDepth, chromaFormat } = frame;
  const left = crop?.left ?? 0, top = crop?.top ?? 0;
  const width = crop?.width ?? frame.width, height = crop?.height ?? frame.height;
  const channels = alphaOnly ? 1 : 4;
  const outputLength = width * height * channels;
  if (output && output.length !== outputLength) {
    throw new RangeError(`RGBA output length ${output.length} does not match required length ${outputLength}`);
  }
  const out = output ?? new Uint8ClampedArray(outputLength);
  const maxValue = (1 << bitDepth) - 1;
  const rangeScale = 1 << Math.max(0, bitDepth - 8);
  const yOffset = fullRange ? 0 : 16 * rangeScale;
  const yRange = fullRange ? maxValue : 219 * rangeScale;
  const chromaMaxValue = (1 << chromaBitDepth) - 1;
  const chromaRangeScale = 1 << Math.max(0, chromaBitDepth - 8);
  const chromaMid = 1 << (chromaBitDepth - 1);
  const chromaRange = fullRange ? chromaMaxValue : 224 * chromaRangeScale;
  const componentOffset = fullRange ? 0 : 16 * chromaRangeScale;
  const componentRange = fullRange ? chromaMaxValue : 219 * chromaRangeScale;
  const luma = frame.luma;
  const iccTransform = colorManagement && iccProfile ? getIccTransform(iccProfile) : null;
  const manageNclx = colorManagement && !iccTransform && colourPrimaries !== 2 &&
    transferCharacteristics !== 2 && PRIMARIES[colourPrimaries] !== undefined;

  if (chromaFormat === CHROMA_MONO) {
    const rgb: MutableVec3 = [0, 0, 0];
    let offset = 0;
    for (let y = 0; y < height; y++) {
      const lumaRow = (top + y) * luma.stride + left;
      for (let x = 0; x < width; x++) {
        const value = (luma.data[lumaRow + x]! - yOffset) / yRange;
        rgb[0] = rgb[1] = rgb[2] = value;
        if (iccTransform) iccTransform(rgb[0], rgb[1], rgb[2], rgb);
        else if (manageNclx) {
          convertNclxToSrgbInto(rgb[0], rgb[1], rgb[2], colourPrimaries, transferCharacteristics, rgb);
        }
        out[offset++] = rgb[0] * 255;
        if (!alphaOnly) {
          out[offset++] = rgb[1] * 255;
          out[offset++] = rgb[2] * 255;
          out[offset++] = 255;
        }
      }
    }
    return out;
  }

  const cb = frame.cb, cr = frame.cr;
  const horizontalShift = chromaFormat === CHROMA_420 || chromaFormat === CHROMA_422 ? 1 : 0;
  const verticalShift = chromaFormat === CHROMA_420 ? 1 : 0;
  const coefficients = matrixCoefficients === 12 || matrixCoefficients === 13
    ? derivedKrKb(colourPrimaries)
    : MATRICES[matrixCoefficients] ?? MATRICES[2]!;

  if (bitDepth === 8 && chromaBitDepth === 8 && !iccTransform && !manageNclx &&
      matrixCoefficients !== 0 && matrixCoefficients !== 8 && matrixCoefficients !== 10 &&
      matrixCoefficients !== 11 && matrixCoefficients !== 13 && matrixCoefficients !== 14 &&
      matrixCoefficients !== 15) {
    writeStandard8(
      out, frame, left, top, width, height, alphaOnly, fullRange, coefficients,
      horizontalShift, verticalShift, chromaSamplePosition, matrixCoefficients,
    );
    return out;
  }

  const rgb: MutableVec3 = [0, 0, 0];
  const chroma: [number, number] = [0, 0];
  let offset = 0;
  for (let y = 0; y < height; y++) {
    const sourceY = top + y;
    const lumaRow = sourceY * luma.stride + left;
    for (let x = 0; x < width; x++) {
      const sourceX = left + x;
      const yCode = luma.data[lumaRow + x]!;
      sampleChromaPair(cb, cr, sourceX, sourceY, horizontalShift, verticalShift, chromaSamplePosition, chroma);
      const cbCode = chroma[0], crCode = chroma[1];

      if (matrixCoefficients === 0) {
        // H.273 identity carries planes in G, B, R order. Unlike YCbCr,
        // limited-range identity applies the luma offset/range to all planes.
        rgb[0] = (crCode - componentOffset) / componentRange;
        rgb[1] = (yCode - yOffset) / yRange;
        rgb[2] = (cbCode - componentOffset) / componentRange;
      } else if (matrixCoefficients === 8) {
        // Integer YCgCo inverse (H.273 equations 54-57).
        const cg = cbCode - chromaMid, co = crCode - chromaMid;
        const temporary = yCode - cg;
        rgb[0] = (temporary + co - yOffset) / yRange;
        rgb[1] = (yCode + cg - yOffset) / yRange;
        rgb[2] = (temporary - co - yOffset) / yRange;
      } else {
        const yPrime = (yCode - yOffset) / yRange;
        const pb = (cbCode - chromaMid) / chromaRange;
        const pr = (crCode - chromaMid) / chromaRange;
        inverseMatrixInto(
          yPrime, pb, pr, matrixCoefficients, coefficients,
          transferCharacteristics, rgb,
        );
      }

      if (iccTransform) iccTransform(rgb[0], rgb[1], rgb[2], rgb);
      else if (manageNclx) {
        convertNclxToSrgbInto(rgb[0], rgb[1], rgb[2], colourPrimaries, transferCharacteristics, rgb);
      }

      out[offset++] = rgb[0] * 255;
      if (!alphaOnly) {
        out[offset++] = rgb[1] * 255;
        out[offset++] = rgb[2] * 255;
        out[offset++] = 255;
      }
    }
  }
  return out;
}

function writeStandard8(
  out: Uint8ClampedArray, frame: DecodedFrame,
  left: number, top: number, width: number, height: number, alphaOnly: boolean,
  fullRange: boolean, coefficients: readonly [number, number],
  horizontalShift: number, verticalShift: number, chromaSamplePosition: number,
  matrixCoefficients: number,
): void {
  const luma = frame.luma, cb = frame.cb, cr = frame.cr;
  const yOffset = fullRange ? 0 : 16, yRange = fullRange ? 255 : 219;
  const chromaRange = fullRange ? 255 : 224;
  const [kr, kb] = coefficients, kg = 1 - kr - kb;
  const redScale = 2 * (1 - kr), blueScale = 2 * (1 - kb);
  const greenRedScale = 2 * kr * (1 - kr) / kg;
  const greenBlueScale = 2 * kb * (1 - kb) / kg;
  const direct = (!horizontalShift && !verticalShift) || chromaSamplePosition === 0;
  const tables = standard8Tables(
    matrixCoefficients, fullRange, yOffset, yRange, chromaRange,
    redScale, blueScale, greenRedScale, greenBlueScale,
  );
  if (direct) {
    if (horizontalShift === 1 && verticalShift === 1 && !(left & 1) && !(top & 1)) {
      writeDirect420Standard8(out, frame, left, top, width, height, alphaOnly, tables);
    } else {
      writeDirectStandard8(
        out, frame, left, top, width, height, alphaOnly, horizontalShift, verticalShift, tables,
      );
    }
    return;
  }
  const x0s = direct ? null : new Uint32Array(width);
  const x1s = direct ? null : new Uint32Array(width);
  const fractionsX = direct ? null : new Float64Array(width);
  if (!direct) {
    const horizontallyCentered = chromaSamplePosition === 3 || chromaSamplePosition === 4;
    for (let x = 0; x < width; x++) {
      const sourceX = horizontallyCentered ? (left + x) / 2 - 0.25 : (left + x) / 2;
      const floorX = Math.floor(sourceX);
      const x0 = Math.max(0, Math.min(cb.width - 1, floorX));
      x0s![x] = x0;
      x1s![x] = Math.max(0, Math.min(cb.width - 1, x0 + 1));
      fractionsX![x] = Math.max(0, Math.min(1, sourceX - floorX));
    }
  }
  let offset = 0;
  for (let y = 0; y < height; y++) {
    const sourceY = top + y;
    const lumaRow = sourceY * luma.stride + left;
    const directChromaRow = (sourceY >> verticalShift) * cb.stride;
    let chromaRow0 = 0, chromaRow1 = 0, fractionY = 0;
    if (!direct) {
      const interpolatedY = verticalShift
        ? (chromaSamplePosition === 2 || chromaSamplePosition === 4 ? sourceY / 2 : sourceY / 2 - 0.25)
        : sourceY;
      const floorY = Math.floor(interpolatedY);
      const y0 = Math.max(0, Math.min(cb.height - 1, floorY));
      const y1 = Math.max(0, Math.min(cb.height - 1, y0 + 1));
      chromaRow0 = y0 * cb.stride;
      chromaRow1 = y1 * cb.stride;
      fractionY = Math.max(0, Math.min(1, interpolatedY - floorY));
    }
    for (let x = 0; x < width; x++) {
      const sourceX = left + x;
      let cbCode: number, crCode: number;
      if (direct) {
        const chromaIndex = directChromaRow + (sourceX >> horizontalShift);
        cbCode = cb.data[chromaIndex]!;
        crCode = cr.data[chromaIndex]!;
      } else {
        const x0 = x0s![x]!, x1 = x1s![x]!, fractionX = fractionsX![x]!;
        const inverseX = 1 - fractionX, inverseY = 1 - fractionY;
        const cbTop = cb.data[chromaRow0 + x0]! * inverseX + cb.data[chromaRow0 + x1]! * fractionX;
        const cbBottom = cb.data[chromaRow1 + x0]! * inverseX + cb.data[chromaRow1 + x1]! * fractionX;
        const crTop = cr.data[chromaRow0 + x0]! * inverseX + cr.data[chromaRow0 + x1]! * fractionX;
        const crBottom = cr.data[chromaRow1 + x0]! * inverseX + cr.data[chromaRow1 + x1]! * fractionX;
        cbCode = cbTop * inverseY + cbBottom * fractionY;
        crCode = crTop * inverseY + crBottom * fractionY;
      }
      const yPrime = tables.y[luma.data[lumaRow + x]!]!;
      const pb = (cbCode - 128) / chromaRange;
      const pr = (crCode - 128) / chromaRange;
      out[offset++] = (yPrime + redScale * pr) * 255;
      if (!alphaOnly) {
        out[offset++] = (yPrime - greenRedScale * pr - greenBlueScale * pb) * 255;
        out[offset++] = (yPrime + blueScale * pb) * 255;
        out[offset++] = 255;
      }
    }
  }
}

function standard8Tables(
  matrix: number, fullRange: boolean, yOffset: number, yRange: number, chromaRange: number,
  redScale: number, blueScale: number, greenRedScale: number, greenBlueScale: number,
): Standard8Tables {
  // Matrix 12 derives Kr/Kb from colour primaries, so the numeric matrix code
  // alone is not a sufficient cache key.
  const key = `${matrix}:${+fullRange}:${redScale}:${blueScale}`;
  let tables = STANDARD_8_TABLES.get(key);
  if (tables) return tables;
  tables = {
    y: new Float64Array(256),
    red: new Float64Array(256),
    greenRed: new Float64Array(256),
    greenBlue: new Float64Array(256),
    blue: new Float64Array(256),
  };
  for (let code = 0; code < 256; code++) {
    tables.y[code] = (code - yOffset) / yRange;
    const chroma = (code - 128) / chromaRange;
    tables.red[code] = redScale * chroma;
    tables.greenRed[code] = greenRedScale * chroma;
    tables.greenBlue[code] = greenBlueScale * chroma;
    tables.blue[code] = blueScale * chroma;
  }
  STANDARD_8_TABLES.set(key, tables);
  return tables;
}

function writeDirectStandard8(
  out: Uint8ClampedArray, frame: DecodedFrame,
  left: number, top: number, width: number, height: number, alphaOnly: boolean,
  horizontalShift: number, verticalShift: number, tables: Standard8Tables,
): void {
  const luma = frame.luma, cb = frame.cb, cr = frame.cr;
  const yTable = tables.y, red = tables.red, greenRed = tables.greenRed;
  const greenBlue = tables.greenBlue, blue = tables.blue;
  let output = 0;
  for (let y = 0; y < height; y++) {
    const sourceY = top + y;
    const lumaRow = sourceY * luma.stride + left;
    const chromaRow = (sourceY >> verticalShift) * cb.stride;
    for (let x = 0; x < width; x++) {
      const sourceX = left + x;
      const chromaIndex = chromaRow + (sourceX >> horizontalShift);
      const crCode = cr.data[chromaIndex]!;
      const yPrime = yTable[luma.data[lumaRow + x]!]!;
      out[output++] = (yPrime + red[crCode]!) * 255;
      if (!alphaOnly) {
        const cbCode = cb.data[chromaIndex]!;
        out[output++] = (yPrime - greenRed[crCode]! - greenBlue[cbCode]!) * 255;
        out[output++] = (yPrime + blue[cbCode]!) * 255;
        out[output++] = 255;
      }
    }
  }
}

function writeDirect420Standard8(
  out: Uint8ClampedArray, frame: DecodedFrame,
  left: number, top: number, width: number, height: number, alphaOnly: boolean,
  tables: Standard8Tables,
): void {
  const luma = frame.luma, cb = frame.cb, cr = frame.cr;
  const yTable = tables.y, red = tables.red, greenRed = tables.greenRed;
  const greenBlue = tables.greenBlue, blue = tables.blue;
  const channels = alphaOnly ? 1 : 4;
  for (let y = 0; y < height; y += 2) {
    const sourceY = top + y;
    const lumaRow0 = sourceY * luma.stride + left;
    const lumaRow1 = Math.min(sourceY + 1, top + height - 1) * luma.stride + left;
    const chromaRow = (sourceY >> 1) * cb.stride + (left >> 1);
    const outputRow0 = y * width * channels;
    const outputRow1 = Math.min(y + 1, height - 1) * width * channels;
    for (let x = 0; x < width; x += 2) {
      const chromaIndex = chromaRow + (x >> 1);
      const cbCode = cb.data[chromaIndex]!, crCode = cr.data[chromaIndex]!;
      const redTerm = red[crCode]!, greenRedTerm = greenRed[crCode]!;
      const greenBlueTerm = greenBlue[cbCode]!, blueTerm = blue[cbCode]!;
      writeDirectPixel(
        out, outputRow0 + x * channels, yTable[luma.data[lumaRow0 + x]!]!,
        redTerm, greenRedTerm, greenBlueTerm, blueTerm, alphaOnly,
      );
      if (x + 1 < width) {
        writeDirectPixel(
          out, outputRow0 + (x + 1) * channels, yTable[luma.data[lumaRow0 + x + 1]!]!,
          redTerm, greenRedTerm, greenBlueTerm, blueTerm, alphaOnly,
        );
      }
      if (y + 1 < height) {
        writeDirectPixel(
          out, outputRow1 + x * channels, yTable[luma.data[lumaRow1 + x]!]!,
          redTerm, greenRedTerm, greenBlueTerm, blueTerm, alphaOnly,
        );
        if (x + 1 < width) {
          writeDirectPixel(
            out, outputRow1 + (x + 1) * channels, yTable[luma.data[lumaRow1 + x + 1]!]!,
            redTerm, greenRedTerm, greenBlueTerm, blueTerm, alphaOnly,
          );
        }
      }
    }
  }
}

function writeDirectPixel(
  out: Uint8ClampedArray, offset: number, yPrime: number,
  red: number, greenRed: number, greenBlue: number, blue: number, alphaOnly: boolean,
): void {
  out[offset] = (yPrime + red) * 255;
  if (!alphaOnly) {
    out[offset + 1] = (yPrime - greenRed - greenBlue) * 255;
    out[offset + 2] = (yPrime + blue) * 255;
    out[offset + 3] = 255;
  }
}

function sampleChromaPair(
  cb: DecodedFrame['luma'], cr: DecodedFrame['luma'], x: number, y: number,
  horizontalShift: number, verticalShift: number, chromaSamplePosition: number,
  output: [number, number],
): void {
  if (!horizontalShift && !verticalShift) {
    const index = y * cb.stride + x;
    output[0] = cb.data[index]!; output[1] = cr.data[index]!;
    return;
  }
  if (chromaSamplePosition === 0) {
    const index = (y >> verticalShift) * cb.stride + (x >> horizontalShift);
    output[0] = cb.data[index]!; output[1] = cr.data[index]!;
    return;
  }
  // AV1's signaled positions are horizontally co-sited; the conventional
  // fallback for CSP_UNKNOWN and the fixed 4:2:2 layout use the same siting.
  const horizontallyCentered = chromaSamplePosition === 3 || chromaSamplePosition === 4;
  const sourceX = horizontalShift ? (horizontallyCentered ? x / 2 - 0.25 : x / 2) : x;
  const sourceY = verticalShift ?
    (chromaSamplePosition === 2 || chromaSamplePosition === 4 ? y / 2 : y / 2 - 0.25) : y;
  const floorX = Math.floor(sourceX), floorY = Math.floor(sourceY);
  const x0 = Math.max(0, Math.min(cb.width - 1, floorX));
  const y0 = Math.max(0, Math.min(cb.height - 1, floorY));
  const x1 = Math.max(0, Math.min(cb.width - 1, x0 + 1));
  const y1 = Math.max(0, Math.min(cb.height - 1, y0 + 1));
  const fractionX = Math.max(0, Math.min(1, sourceX - floorX));
  const fractionY = Math.max(0, Math.min(1, sourceY - floorY));
  const row0 = y0 * cb.stride, row1 = y1 * cb.stride;
  const inverseX = 1 - fractionX, inverseY = 1 - fractionY;
  const cbTop = cb.data[row0 + x0]! * inverseX + cb.data[row0 + x1]! * fractionX;
  const cbBottom = cb.data[row1 + x0]! * inverseX + cb.data[row1 + x1]! * fractionX;
  const crTop = cr.data[row0 + x0]! * inverseX + cr.data[row0 + x1]! * fractionX;
  const crBottom = cr.data[row1 + x0]! * inverseX + cr.data[row1 + x1]! * fractionX;
  output[0] = cbTop * inverseY + cbBottom * fractionY;
  output[1] = crTop * inverseY + crBottom * fractionY;
}

function inverseMatrixInto(
  y: number, pb: number, pr: number,
  matrix: number, coefficients: readonly [number, number], transfer: number,
  output: MutableVec3,
): void {
  if (matrix === 10 || matrix === 13) {
    inverseConstantLuminanceInto(y, pb, pr, coefficients, transfer, matrix, output);
    return;
  }
  if (matrix === 11) {
    output[0] = 2 * pr + 0.991902 * y;
    output[1] = y;
    output[2] = (2 * pb + y) / 0.986566;
    return;
  }
  if (matrix === 14) {
    const encoded = transfer === 18 ? ICTCP_HLG_TO_LMS : ICTCP_PQ_TO_LMS;
    const l0 = inverseTransfer(encoded[0][0] * y + encoded[0][1] * pb + encoded[0][2] * pr, transfer, matrix);
    const l1 = inverseTransfer(encoded[1][0] * y + encoded[1][1] * pb + encoded[1][2] * pr, transfer, matrix);
    const l2 = inverseTransfer(encoded[2][0] * y + encoded[2][1] * pb + encoded[2][2] * pr, transfer, matrix);
    output[0] = forwardTransfer(LMS_ICTCP_TO_RGB[0][0] * l0 + LMS_ICTCP_TO_RGB[0][1] * l1 + LMS_ICTCP_TO_RGB[0][2] * l2, transfer, matrix);
    output[1] = forwardTransfer(LMS_ICTCP_TO_RGB[1][0] * l0 + LMS_ICTCP_TO_RGB[1][1] * l1 + LMS_ICTCP_TO_RGB[1][2] * l2, transfer, matrix);
    output[2] = forwardTransfer(LMS_ICTCP_TO_RGB[2][0] * l0 + LMS_ICTCP_TO_RGB[2][1] * l1 + LMS_ICTCP_TO_RGB[2][2] * l2, transfer, matrix);
    return;
  }
  if (matrix === 15) {
    const l0 = inverseTransfer(IPT_TO_LMS[0][0] * y + IPT_TO_LMS[0][1] * pb + IPT_TO_LMS[0][2] * pr, transfer, matrix);
    const l1 = inverseTransfer(IPT_TO_LMS[1][0] * y + IPT_TO_LMS[1][1] * pb + IPT_TO_LMS[1][2] * pr, transfer, matrix);
    const l2 = inverseTransfer(IPT_TO_LMS[2][0] * y + IPT_TO_LMS[2][1] * pb + IPT_TO_LMS[2][2] * pr, transfer, matrix);
    output[0] = forwardTransfer(LMS_IPT_TO_RGB[0][0] * l0 + LMS_IPT_TO_RGB[0][1] * l1 + LMS_IPT_TO_RGB[0][2] * l2, transfer, matrix);
    output[1] = forwardTransfer(LMS_IPT_TO_RGB[1][0] * l0 + LMS_IPT_TO_RGB[1][1] * l1 + LMS_IPT_TO_RGB[1][2] * l2, transfer, matrix);
    output[2] = forwardTransfer(LMS_IPT_TO_RGB[2][0] * l0 + LMS_IPT_TO_RGB[2][1] * l1 + LMS_IPT_TO_RGB[2][2] * l2, transfer, matrix);
    return;
  }

  const [kr, kb] = coefficients;
  const kg = 1 - kr - kb;
  output[0] = y + 2 * (1 - kr) * pr;
  output[2] = y + 2 * (1 - kb) * pb;
  output[1] = y - 2 * kr * (1 - kr) / kg * pr - 2 * kb * (1 - kb) / kg * pb;
}

function inverseConstantLuminanceInto(
  yPrime: number, pb: number, pr: number,
  coefficients: readonly [number, number], transfer: number, matrix: number,
  output: MutableVec3,
): void {
  const [kr, kb] = coefficients;
  const kg = 1 - kr - kb;
  const negativeB = forwardTransfer(1 - kb, transfer, matrix);
  const positiveB = 1 - forwardTransfer(kb, transfer, matrix);
  const negativeR = forwardTransfer(1 - kr, transfer, matrix);
  const positiveR = 1 - forwardTransfer(kr, transfer, matrix);
  const bPrime = yPrime + 2 * (pb <= 0 ? negativeB : positiveB) * pb;
  const rPrime = yPrime + 2 * (pr <= 0 ? negativeR : positiveR) * pr;
  const linearY = inverseTransfer(yPrime, transfer, matrix);
  const linearB = inverseTransfer(bPrime, transfer, matrix);
  const linearR = inverseTransfer(rPrime, transfer, matrix);
  const linearG = (linearY - kr * linearR - kb * linearB) / kg;
  output[0] = rPrime;
  output[1] = forwardTransfer(linearG, transfer, matrix);
  output[2] = bPrime;
}

function derivedKrKb(primariesCode: number): readonly [number, number] {
  const p = PRIMARIES[primariesCode] ?? PRIMARIES[1]!;
  const [xr, yr] = p.r, [xg, yg] = p.g, [xb, yb] = p.b, [xw, yw] = p.w;
  const zr = 1 - xr - yr, zg = 1 - xg - yg, zb = 1 - xb - yb, zw = 1 - xw - yw;
  const denominator = yw * (
    xr * (yg * zb - yb * zg) + xg * (yb * zr - yr * zb) + xb * (yr * zg - yg * zr)
  );
  if (Math.abs(denominator) < 1e-12) return MATRICES[1]!;
  const kr = yr * (
    xw * (yg * zb - yb * zg) + yw * (xb * zg - xg * zb) + zw * (xg * yb - xb * yg)
  ) / denominator;
  const kb = yb * (
    xw * (yr * zg - yg * zr) + yw * (xg * zr - xr * zg) + zw * (xr * yg - xg * yr)
  ) / denominator;
  return [kr, kb];
}

function forwardTransfer(value: number, transfer: number, matrix: number): number {
  const sign = value < 0 ? -1 : 1;
  const absolute = Math.abs(value);
  switch (transfer) {
    case 1: case 6: case 14: return value < 0.018 ? 4.5 * value : 1.099 * Math.pow(value, 0.45) - 0.099;
    case 15: return value < 0.0181 ? 4.5 * value : 1.0993 * Math.pow(value, 0.45) - 0.0993;
    case 4: return sign * Math.pow(absolute, 1 / 2.2);
    case 5: return sign * Math.pow(absolute, 1 / 2.8);
    case 7: {
      const beta = 0.0228, alpha = 1.1115;
      return value < beta ? 4 * value : alpha * Math.pow(value, 0.45) - (alpha - 1);
    }
    case 8: return value;
    case 9: return value < 0.01 ? 0 : 1 + Math.log10(value) / 2;
    case 10: return value < Math.sqrt(10) / 1000 ? 0 : 1 + Math.log10(value) / 2.5;
    case 11: {
      const alpha = 1.099, beta = 0.018;
      if (value <= -beta) return -alpha * Math.pow(-value, 0.45) + (alpha - 1);
      return value < beta ? 4.5 * value : alpha * Math.pow(value, 0.45) - (alpha - 1);
    }
    case 12: {
      const alpha = 1.099, beta = 0.018, gamma = 0.0045;
      if (value <= -gamma) return -(alpha * Math.pow(-4 * value, 0.45) - (alpha - 1)) / 4;
      return value < beta ? 4.5 * value : alpha * Math.pow(value, 0.45) - (alpha - 1);
    }
    case 13: {
      const alpha = 1.055, beta = 0.0031308;
      if (matrix !== 0 && value < -beta) return -alpha * Math.pow(-value, 1 / 2.4) + (alpha - 1);
      return value < beta ? 12.92 * value : alpha * Math.pow(value, 1 / 2.4) - (alpha - 1);
    }
    case 16: {
      const c1 = 107 / 128, c2 = 2413 / 128, c3 = 2392 / 128;
      const m = 2523 / 32, n = 653 / 4096;
      const powered = Math.pow(Math.max(0, value), n);
      return Math.pow((c1 + c2 * powered) / (1 + c3 * powered), m);
    }
    case 17: return Math.pow(Math.max(0, 48 * value / 52.37), 1 / 2.6);
    case 18: {
      const a = 0.17883277, b = 0.28466892, c = 0.55991073;
      return value <= 1 / 12 ? Math.sqrt(Math.max(0, 3 * value)) : a * Math.log(12 * value - b) + c;
    }
    default: {
      const beta = 0.018053968510807, alpha = 1.099296826809442;
      return value < beta ? 4.5 * value : alpha * Math.pow(value, 0.45) - (alpha - 1);
    }
  }
}

function inverseTransfer(value: number, transfer: number, matrix: number): number {
  const sign = value < 0 ? -1 : 1;
  const absolute = Math.abs(value);
  switch (transfer) {
    case 1: case 6: case 14: {
      const threshold = 4.5 * 0.018;
      return value < threshold ? value / 4.5 : Math.pow((value + 0.099) / 1.099, 1 / 0.45);
    }
    case 15: {
      const threshold = 4.5 * 0.0181;
      return value < threshold ? value / 4.5 : Math.pow((value + 0.0993) / 1.0993, 1 / 0.45);
    }
    case 4: return sign * Math.pow(absolute, 2.2);
    case 5: return sign * Math.pow(absolute, 2.8);
    case 7: {
      const beta = 0.0228, alpha = 1.1115;
      return value < 4 * beta ? value / 4 : Math.pow((value + alpha - 1) / alpha, 1 / 0.45);
    }
    case 8: return value;
    case 9: return value <= 0 ? 0 : Math.pow(10, 2 * (value - 1));
    case 10: return value <= 0 ? 0 : Math.pow(10, 2.5 * (value - 1));
    case 11: {
      const alpha = 1.099, threshold = 4.5 * 0.018;
      if (value <= -threshold) return -Math.pow((-value + alpha - 1) / alpha, 1 / 0.45);
      return value < threshold ? value / 4.5 : Math.pow((value + alpha - 1) / alpha, 1 / 0.45);
    }
    case 12: {
      const alpha = 1.099, positive = 4.5 * 0.018, negative = -4.5 * 0.0045;
      if (value <= negative) return -Math.pow((alpha - 1 - 4 * value) / alpha, 1 / 0.45) / 4;
      return value < positive ? value / 4.5 : Math.pow((value + alpha - 1) / alpha, 1 / 0.45);
    }
    case 13: {
      const alpha = 1.055, threshold = 12.92 * 0.0031308;
      if (matrix !== 0 && value < -threshold) return -Math.pow((-value + alpha - 1) / alpha, 2.4);
      return value < threshold ? value / 12.92 : Math.pow((value + alpha - 1) / alpha, 2.4);
    }
    case 16: {
      const c1 = 107 / 128, c2 = 2413 / 128, c3 = 2392 / 128;
      const m = 2523 / 32, n = 653 / 4096;
      const powered = Math.pow(Math.max(0, value), 1 / m);
      return Math.pow(Math.max(0, (powered - c1) / (c2 - c3 * powered)), 1 / n);
    }
    case 17: return 52.37 / 48 * Math.pow(Math.max(0, value), 2.6);
    case 18: {
      const a = 0.17883277, b = 0.28466892, c = 0.55991073;
      return value <= 0.5 ? value * value / 3 : (Math.exp((value - c) / a) + b) / 12;
    }
    default: {
      const beta = 0.018053968510807, alpha = 1.099296826809442;
      const threshold = 4.5 * beta;
      return value < threshold ? value / 4.5 : Math.pow((value + alpha - 1) / alpha, 1 / 0.45);
    }
  }
}

function multiply3(matrix: Mat3, vector: Vec3): Vec3 {
  return [
    matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
    matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
    matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
  ];
}

function convertNclxToSrgbInto(
  red: number, green: number, blue: number,
  primariesCode: number, transfer: number, output: MutableVec3,
): void {
  if (primariesCode === 1 && transfer === 13) {
    output[0] = red; output[1] = green; output[2] = blue;
    return;
  }
  const sourcePrimaries = PRIMARIES[primariesCode];
  if (!sourcePrimaries || !isKnownTransfer(transfer)) {
    output[0] = red; output[1] = green; output[2] = blue;
    return;
  }
  const linearRed = inverseTransfer(red, transfer, 0);
  const linearGreen = inverseTransfer(green, transfer, 0);
  const linearBlue = inverseTransfer(blue, transfer, 0);
  let conversion = NCLX_TO_SRGB.get(primariesCode);
  if (!conversion) {
    conversion = multiplyMatrices(
      XYZ_D65_TO_SRGB,
      multiplyMatrices(chromaticAdaptation(sourcePrimaries.w, D65), rgbToXyzMatrix(sourcePrimaries)),
    );
    NCLX_TO_SRGB.set(primariesCode, conversion);
  }
  let convertedRed = conversion[0][0] * linearRed + conversion[0][1] * linearGreen + conversion[0][2] * linearBlue;
  let convertedGreen = conversion[1][0] * linearRed + conversion[1][1] * linearGreen + conversion[1][2] * linearBlue;
  let convertedBlue = conversion[2][0] * linearRed + conversion[2][1] * linearGreen + conversion[2][2] * linearBlue;
  if (transfer === 16) {
    const scale = 10_000 / 203;
    convertedRed *= scale; convertedGreen *= scale; convertedBlue *= scale;
    const luminance = Math.max(0, 0.2126 * convertedRed + 0.7152 * convertedGreen + 0.0722 * convertedBlue);
    if (luminance > 1e-12) {
      const mapped = luminance * (1 + luminance / (scale * scale)) / (1 + luminance);
      const factor = mapped / luminance;
      convertedRed *= factor; convertedGreen *= factor; convertedBlue *= factor;
    }
  } else if (transfer === 18) {
    const luminance = Math.max(1e-12, 0.2627 * convertedRed + 0.6780 * convertedGreen + 0.0593 * convertedBlue);
    const factor = luminance ** 0.2;
    convertedRed *= factor; convertedGreen *= factor; convertedBlue *= factor;
  }
  output[0] = forwardTransfer(convertedRed, 13, 1);
  output[1] = forwardTransfer(convertedGreen, 13, 1);
  output[2] = forwardTransfer(convertedBlue, 13, 1);
}

function isKnownTransfer(transfer: number): boolean {
  return transfer === 1 || transfer >= 4 && transfer <= 18;
}

function rgbToXyzMatrix(primaries: Chromaticities): Mat3 {
  const column = (xy: readonly [number, number]): Vec3 => [xy[0] / xy[1], 1, (1 - xy[0] - xy[1]) / xy[1]];
  const r = column(primaries.r), g = column(primaries.g), b = column(primaries.b);
  const unscaled: Mat3 = [
    [r[0], g[0], b[0]],
    [r[1], g[1], b[1]],
    [r[2], g[2], b[2]],
  ];
  const white = xyToXyz(primaries.w);
  const scales = multiply3(invert3(unscaled), white);
  return [
    [unscaled[0][0] * scales[0], unscaled[0][1] * scales[1], unscaled[0][2] * scales[2]],
    [unscaled[1][0] * scales[0], unscaled[1][1] * scales[1], unscaled[1][2] * scales[2]],
    [unscaled[2][0] * scales[0], unscaled[2][1] * scales[1], unscaled[2][2] * scales[2]],
  ];
}

function xyToXyz(xy: readonly [number, number]): Vec3 {
  return [xy[0] / xy[1], 1, (1 - xy[0] - xy[1]) / xy[1]];
}

function chromaticAdaptation(
  sourceWhite: readonly [number, number], targetWhite: readonly [number, number],
): Mat3 {
  if (Math.abs(sourceWhite[0] - targetWhite[0]) < 1e-9 &&
      Math.abs(sourceWhite[1] - targetWhite[1]) < 1e-9) {
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  }
  const sourceCone = multiply3(BRADFORD, xyToXyz(sourceWhite));
  const targetCone = multiply3(BRADFORD, xyToXyz(targetWhite));
  const scale: Mat3 = [
    [targetCone[0] / sourceCone[0], 0, 0],
    [0, targetCone[1] / sourceCone[1], 0],
    [0, 0, targetCone[2] / sourceCone[2]],
  ];
  return multiplyMatrices(BRADFORD_INVERSE, multiplyMatrices(scale, BRADFORD));
}

function multiplyMatrices(left: Mat3, right: Mat3): Mat3 {
  const value = (row: number, column: number): number =>
    left[row]![0] * right[0][column] + left[row]![1] * right[1][column] + left[row]![2] * right[2][column];
  return [
    [value(0, 0), value(0, 1), value(0, 2)],
    [value(1, 0), value(1, 1), value(1, 2)],
    [value(2, 0), value(2, 1), value(2, 2)],
  ];
}

function getIccTransform(profile: Uint8Array): IccTransform | null {
  const cached = ICC_TRANSFORMS.get(profile);
  if (cached !== undefined) return cached;
  const transform = parseMatrixIccProfile(profile);
  ICC_TRANSFORMS.set(profile, transform);
  return transform;
}

function parseMatrixIccProfile(profile: Uint8Array): IccTransform | null {
  if (profile.length < 132) return null;
  const view = new DataView(profile.buffer, profile.byteOffset, profile.byteLength);
  const declaredSize = view.getUint32(0);
  if (declaredSize < 132 || declaredSize > profile.length || ascii(profile, 16) !== 'RGB ' ||
      ascii(profile, 20) !== 'XYZ ' || ascii(profile, 36) !== 'acsp') {
    return null;
  }
  const count = view.getUint32(128);
  if (count > 4096 || 132 + count * 12 > declaredSize) return null;
  const tags = new Map<string, { offset: number; size: number }>();
  for (let index = 0; index < count; index++) {
    const entry = 132 + index * 12;
    const offset = view.getUint32(entry + 4), size = view.getUint32(entry + 8);
    if (offset > declaredSize || size > declaredSize - offset) return null;
    tags.set(ascii(profile, entry), { offset, size });
  }
  const rXyz = readIccXyz(profile, view, tags.get('rXYZ'));
  const gXyz = readIccXyz(profile, view, tags.get('gXYZ'));
  const bXyz = readIccXyz(profile, view, tags.get('bXYZ'));
  const rTrc = readIccCurve(profile, view, tags.get('rTRC'));
  const gTrc = readIccCurve(profile, view, tags.get('gTRC'));
  const bTrc = readIccCurve(profile, view, tags.get('bTRC'));
  if (!rXyz || !gXyz || !bXyz || !rTrc || !gTrc || !bTrc) return null;
  const sourceToD50: Mat3 = [
    [rXyz[0], gXyz[0], bXyz[0]],
    [rXyz[1], gXyz[1], bXyz[1]],
    [rXyz[2], gXyz[2], bXyz[2]],
  ];
  const sourceToSrgb = multiplyMatrices(XYZ_D65_TO_SRGB, multiplyMatrices(D50_TO_D65, sourceToD50));
  return (red: number, green: number, blue: number, output: MutableVec3): void => {
    const linearRed = rTrc(red), linearGreen = gTrc(green), linearBlue = bTrc(blue);
    output[0] = forwardTransfer(
      sourceToSrgb[0][0] * linearRed + sourceToSrgb[0][1] * linearGreen + sourceToSrgb[0][2] * linearBlue, 13, 1,
    );
    output[1] = forwardTransfer(
      sourceToSrgb[1][0] * linearRed + sourceToSrgb[1][1] * linearGreen + sourceToSrgb[1][2] * linearBlue, 13, 1,
    );
    output[2] = forwardTransfer(
      sourceToSrgb[2][0] * linearRed + sourceToSrgb[2][1] * linearGreen + sourceToSrgb[2][2] * linearBlue, 13, 1,
    );
  };
}

function readIccXyz(
  profile: Uint8Array, view: DataView, tag: { offset: number; size: number } | undefined,
): Vec3 | null {
  if (!tag || tag.size < 20 || ascii(profile, tag.offset) !== 'XYZ ') return null;
  return [fixed16(view, tag.offset + 8), fixed16(view, tag.offset + 12), fixed16(view, tag.offset + 16)];
}

function readIccCurve(
  profile: Uint8Array, view: DataView, tag: { offset: number; size: number } | undefined,
): ((value: number) => number) | null {
  if (!tag || tag.size < 12) return null;
  const type = ascii(profile, tag.offset);
  if (type === 'curv') {
    const count = view.getUint32(tag.offset + 8);
    if (count === 0) return value => value;
    if (count === 1) {
      if (tag.size < 14) return null;
      const gamma = view.getUint16(tag.offset + 12) / 256;
      return value => Math.max(0, Math.min(1, value)) ** gamma;
    }
    if (count > 65_536 || 12 + count * 2 > tag.size) return null;
    const table = new Uint16Array(count);
    for (let index = 0; index < count; index++) table[index] = view.getUint16(tag.offset + 12 + index * 2);
    return value => {
      const position = Math.max(0, Math.min(1, value)) * (count - 1);
      const lower = Math.floor(position), upper = Math.min(count - 1, lower + 1);
      if (lower === upper) return table[lower]! / 65_535;
      return (table[lower]! * (upper - position) + table[upper]! * (position - lower)) / 65_535;
    };
  }
  if (type !== 'para' || tag.size < 16) return null;
  const functionType = view.getUint16(tag.offset + 8);
  const parameterCounts = [1, 3, 4, 5, 7];
  const parameterCount = parameterCounts[functionType];
  if (parameterCount === undefined || 12 + parameterCount * 4 > tag.size) return null;
  const p = Array.from({ length: parameterCount }, (_, index) => fixed16(view, tag.offset + 12 + index * 4));
  return value => {
    const x = Math.max(0, Math.min(1, value));
    const [g, a = 1, b = 0, c = 0, d = 0, e = 0, f = 0] = p;
    if (functionType === 0) return x ** g!;
    const threshold = -b / a;
    if (functionType === 1) return x >= threshold ? (a * x + b) ** g! : 0;
    if (functionType === 2) return x >= threshold ? (a * x + b) ** g! + c : c;
    if (functionType === 3) return x >= d ? (a * x + b) ** g! : c * x;
    return x >= d ? (a * x + b) ** g! + e : c * x + f;
  };
}

function fixed16(view: DataView, offset: number): number {
  return view.getInt32(offset) / 65_536;
}

function ascii(bytes: Uint8Array, offset: number): string {
  if (offset < 0 || offset + 4 > bytes.length) return '';
  return String.fromCharCode(bytes[offset]!, bytes[offset + 1]!, bytes[offset + 2]!, bytes[offset + 3]!);
}

function invert3(matrix: Mat3): Mat3 {
  const [a, b, c] = matrix[0], [d, e, f] = matrix[1], [g, h, i] = matrix[2];
  const A = e * i - f * h, B = c * h - b * i, Cc = b * f - c * e;
  const D = f * g - d * i, E = a * i - c * g, F = c * d - a * f;
  const G = d * h - e * g, H = b * g - a * h, I = a * e - b * d;
  const determinant = a * A + b * D + c * G;
  return [
    [A / determinant, B / determinant, Cc / determinant],
    [D / determinant, E / determinant, F / determinant],
    [G / determinant, H / determinant, I / determinant],
  ];
}
