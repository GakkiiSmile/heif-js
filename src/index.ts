import { detectFormat, HeifFile } from './bmff.ts';
import type { ImageItem } from './bmff.ts';
import { frameToRgba } from './color.ts';
import { Av1Decoder } from './av1/decode.ts';
import { OBU_SEQUENCE_HEADER, parseObus, parseSequenceHeader } from './av1/obu.ts';
import { HevcDecoder } from './hevc/decode.ts';
import { nalsFromLengthPrefixed, parseHvcC } from './hevc/nal.ts';
import { assertDimensions, ResourceLimitError, resolveDecodeLimits } from './limits.ts';
import type { DecodeOptions, ResolvedDecodeLimits } from './limits.ts';

export { detectFormat, HeifFile } from './bmff.ts';
export type { Clap, ImageItem, NclxColor } from './bmff.ts';
export { DEFAULT_DECODE_LIMITS } from './limits.ts';
export type { DecodeOptions } from './limits.ts';

export type BinaryInput = ArrayBuffer | ArrayBufferView;

export interface DecodedImage {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export type DecodeErrorCode =
  | 'INVALID_INPUT'
  | 'UNSUPPORTED_FORMAT'
  | 'UNSUPPORTED_CODEC'
  | 'MISSING_CONFIG'
  | 'RESOURCE_LIMIT'
  | 'DECODE_FAILED'
  | 'IMAGE_BITMAP_UNAVAILABLE';

export class DecodeError extends Error {
  readonly code: DecodeErrorCode;
  readonly cause?: unknown;

  constructor(code: DecodeErrorCode, message: string, cause?: unknown) {
    super(message);
    this.name = 'DecodeError';
    this.code = code;
    this.cause = cause;
  }
}

/** Decode HEIC/HEIF/AVIF binary into an RGBA8 buffer. */
export function decodeToRgba(input: BinaryInput, options: DecodeOptions = {}): DecodedImage {
  const bytes = asUint8Array(input);
  if (bytes.byteLength < 12) throw new DecodeError('INVALID_INPUT', 'Image input is too short');
  const format = detectFormat(bytes);
  if (format === 'unknown') throw new DecodeError('UNSUPPORTED_FORMAT', 'Input is not a supported HEIC, HEIF, or AVIF file');

  let limits: ResolvedDecodeLimits;
  try {
    limits = resolveDecodeLimits(options);
  } catch (error) {
    throw new DecodeError('INVALID_INPUT', error instanceof Error ? error.message : 'Invalid decode options', error);
  }

  try {
    const file = new HeifFile().parse(bytes, limits);
    const context: ItemDecodeContext = { active: new Set(), limits, decodedPixels: 0 };
    return decodeItem(file, file.primary, context, true);
  } catch (error) {
    if (error instanceof DecodeError) throw error;
    if (error instanceof ResourceLimitError) {
      throw new DecodeError('RESOURCE_LIMIT', error.message, error);
    }
    throw new DecodeError('DECODE_FAILED', error instanceof Error ? error.message : 'Image decoding failed', error);
  }
}

/** Decode binary into browser ImageData. */
export function decodeToImageData(input: BinaryInput, options: DecodeOptions = {}): ImageData {
  if (typeof ImageData === 'undefined') {
    throw new DecodeError('IMAGE_BITMAP_UNAVAILABLE', 'ImageData is not available in this runtime');
  }
  const image = decodeToRgba(input, options);
  // DOM ImageData requires an ArrayBuffer-backed view (not SharedArrayBuffer).
  const pixels = new Uint8ClampedArray(image.data.length);
  pixels.set(image.data);
  return new ImageData(pixels, image.width, image.height);
}

/** Decode binary into the requested browser-native ImageBitmap result. */
export async function decode(input: BinaryInput, options: DecodeOptions = {}): Promise<ImageBitmap> {
  if (typeof createImageBitmap !== 'function') {
    throw new DecodeError('IMAGE_BITMAP_UNAVAILABLE', 'createImageBitmap is not available in this runtime');
  }
  return createImageBitmap(decodeToImageData(input, options));
}

export default decode;

interface ItemDecodeContext {
  active: Set<number>;
  limits: ResolvedDecodeLimits;
  decodedPixels: number;
}

function decodeItem(
  file: HeifFile, item: ImageItem, context: ItemDecodeContext, includeAlpha: boolean,
): DecodedImage {
  if (item.unsupportedEssentialProperties.length) {
    throw new DecodeError(
      'UNSUPPORTED_CODEC',
      `HEIF item ${item.itemId} requires unsupported properties: ${item.unsupportedEssentialProperties.join(', ')}`,
    );
  }
  if (context.active.has(item.itemId)) {
    throw new DecodeError('DECODE_FAILED', 'Cyclic HEIF item references are not supported');
  }
  if (item.width && item.height) {
    validateImageDimensions(item.width, item.height, context.limits, `HEIF item ${item.itemId}`);
    ensurePixelBudget(item.width * item.height, context, `HEIF item ${item.itemId}`);
  }
  if (context.active.size >= context.limits.maxReferenceDepth) {
    throw new DecodeError(
      'RESOURCE_LIMIT',
      `HEIF item-reference depth exceeds configured limit ${context.limits.maxReferenceDepth}`,
    );
  }
  context.active.add(item.itemId);
  let image: DecodedImage;
  try {
    if (item.type === 'hvc1' || item.type === 'hev1') image = decodeHevcItem(item, context.limits, includeAlpha);
    else if (item.type === 'av01') image = decodeAv1Item(item, context.limits, includeAlpha);
    else if (item.type === 'grid') image = decodeGridItem(file, item, context, includeAlpha);
    else if (item.type === 'iden') image = decodeIdentityItem(file, item, context, includeAlpha);
    else if (item.type === 'iovl') image = decodeOverlayItem(file, item, context, includeAlpha);
    else throw new DecodeError('UNSUPPORTED_CODEC', `Unsupported HEIF item codec: ${item.type}`);
    if ((item.width && image.width !== item.width) || (item.height && image.height !== item.height)) {
      throw new DecodeError(
        'DECODE_FAILED',
        `HEIF item ${item.itemId} decoded to ${image.width}x${image.height}, ` +
        `but its spatial-extents property declares ${item.width}x${item.height}`,
      );
    }
    image = applyItemTransforms(image, item);
    validateAndChargeImage(image, context, `HEIF item ${item.itemId}`);
    if (includeAlpha) {
      image = attachAuxiliaryAlpha(file, item, image, context);
      if ((item.references.prem?.length ?? 0) > 0) image = unpremultiplyAlpha(image);
    }
    return image;
  } finally {
    context.active.delete(item.itemId);
  }
}

function decodeAv1Item(
  item: ImageItem, limits: ResolvedDecodeLimits, colorManagement: boolean,
): DecodedImage {
  const decoded = new Av1Decoder(limits).decode(item.data);
  validateAv1Configuration(item, decoded.sequence);
  const matrix = item.nclx?.matrixCoefficients ?? decoded.sequence.matrixCoefficients;
  const fullRange = item.nclx?.fullRangeFlag ?? decoded.sequence.fullRange;
  const primaries = item.nclx?.colourPrimaries ?? decoded.sequence.colorPrimaries;
  const transfer = item.nclx?.transferCharacteristics ?? decoded.sequence.transferCharacteristics;
  return {
    width: decoded.frame.width,
    height: decoded.frame.height,
    data: frameToRgba(
      decoded.frame, matrix, fullRange, primaries, transfer, decoded.sequence.chromaSamplePosition,
      item.icc, colorManagement,
    ),
  };
}

function validateAv1Configuration(item: ImageItem, sequence: ReturnType<Av1Decoder['decode']>['sequence']): void {
  const configuration = item.config;
  if (!configuration) throw new DecodeError('MISSING_CONFIG', 'AV1 image item has no av1C configuration');
  if (configuration.length < 4 || !(configuration[0]! & 0x80) || (configuration[0]! & 0x7f) !== 1) {
    throw new DecodeError('DECODE_FAILED', 'AV1 image item has an invalid av1C configuration');
  }
  const profile = configuration[1]! >> 5;
  const level = configuration[1]! & 0x1f;
  const tier = configuration[2]! >> 7;
  const highBitDepth = (configuration[2]! >> 6) & 1;
  const twelveBit = (configuration[2]! >> 5) & 1;
  const bitDepth = highBitDepth ? (twelveBit ? 12 : 10) : 8;
  const monochrome = (configuration[2]! >> 4) & 1;
  const subsamplingX = (configuration[2]! >> 3) & 1;
  const subsamplingY = (configuration[2]! >> 2) & 1;
  const chromaSamplePosition = configuration[2]! & 3;
  if (profile !== sequence.profile || level !== sequence.seqLevelIdx[0] || tier !== sequence.seqTier[0] ||
      bitDepth !== sequence.bitDepth ||
      !!monochrome !== sequence.monochrome || subsamplingX !== sequence.subsamplingX ||
      subsamplingY !== sequence.subsamplingY ||
      (subsamplingX && subsamplingY && chromaSamplePosition !== sequence.chromaSamplePosition)) {
    throw new DecodeError('DECODE_FAILED', 'AV1 av1C configuration does not match the sequence header');
  }
  if ((configuration[3]! & 0xe0) || (!(configuration[3]! & 0x10) && (configuration[3]! & 0x0f))) {
    throw new DecodeError('DECODE_FAILED', 'AV1 image item has invalid reserved bits in av1C');
  }
  if (configuration.length > 4) {
    let configSequence: ReturnType<typeof parseSequenceHeader> | undefined;
    try {
      const sequenceHeaders = parseObus(configuration.subarray(4))
        .filter(obu => obu.type === OBU_SEQUENCE_HEADER);
      if (sequenceHeaders.length > 1) {
        throw new Error('av1C contains multiple sequence headers');
      }
      if (sequenceHeaders.length === 1) configSequence = parseSequenceHeader(sequenceHeaders[0]!.payload);
    } catch (error) {
      throw new DecodeError(
        'DECODE_FAILED', error instanceof Error ? `Invalid av1C configOBUs: ${error.message}` : 'Invalid av1C configOBUs', error,
      );
    }
    if (configSequence && JSON.stringify(configSequence) !== JSON.stringify(sequence)) {
      throw new DecodeError('DECODE_FAILED', 'AV1 sequence header in av1C does not match the image item');
    }
  }
}

function decodeGridItem(
  file: HeifFile, item: ImageItem, context: ItemDecodeContext, includeAlpha: boolean,
): DecodedImage {
  const rows = item.gridRows, columns = item.gridCols;
  const ids = item.gridTiles ?? [];
  if (!rows || !columns || ids.length < rows * columns) {
    throw new DecodeError('DECODE_FAILED', 'HEIF grid has missing tile references');
  }
  const firstItem = file.items.get(ids[0]!);
  if (!firstItem) throw new DecodeError('DECODE_FAILED', `HEIF grid tile ${ids[0]} is missing`);
  const first = decodeItem(file, firstItem, context, includeAlpha);
  const tileWidth = first.width, tileHeight = first.height;
  const naturalWidth = tileWidth * columns, naturalHeight = tileHeight * rows;
  if (!Number.isSafeInteger(naturalWidth) || !Number.isSafeInteger(naturalHeight)) {
    throw new DecodeError('RESOURCE_LIMIT', 'HEIF grid dimensions exceed JavaScript safe range');
  }
  const width = item.width || naturalWidth;
  const height = item.height || naturalHeight;
  validateImageDimensions(width, height, context.limits, 'HEIF grid');
  ensurePixelBudget(width * height, context, 'HEIF grid');
  const data = new Uint8ClampedArray(width * height * 4);

  const copyTile = (tile: DecodedImage, row: number, column: number): void => {
    if (tile.width !== tileWidth || tile.height !== tileHeight) {
      throw new DecodeError('DECODE_FAILED', 'HEIF grid tiles do not have uniform dimensions');
    }
    const x0 = column * tileWidth, y0 = row * tileHeight;
    const copyWidth = Math.max(0, Math.min(tile.width, width - x0));
    const copyHeight = Math.max(0, Math.min(tile.height, height - y0));
    for (let y = 0; y < copyHeight; y++) {
      data.set(tile.data.subarray(y * tile.width * 4, (y * tile.width + copyWidth) * 4),
        ((y0 + y) * width + x0) * 4);
    }
  };

  copyTile(first, 0, 0);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      if (row === 0 && column === 0) continue;
      const id = ids[row * columns + column]!;
      const tileItem = file.items.get(id);
      if (!tileItem) throw new DecodeError('DECODE_FAILED', `HEIF grid tile ${id} is missing`);
      copyTile(decodeItem(file, tileItem, context, includeAlpha), row, column);
    }
  }
  return { width, height, data };
}

function decodeIdentityItem(
  file: HeifFile, item: ImageItem, context: ItemDecodeContext, includeAlpha: boolean,
): DecodedImage {
  const ids = item.references.dimg ?? [];
  if (ids.length !== 1) {
    throw new DecodeError('DECODE_FAILED', `HEIF identity item must have exactly one dimg reference (found ${ids.length})`);
  }
  const source = file.items.get(ids[0]!);
  if (!source) throw new DecodeError('DECODE_FAILED', `HEIF identity source ${ids[0]} is missing`);
  return decodeItem(file, source, context, includeAlpha);
}

function decodeOverlayItem(
  file: HeifFile, item: ImageItem, context: ItemDecodeContext, includeAlpha: boolean,
): DecodedImage {
  const ids = item.references.dimg ?? [];
  if (ids.length === 0) throw new DecodeError('DECODE_FAILED', 'HEIF overlay has no dimg references');
  const bytes = item.data;
  if (bytes.length < 10) throw new DecodeError('DECODE_FAILED', 'HEIF overlay header is truncated');
  if (bytes[0] !== 0) throw new DecodeError('UNSUPPORTED_CODEC', `Unsupported HEIF overlay version ${bytes[0]}`);
  const fieldLength = (bytes[1]! & 1) ? 4 : 2;
  const required = 10 + fieldLength * (2 + ids.length * 2);
  if (bytes.length < required) throw new DecodeError('DECODE_FAILED', 'HEIF overlay offsets are truncated');

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let position = 2;
  const background = new Uint8Array(4);
  for (let component = 0; component < 4; component++, position += 2) {
    // HEIF stores overlay background components as unsigned 16-bit values.
    background[component] = view.getUint16(position) >>> 8;
  }
  const readUnsigned = (): number => {
    const value = fieldLength === 2 ? view.getUint16(position) : view.getUint32(position);
    position += fieldLength;
    return value;
  };
  const readSigned = (): number => {
    const value = fieldLength === 2 ? view.getInt16(position) : view.getInt32(position);
    position += fieldLength;
    return value;
  };
  const width = readUnsigned(), height = readUnsigned();
  validateImageDimensions(width, height, context.limits, 'HEIF overlay canvas');
  ensurePixelBudget(width * height, context, 'HEIF overlay canvas');
  if ((item.width && item.width !== width) || (item.height && item.height !== height)) {
    throw new DecodeError('DECODE_FAILED', 'HEIF overlay canvas size disagrees with its ispe property');
  }

  // libheif's iovl output is an opaque RGB canvas. The fourth background
  // component is retained by the file format but does not create an alpha plane.
  const data = new Uint8ClampedArray(width * height * 4);
  for (let pixel = 0; pixel < width * height; pixel++) {
    const offset = pixel * 4;
    data[offset] = background[0]!;
    data[offset + 1] = background[1]!;
    data[offset + 2] = background[2]!;
    data[offset + 3] = 255;
  }

  for (const id of ids) {
    const x = readSigned(), y = readSigned();
    const sourceItem = file.items.get(id);
    if (!sourceItem) throw new DecodeError('DECODE_FAILED', `HEIF overlay source ${id} is missing`);
    const source = decodeItem(file, sourceItem, context, includeAlpha);
    compositeOver(data, width, height, source, x, y);
  }
  return { width, height, data };
}

function compositeOver(
  destination: Uint8ClampedArray, width: number, height: number,
  source: DecodedImage, offsetX: number, offsetY: number,
): void {
  const sourceX = Math.max(0, -offsetX), sourceY = Math.max(0, -offsetY);
  const destinationX = Math.max(0, offsetX), destinationY = Math.max(0, offsetY);
  const copyWidth = Math.min(source.width - sourceX, width - destinationX);
  const copyHeight = Math.min(source.height - sourceY, height - destinationY);
  if (copyWidth <= 0 || copyHeight <= 0) return;
  for (let y = 0; y < copyHeight; y++) {
    for (let x = 0; x < copyWidth; x++) {
      const sourceOffset = ((sourceY + y) * source.width + sourceX + x) * 4;
      const destinationOffset = ((destinationY + y) * width + destinationX + x) * 4;
      const alpha = source.data[sourceOffset + 3]!;
      if (alpha === 255) {
        destination[destinationOffset] = source.data[sourceOffset]!;
        destination[destinationOffset + 1] = source.data[sourceOffset + 1]!;
        destination[destinationOffset + 2] = source.data[sourceOffset + 2]!;
      } else if (alpha !== 0) {
        const inverse = 255 - alpha;
        for (let component = 0; component < 3; component++) {
          destination[destinationOffset + component] =
            (source.data[sourceOffset + component]! * alpha + destination[destinationOffset + component]! * inverse) / 255;
        }
      }
    }
  }
}

function attachAuxiliaryAlpha(
  file: HeifFile, item: ImageItem, image: DecodedImage, context: ItemDecodeContext,
): DecodedImage {
  let alphaItem: ImageItem | undefined;
  for (const candidate of file.items.values()) {
    const pointsToItem = candidate.references.auxl?.includes(item.itemId);
    const itemPointsToCandidate = item.references.auxl?.includes(candidate.itemId);
    if ((pointsToItem || itemPointsToCandidate) &&
        (!candidate.auxType || /alpha|auxid:1/i.test(candidate.auxType))) {
      alphaItem = candidate;
      break;
    }
  }
  if (!alphaItem) return image;
  const alpha = decodeItem(file, alphaItem, context, false);
  const data = new Uint8ClampedArray(image.data);
  for (let y = 0; y < image.height; y++) {
    const ay = Math.min(alpha.height - 1, Math.floor(y * alpha.height / image.height));
    for (let x = 0; x < image.width; x++) {
      const ax = Math.min(alpha.width - 1, Math.floor(x * alpha.width / image.width));
      data[(y * image.width + x) * 4 + 3] = alpha.data[(ay * alpha.width + ax) * 4]!;
    }
  }
  return { width: image.width, height: image.height, data };
}

function unpremultiplyAlpha(image: DecodedImage): DecodedImage {
  const data = new Uint8ClampedArray(image.data);
  for (let offset = 0; offset < data.length; offset += 4) {
    const alpha = data[offset + 3]!;
    if (alpha === 0) {
      data[offset] = data[offset + 1] = data[offset + 2] = 0;
    } else if (alpha !== 255) {
      data[offset] = Math.min(255, Math.round(data[offset]! * 255 / alpha));
      data[offset + 1] = Math.min(255, Math.round(data[offset + 1]! * 255 / alpha));
      data[offset + 2] = Math.min(255, Math.round(data[offset + 2]! * 255 / alpha));
    }
  }
  return { width: image.width, height: image.height, data };
}

function applyItemTransforms(image: DecodedImage, item: ImageItem): DecodedImage {
  let result = image;
  for (const transform of item.transformations) {
    if (transform === 'clap' && item.clap) result = cropCleanAperture(result, item.clap);
    else if (transform === 'imir') result = mirrorImage(result, item.imir);
    else if (transform === 'irot' && item.irot) result = rotateImageCcw(result, item.irot);
  }
  return result;
}

function cropCleanAperture(image: DecodedImage, clap: NonNullable<ImageItem['clap']>): DecodedImage {
  if (!clap.cleanApertureWidthD || !clap.cleanApertureHeightD || !clap.horizOffD || !clap.vertOffD) {
    throw new DecodeError('DECODE_FAILED', 'HEIF clean-aperture property has a zero denominator');
  }
  const cleanWidth = Math.max(1, Math.round(clap.cleanApertureWidthN / clap.cleanApertureWidthD));
  const cleanHeight = Math.max(1, Math.round(clap.cleanApertureHeightN / clap.cleanApertureHeightD));
  const centerX = clap.horizOffN / clap.horizOffD + (image.width - 1) / 2;
  const centerY = clap.vertOffN / clap.vertOffD + (image.height - 1) / 2;
  const left = clampInt(Math.floor(centerX - (cleanWidth - 1) / 2), 0, Math.max(0, image.width - cleanWidth));
  // HEIF clap rounds the vertical origin to nearest, while the horizontal
  // origin is rounded down (ISO BMFF clean-aperture convention).
  const top = clampInt(Math.round(centerY - (cleanHeight - 1) / 2), 0, Math.max(0, image.height - cleanHeight));
  const width = Math.min(cleanWidth, image.width - left), height = Math.min(cleanHeight, image.height - top);
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    data.set(image.data.subarray(((top + y) * image.width + left) * 4,
      ((top + y) * image.width + left + width) * 4), y * width * 4);
  }
  return { width, height, data };
}

function mirrorImage(image: DecodedImage, direction: number): DecodedImage {
  const data = new Uint8ClampedArray(image.data.length);
  for (let y = 0; y < image.height; y++) {
    for (let x = 0; x < image.width; x++) {
      const dx = direction === 1 ? image.width - 1 - x : x;
      const dy = direction === 0 ? image.height - 1 - y : y;
      data.set(image.data.subarray((y * image.width + x) * 4, (y * image.width + x + 1) * 4),
        (dy * image.width + dx) * 4);
    }
  }
  return { width: image.width, height: image.height, data };
}

function rotateImageCcw(image: DecodedImage, quarterTurns: number): DecodedImage {
  const turns = quarterTurns & 3;
  if (!turns) return image;
  const width = turns & 1 ? image.height : image.width;
  const height = turns & 1 ? image.width : image.height;
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < image.height; y++) {
    for (let x = 0; x < image.width; x++) {
      let dx: number, dy: number;
      if (turns === 1) { dx = y; dy = image.width - 1 - x; }
      else if (turns === 2) { dx = image.width - 1 - x; dy = image.height - 1 - y; }
      else { dx = image.height - 1 - y; dy = x; }
      data.set(image.data.subarray((y * image.width + x) * 4, (y * image.width + x + 1) * 4),
        (dy * width + dx) * 4);
    }
  }
  return { width, height, data };
}

function clampInt(value: number, minimum: number, maximum: number): number {
  return value < minimum ? minimum : value > maximum ? maximum : value;
}

function decodeHevcItem(
  item: ImageItem, limits: ResolvedDecodeLimits, colorManagement: boolean,
): DecodedImage {
  if (!item.config) throw new DecodeError('MISSING_CONFIG', 'HEVC image item has no hvcC configuration');
  const config = parseHvcC(item.config);
  const itemNals = nalsFromLengthPrefixed(item.data, config.lengthSize);
  const decoder = new HevcDecoder(limits);
  decoder.registerParamSets(config.paramSets);
  decoder.registerParamSets(itemNals);
  const slices = itemNals.filter(nal => nal.type <= 31);
  if (slices.length === 0) throw new DecodeError('DECODE_FAILED', 'HEVC image item contains no slice NAL unit');
  const frame = decoder.decodeFrame(slices);
  const sps = decoder.decodedSps;
  const matrix = item.nclx?.matrixCoefficients ?? sps?.matrixCoefficients ?? 6;
  const fullRange = item.nclx?.fullRangeFlag ?? sps?.fullRange ?? false;
  const primaries = item.nclx?.colourPrimaries ?? sps?.colourPrimaries ?? 2;
  const transfer = item.nclx?.transferCharacteristics ?? sps?.transferCharacteristics ?? 2;
  // Translate HEVC chroma_sample_loc_type to the internal AV1-compatible
  // co-sited/centered modes. Field-specific types 4/5 use the closest
  // progressive-frame location.
  const chromaLocation = [1, 3, 2, 4, 1, 2][sps?.chromaSampleLocation ?? 0] ?? 1;
  let image: DecodedImage = {
    width: frame.width, height: frame.height,
    data: frameToRgba(
      frame, matrix, fullRange, primaries, transfer, chromaLocation, item.icc, colorManagement,
    ),
  };
  if (sps?.conformance) {
    const subWidth = sps.chromaFormatIdc === 1 || sps.chromaFormatIdc === 2 ? 2 : 1;
    const subHeight = sps.chromaFormatIdc === 1 ? 2 : 1;
    const left = sps.conformance.left * subWidth;
    const right = sps.conformance.right * subWidth;
    const top = sps.conformance.top * subHeight;
    const bottom = sps.conformance.bottom * subHeight;
    if (left + right >= image.width || top + bottom >= image.height) {
      throw new DecodeError('DECODE_FAILED', 'HEVC conformance window is outside the coded picture');
    }
    image = cropImage(image, left, top, image.width - left - right, image.height - top - bottom);
  }
  return image;
}

function validateImageDimensions(
  width: number, height: number, limits: ResolvedDecodeLimits, label: string,
): void {
  try {
    assertDimensions(width, height, limits, label);
  } catch (error) {
    if (error instanceof ResourceLimitError) {
      throw new DecodeError('RESOURCE_LIMIT', error.message, error);
    }
    throw new DecodeError('DECODE_FAILED', error instanceof Error ? error.message : `${label} has invalid dimensions`, error);
  }
}

function cropImage(
  image: DecodedImage, left: number, top: number, width: number, height: number,
): DecodedImage {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    const start = ((top + y) * image.width + left) * 4;
    data.set(image.data.subarray(start, start + width * 4), y * width * 4);
  }
  return { width, height, data };
}

function validateAndChargeImage(image: DecodedImage, context: ItemDecodeContext, label: string): void {
  validateImageDimensions(image.width, image.height, context.limits, label);
  const pixels = image.width * image.height;
  ensurePixelBudget(pixels, context, label);
  context.decodedPixels += pixels;
}

function ensurePixelBudget(pixels: number, context: ItemDecodeContext, label: string): void {
  if (pixels > context.limits.maxTotalPixels - context.decodedPixels) {
    throw new DecodeError(
      'RESOURCE_LIMIT',
      `${label} exceeds the configured cumulative decode limit ${context.limits.maxTotalPixels} pixels`,
    );
  }
}

function asUint8Array(input: BinaryInput): Uint8Array {
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  throw new DecodeError('INVALID_INPUT', 'Expected an ArrayBuffer or typed-array view');
}
