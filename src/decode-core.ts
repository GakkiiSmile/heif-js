import { detectFormat, HeifFile } from './bmff.ts';
import type { ImageItem } from './bmff.ts';
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

export interface DecodedItemImage extends DecodedImage {
  /** One byte for an auxiliary alpha decode, four bytes for public RGBA output. */
  channels: 1 | 4;
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

export type ItemCodecDecoder = (
  item: ImageItem, limits: ResolvedDecodeLimits, colorManagement: boolean,
  output?: Uint8ClampedArray,
) => DecodedItemImage;

export interface CodecDecoders {
  hevc?: ItemCodecDecoder;
  av1?: ItemCodecDecoder;
}

export interface DecoderApi {
  decodeToRgba(input: BinaryInput, options?: DecodeOptions): DecodedImage;
  decodeToImageData(input: BinaryInput, options?: DecodeOptions): ImageData;
  decode(input: BinaryInput, options?: DecodeOptions): Promise<ImageBitmap>;
}

/** Bind the shared HEIF container/derived-image pipeline to one or more codecs. */
export function createDecoder(codecs: Readonly<CodecDecoders>): DecoderApi {
  const decodeToRgba = (input: BinaryInput, options: DecodeOptions = {}): DecodedImage =>
    decodeToRgbaWithCodecs(input, options, codecs);
  const decodeToImageData = (input: BinaryInput, options: DecodeOptions = {}): ImageData => {
    if (typeof ImageData === 'undefined') {
      throw new DecodeError('IMAGE_BITMAP_UNAVAILABLE', 'ImageData is not available in this runtime');
    }
    const image = decodeToRgba(input, options);
    // Internal decode buffers are ArrayBuffer-backed. Keep the fallback for a
    // future alternate decoder returning a SharedArrayBuffer view.
    const pixels = image.data.buffer instanceof ArrayBuffer
      ? image.data as Uint8ClampedArray<ArrayBuffer>
      : new Uint8ClampedArray(image.data);
    return new ImageData(pixels, image.width, image.height);
  };
  const decode = async (input: BinaryInput, options: DecodeOptions = {}): Promise<ImageBitmap> => {
    if (typeof createImageBitmap !== 'function') {
      throw new DecodeError('IMAGE_BITMAP_UNAVAILABLE', 'createImageBitmap is not available in this runtime');
    }
    return createImageBitmap(decodeToImageData(input, options));
  };
  return { decodeToRgba, decodeToImageData, decode };
}

function decodeToRgbaWithCodecs(
  input: BinaryInput, options: DecodeOptions, codecs: Readonly<CodecDecoders>,
): DecodedImage {
  const bytes = asUint8Array(input);
  const output = options.output;
  if (output !== undefined && !(output instanceof Uint8ClampedArray)) {
    throw new DecodeError('INVALID_INPUT', 'output must be a Uint8ClampedArray');
  }
  if (output && viewsOverlap(bytes, output)) {
    throw new DecodeError('INVALID_INPUT', 'output must not overlap the encoded input');
  }
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
    const context: ItemDecodeContext = {
      active: new Set(), cache: new Map(), cacheable: findCacheableItems(file), codecs, limits, decodedPixels: 0,
    };
    const image = decodeItem(file, file.primary, context, true, output);
    if (image.channels !== 4) throw new DecodeError('DECODE_FAILED', 'Primary image did not decode to RGBA');
    // Keep the public object shape stable; `channels` is internal cache metadata.
    return { width: image.width, height: image.height, data: image.data };
  } catch (error) {
    if (error instanceof DecodeError) throw error;
    if (error instanceof ResourceLimitError) {
      throw new DecodeError('RESOURCE_LIMIT', error.message, error);
    }
    throw new DecodeError('DECODE_FAILED', error instanceof Error ? error.message : 'Image decoding failed', error);
  }
}

interface ItemDecodeContext {
  active: Set<number>;
  cache: Map<string, DecodedItemImage>;
  cacheable: Set<number>;
  codecs: Readonly<CodecDecoders>;
  limits: ResolvedDecodeLimits;
  decodedPixels: number;
}

function decodeItem(
  file: HeifFile, item: ImageItem, context: ItemDecodeContext, includeAlpha: boolean,
  output?: Uint8ClampedArray,
): DecodedItemImage {
  const cacheKey = `${item.itemId}:${+includeAlpha}`;
  if (context.cacheable.has(item.itemId)) {
    const cached = context.cache.get(cacheKey);
    if (cached) return cached;
  }
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
  let image: DecodedItemImage;
  let mutable = item.type !== 'iden';
  try {
    const directOutput = output && includeAlpha && item.transformations.length === 0 ? output : undefined;
    const codecOutput = directOutput &&
      item.width > 0 && item.height > 0 && item.width * item.height * 4 === directOutput.length
      ? directOutput : undefined;
    if (item.type === 'hvc1' || item.type === 'hev1') {
      if (!context.codecs.hevc) {
        throw new DecodeError('UNSUPPORTED_CODEC', 'HEVC image item requires the HEIC/HEVC decoder entry');
      }
      image = context.codecs.hevc(item, context.limits, includeAlpha, codecOutput);
    } else if (item.type === 'av01') {
      if (!context.codecs.av1) {
        throw new DecodeError('UNSUPPORTED_CODEC', 'AV1 image item requires the AVIF/AV1 decoder entry');
      }
      image = context.codecs.av1(item, context.limits, includeAlpha, codecOutput);
    }
    else if (item.type === 'grid') image = decodeGridItem(file, item, context, includeAlpha, directOutput);
    else if (item.type === 'iden') image = decodeIdentityItem(file, item, context, includeAlpha, directOutput);
    else if (item.type === 'iovl') image = decodeOverlayItem(file, item, context, includeAlpha, directOutput);
    else throw new DecodeError('UNSUPPORTED_CODEC', `Unsupported HEIF item codec: ${item.type}`);
    if ((item.width && image.width !== item.width) || (item.height && image.height !== item.height)) {
      throw new DecodeError(
        'DECODE_FAILED',
        `HEIF item ${item.itemId} decoded to ${image.width}x${image.height}, ` +
        `but its spatial-extents property declares ${item.width}x${item.height}`,
      );
    }
    const transformed = applyItemTransforms(image, item);
    if (transformed !== image) mutable = true;
    image = transformed;
    validateAndChargeImage(image, context, `HEIF item ${item.itemId}`);
    if (includeAlpha) {
      const unpremultiply = (item.references.prem?.length ?? 0) > 0;
      const attached = attachAuxiliaryAlpha(file, item, image, context, mutable, unpremultiply);
      image = attached.image;
      mutable = attached.mutable;
      if (unpremultiply && !attached.attached) {
        const unpremultiplied = unpremultiplyAlpha(image, mutable);
        image = unpremultiplied.image;
        mutable = unpremultiplied.mutable;
      }
    }
    if (output && image.data !== output) {
      if (output.length !== image.data.length) {
        throw new DecodeError(
          'INVALID_INPUT',
          `output length ${output.length} does not match required length ${image.data.length}`,
        );
      }
      output.set(image.data);
      image = { width: image.width, height: image.height, data: output, channels: image.channels };
      mutable = true;
    }
    if (context.cacheable.has(item.itemId)) context.cache.set(cacheKey, image);
    return image;
  } finally {
    context.active.delete(item.itemId);
  }
}

function viewsOverlap(input: Uint8Array, output: Uint8ClampedArray): boolean {
  if (input.buffer !== output.buffer) return false;
  const inputEnd = input.byteOffset + input.byteLength;
  const outputEnd = output.byteOffset + output.byteLength;
  return input.byteOffset < outputEnd && output.byteOffset < inputEnd;
}

function findCacheableItems(file: HeifFile): Set<number> {
  const incoming = new Map<number, number>();
  incoming.set(file.primaryItemId, 1);
  for (const item of file.items.values()) {
    for (const ids of Object.values(item.references)) {
      for (const id of ids) incoming.set(id, (incoming.get(id) ?? 0) + 1);
    }
  }
  return new Set([...incoming].filter(([, count]) => count > 1).map(([id]) => id));
}

function decodeGridItem(
  file: HeifFile, item: ImageItem, context: ItemDecodeContext, includeAlpha: boolean,
  output?: Uint8ClampedArray,
): DecodedItemImage {
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
  const channels = first.channels;
  validateImageDimensions(width, height, context.limits, 'HEIF grid');
  ensurePixelBudget(width * height, context, 'HEIF grid');
  const outputLength = width * height * channels;
  const data = output?.length === outputLength ? output : new Uint8ClampedArray(outputLength);

  const copyTile = (tile: DecodedItemImage, row: number, column: number): void => {
    if (tile.width !== tileWidth || tile.height !== tileHeight || tile.channels !== channels) {
      throw new DecodeError('DECODE_FAILED', 'HEIF grid tiles do not have uniform dimensions');
    }
    const x0 = column * tileWidth, y0 = row * tileHeight;
    const copyWidth = Math.max(0, Math.min(tile.width, width - x0));
    const copyHeight = Math.max(0, Math.min(tile.height, height - y0));
    for (let y = 0; y < copyHeight; y++) {
      data.set(tile.data.subarray(y * tile.width * channels, (y * tile.width + copyWidth) * channels),
        ((y0 + y) * width + x0) * channels);
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
  return { width, height, data, channels };
}

function decodeIdentityItem(
  file: HeifFile, item: ImageItem, context: ItemDecodeContext, includeAlpha: boolean,
  output?: Uint8ClampedArray,
): DecodedItemImage {
  const ids = item.references.dimg ?? [];
  if (ids.length !== 1) {
    throw new DecodeError('DECODE_FAILED', `HEIF identity item must have exactly one dimg reference (found ${ids.length})`);
  }
  const source = file.items.get(ids[0]!);
  if (!source) throw new DecodeError('DECODE_FAILED', `HEIF identity source ${ids[0]} is missing`);
  return decodeItem(file, source, context, includeAlpha, output);
}

function decodeOverlayItem(
  file: HeifFile, item: ImageItem, context: ItemDecodeContext, includeAlpha: boolean,
  output?: Uint8ClampedArray,
): DecodedItemImage {
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
  const channels = includeAlpha ? 4 : 1;
  const outputLength = width * height * channels;
  const data = output?.length === outputLength ? output : new Uint8ClampedArray(outputLength);
  if (channels === 1) {
    data.fill(background[0]!);
  } else {
    for (let pixel = 0; pixel < width * height; pixel++) {
      const offset = pixel * 4;
      data[offset] = background[0]!;
      data[offset + 1] = background[1]!;
      data[offset + 2] = background[2]!;
      data[offset + 3] = 255;
    }
  }

  for (const id of ids) {
    const x = readSigned(), y = readSigned();
    const sourceItem = file.items.get(id);
    if (!sourceItem) throw new DecodeError('DECODE_FAILED', `HEIF overlay source ${id} is missing`);
    const source = decodeItem(file, sourceItem, context, includeAlpha);
    compositeOver(data, width, height, channels, source, x, y);
  }
  return { width, height, data, channels };
}

function compositeOver(
  destination: Uint8ClampedArray, width: number, height: number,
  channels: 1 | 4, source: DecodedItemImage, offsetX: number, offsetY: number,
): void {
  const sourceX = Math.max(0, -offsetX), sourceY = Math.max(0, -offsetY);
  const destinationX = Math.max(0, offsetX), destinationY = Math.max(0, offsetY);
  const copyWidth = Math.min(source.width - sourceX, width - destinationX);
  const copyHeight = Math.min(source.height - sourceY, height - destinationY);
  if (copyWidth <= 0 || copyHeight <= 0) return;
  if (channels === 1) {
    if (source.channels !== 1) throw new DecodeError('DECODE_FAILED', 'HEIF overlay channel layouts disagree');
    for (let y = 0; y < copyHeight; y++) {
      const sourceOffset = (sourceY + y) * source.width + sourceX;
      const destinationOffset = (destinationY + y) * width + destinationX;
      destination.set(source.data.subarray(sourceOffset, sourceOffset + copyWidth), destinationOffset);
    }
    return;
  }
  if (source.channels !== 4) throw new DecodeError('DECODE_FAILED', 'HEIF overlay channel layouts disagree');
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
  file: HeifFile, item: ImageItem, image: DecodedItemImage, context: ItemDecodeContext,
  mutable: boolean, unpremultiply: boolean,
): { image: DecodedItemImage; mutable: boolean; attached: boolean } {
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
  if (!alphaItem) return { image, mutable, attached: false };
  const alpha = decodeItem(file, alphaItem, context, false);
  if (image.channels !== 4) throw new DecodeError('DECODE_FAILED', 'Cannot attach alpha to a non-RGBA image');
  let result = image;
  if (!mutable) {
    result = { width: image.width, height: image.height, data: new Uint8ClampedArray(image.data), channels: 4 };
    mutable = true;
  }
  const alphaChannels = alpha.channels;
  const xIndices = new Uint32Array(result.width);
  for (let x = 0; x < result.width; x++) {
    xIndices[x] = Math.min(alpha.width - 1, Math.floor(x * alpha.width / result.width));
  }
  for (let y = 0; y < image.height; y++) {
    const ay = Math.min(alpha.height - 1, Math.floor(y * alpha.height / image.height));
    const alphaRow = ay * alpha.width * alphaChannels;
    for (let x = 0; x < image.width; x++) {
      const offset = (y * image.width + x) * 4;
      const value = alpha.data[alphaRow + xIndices[x]! * alphaChannels]!;
      result.data[offset + 3] = value;
      if (unpremultiply) unpremultiplyPixel(result.data, offset, value);
    }
  }
  return { image: result, mutable, attached: true };
}

function unpremultiplyAlpha(
  image: DecodedItemImage, mutable: boolean,
): { image: DecodedItemImage; mutable: boolean } {
  if (image.channels !== 4) return { image, mutable };
  let first = -1;
  for (let offset = 3; offset < image.data.length; offset += 4) {
    if (image.data[offset] !== 255) { first = offset - 3; break; }
  }
  if (first < 0) return { image, mutable };
  let result = image;
  if (!mutable) {
    result = { width: image.width, height: image.height, data: new Uint8ClampedArray(image.data), channels: 4 };
    mutable = true;
  }
  for (let offset = first; offset < result.data.length; offset += 4) {
    unpremultiplyPixel(result.data, offset, result.data[offset + 3]!);
  }
  return { image: result, mutable };
}

function unpremultiplyPixel(data: Uint8ClampedArray, offset: number, alpha: number): void {
  if (alpha === 0) {
    data[offset] = data[offset + 1] = data[offset + 2] = 0;
  } else if (alpha !== 255) {
    data[offset] = Math.min(255, Math.round(data[offset]! * 255 / alpha));
    data[offset + 1] = Math.min(255, Math.round(data[offset + 1]! * 255 / alpha));
    data[offset + 2] = Math.min(255, Math.round(data[offset + 2]! * 255 / alpha));
  }
}

type GeometryOperation = readonly ['mirror' | 'rotate', number];

function applyItemTransforms(image: DecodedItemImage, item: ImageItem): DecodedItemImage {
  let result = image;
  let geometry: GeometryOperation[] = [];
  const flushGeometry = (): void => {
    if (geometry.length) result = transformImage(result, geometry);
    geometry = [];
  };
  for (const transform of item.transformations) {
    if (transform === 'clap' && item.clap) {
      flushGeometry();
      result = cropCleanAperture(result, item.clap);
    } else if (transform === 'imir') geometry.push(['mirror', item.imir]);
    else if (transform === 'irot' && item.irot) geometry.push(['rotate', item.irot]);
  }
  flushGeometry();
  return result;
}

function cropCleanAperture(image: DecodedItemImage, clap: NonNullable<ImageItem['clap']>): DecodedItemImage {
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
  if (left === 0 && top === 0 && width === image.width && height === image.height) return image;
  const channels = image.channels;
  const data = new Uint8ClampedArray(width * height * channels);
  for (let y = 0; y < height; y++) {
    data.set(image.data.subarray(((top + y) * image.width + left) * channels,
      ((top + y) * image.width + left + width) * channels), y * width * channels);
  }
  return { width, height, data, channels };
}

function transformImage(image: DecodedItemImage, operations: readonly GeometryOperation[]): DecodedItemImage {
  let a = 1, b = 0, c = 0, d = 0, e = 1, f = 0;
  let width = image.width, height = image.height;
  for (const [kind, value] of operations) {
    if (kind === 'mirror') {
      if (value === 1) { a = -a; b = -b; c = width - 1 - c; }
      else { d = -d; e = -e; f = height - 1 - f; }
      continue;
    }
    const turns = value & 3;
    const oldA = a, oldB = b, oldC = c, oldD = d, oldE = e, oldF = f;
    if (turns === 1) {
      a = oldD; b = oldE; c = oldF;
      d = -oldA; e = -oldB; f = width - 1 - oldC;
      [width, height] = [height, width];
    } else if (turns === 2) {
      a = -oldA; b = -oldB; c = width - 1 - oldC;
      d = -oldD; e = -oldE; f = height - 1 - oldF;
    } else if (turns === 3) {
      a = -oldD; b = -oldE; c = height - 1 - oldF;
      d = oldA; e = oldB; f = oldC;
      [width, height] = [height, width];
    }
  }
  if (a === 1 && b === 0 && c === 0 && d === 0 && e === 1 && f === 0 &&
      width === image.width && height === image.height) return image;

  const data = new Uint8ClampedArray(image.data.length);
  if (image.channels === 4 && !(image.data.byteOffset & 3)) {
    const source = new Uint32Array(image.data.buffer, image.data.byteOffset, image.data.length >> 2);
    const destination = new Uint32Array(data.buffer);
    for (let y = 0; y < image.height; y++) {
      for (let x = 0; x < image.width; x++) {
        const dx = a * x + b * y + c, dy = d * x + e * y + f;
        destination[dy * width + dx] = source[y * image.width + x]!;
      }
    }
  } else {
    const channels = image.channels;
    for (let y = 0; y < image.height; y++) {
      for (let x = 0; x < image.width; x++) {
        const dx = a * x + b * y + c, dy = d * x + e * y + f;
        const source = (y * image.width + x) * channels;
        const destination = (dy * width + dx) * channels;
        for (let channel = 0; channel < channels; channel++) data[destination + channel] = image.data[source + channel]!;
      }
    }
  }
  return { width, height, data, channels: image.channels };
}

function clampInt(value: number, minimum: number, maximum: number): number {
  return value < minimum ? minimum : value > maximum ? maximum : value;
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

function validateAndChargeImage(image: DecodedItemImage, context: ItemDecodeContext, label: string): void {
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
  if (ArrayBuffer.isView(input)) {
    const view = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    return input.buffer instanceof ArrayBuffer ? view : new Uint8Array(view);
  }
  throw new DecodeError('INVALID_INPUT', 'Expected an ArrayBuffer or typed-array view');
}
