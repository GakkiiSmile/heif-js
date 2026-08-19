/** Resource limits applied while parsing and decoding untrusted image data. */
export interface DecodeOptions {
  /** Maximum width or height of any decoded/derived image. Default: 65,536. */
  maxDimension?: number;
  /** Maximum pixels in one decoded/derived image. Default: 64 Mi pixels. */
  maxPixels?: number;
  /** Maximum cumulative pixels decoded through item references. Default: 128 Mi pixels. */
  maxTotalPixels?: number;
  /** Maximum number of HEIF items. Default: 4,096. */
  maxItems?: number;
  /** Maximum number of properties in one HEIF property container. Default: 65,536. */
  maxProperties?: number;
  /** Maximum extents used to assemble one HEIF item. Default: 65,536. */
  maxExtentsPerItem?: number;
  /** Maximum assembled encoded bytes for one HEIF item. Default: 256 MiB. */
  maxItemBytes?: number;
  /** Maximum boxes in any parsed ISO-BMFF box list. Default: 262,144. */
  maxBoxes?: number;
  /** Maximum recursive derived/auxiliary-item depth. Default: 128. */
  maxReferenceDepth?: number;
}

export interface ResolvedDecodeLimits {
  maxDimension: number;
  maxPixels: number;
  maxTotalPixels: number;
  maxItems: number;
  maxProperties: number;
  maxExtentsPerItem: number;
  maxItemBytes: number;
  maxBoxes: number;
  maxReferenceDepth: number;
}

export const DEFAULT_DECODE_LIMITS: Readonly<ResolvedDecodeLimits> = Object.freeze({
  maxDimension: 65_536,
  maxPixels: 64 * 1024 * 1024,
  maxTotalPixels: 128 * 1024 * 1024,
  maxItems: 4_096,
  maxProperties: 65_536,
  maxExtentsPerItem: 65_536,
  maxItemBytes: 256 * 1024 * 1024,
  maxBoxes: 262_144,
  maxReferenceDepth: 128,
});

export class ResourceLimitError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ResourceLimitError';
  }
}

export function resolveDecodeLimits(options: DecodeOptions = {}): ResolvedDecodeLimits {
  const result = { ...DEFAULT_DECODE_LIMITS };
  for (const key of Object.keys(result) as (keyof ResolvedDecodeLimits)[]) {
    const supplied = options[key];
    if (supplied === undefined) continue;
    if (!Number.isSafeInteger(supplied) || supplied <= 0) {
      throw new TypeError(`${key} must be a positive safe integer`);
    }
    result[key] = supplied;
  }
  return result;
}

export function assertDimensions(
  width: number, height: number, limits: Pick<ResolvedDecodeLimits, 'maxDimension' | 'maxPixels'>,
  label: string,
): void {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0) {
    throw new Error(`${label} has invalid dimensions ${width}x${height}`);
  }
  if (width > limits.maxDimension || height > limits.maxDimension || width > Math.floor(limits.maxPixels / height)) {
    throw new ResourceLimitError(
      `${label} dimensions ${width}x${height} exceed the configured limit ` +
      `(${limits.maxDimension} per dimension, ${limits.maxPixels} pixels)`,
    );
  }
  if (width > Math.floor(0x3fffffff / height)) {
    throw new ResourceLimitError(`${label} is too large for an RGBA8 output buffer`);
  }
}
