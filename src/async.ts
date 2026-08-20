import { detectFormat, HeifFile } from './bmff.ts';
import type { BinaryInput, DecodeOptions, DecodedImage } from './decode-core.ts';

export type {
  BinaryInput, DecodeOptions, DecodedImage,
} from './decode-core.ts';

type DecoderModule = Pick<typeof import('./index.ts'), 'decode'>;

/** Decode after dynamically loading only the codec graph required by the input. */
export async function decode(
  input: BinaryInput, options: DecodeOptions = {},
): Promise<DecodedImage> {
  const bytes = snapshotInput(input);
  const decoder = await loadDecoder(bytes, options);
  return decoder.decode(bytes, options);
}

export default decode;

async function loadDecoder(bytes: Uint8Array, options: DecodeOptions): Promise<DecoderModule> {
  const format = detectFormat(bytes);
  if (format === 'avif') return import('./avif.ts');
  if (format === 'heic') return import('./heic.ts');
  if (format === 'heif') {
    let file: HeifFile;
    try {
      file = new HeifFile().parse(bytes, options);
    } catch {
      return import('./index.ts');
    }
    let hasAv1 = false, hasHevc = false;
    for (const item of file.items.values()) {
      hasAv1 ||= item.type === 'av01';
      hasHevc ||= item.type === 'hvc1' || item.type === 'hev1';
    }
    if (hasAv1 && !hasHevc) return import('./avif.ts');
    if (hasHevc && !hasAv1) return import('./heic.ts');
  }
  // Mixed-codec HEIF and invalid/unsupported inputs use the full entry so its
  // established DecodeError classification remains unchanged.
  return import('./index.ts');
}

function snapshotInput(input: BinaryInput): Uint8Array {
  if (input instanceof ArrayBuffer) return new Uint8Array(input.slice(0));
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(new Uint8Array(input.buffer, input.byteOffset, input.byteLength));
  }
  // Let the established synchronous entry classify invalid runtime values.
  return new Uint8Array(0);
}
