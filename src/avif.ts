import { createDecoder } from './decode-core.ts';
import { decodeAv1Item } from './avif-codec.ts';

export { DecodeError, HeifFile, detectFormat } from './decode-core.ts';
export { DEFAULT_DECODE_LIMITS } from './decode-core.ts';
export type {
  BinaryInput, Clap, DecodeErrorCode, DecodeOptions, DecodedImage, ImageItem, NclxColor,
} from './decode-core.ts';

const decoder = createDecoder({ av1: decodeAv1Item });

export const decodeToRgba = decoder.decodeToRgba;
export const decodeToImageData = decoder.decodeToImageData;
export const decode = decoder.decode;

export default decode;
