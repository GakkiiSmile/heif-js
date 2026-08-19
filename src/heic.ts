import { createDecoder } from './decode-core.ts';
import { decodeHevcItem } from './heic-codec.ts';

export { DecodeError, HeifFile, detectFormat } from './decode-core.ts';
export { DEFAULT_DECODE_LIMITS } from './decode-core.ts';
export type {
  BinaryInput, Clap, DecodeErrorCode, DecodeOptions, DecodedImage, ImageItem, NclxColor,
} from './decode-core.ts';

const decoder = createDecoder({ hevc: decodeHevcItem });

export const decodeToRgba = decoder.decodeToRgba;
export const decodeToImageData = decoder.decodeToImageData;
export const decode = decoder.decode;

export default decode;
