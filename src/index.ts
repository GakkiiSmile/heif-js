import { createDecoder } from './decode-core.ts';
import { decodeAv1Item } from './avif-codec.ts';
import { decodeHevcItem } from './heic-codec.ts';

export type {
  BinaryInput, DecodeOptions, DecodedImage,
} from './decode-core.ts';

export const decode = createDecoder({ hevc: decodeHevcItem, av1: decodeAv1Item });

export default decode;
