import { createDecoder } from './decode-core.ts';
import { decodeAv1Item } from './avif-codec.ts';

export type {
  BinaryInput, DecodeOptions, DecodedImage,
} from './decode-core.ts';

export const decode = createDecoder({ av1: decodeAv1Item });

export default decode;
