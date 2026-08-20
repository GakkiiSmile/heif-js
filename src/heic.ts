import { createDecoder } from './decode-core.ts';
import { decodeHevcItem } from './heic-codec.ts';

export type {
  BinaryInput, DecodeOptions, DecodedImage,
} from './decode-core.ts';

export const decode = createDecoder({ hevc: decodeHevcItem });

export default decode;
