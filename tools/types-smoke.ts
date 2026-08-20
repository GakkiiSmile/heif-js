import decodeDefault, { decode } from 'heif-js';
import type { BinaryInput, DecodeOptions, DecodedImage } from 'heif-js';
import { DEFAULT_DECODE_LIMITS, HeifFile, detectFormat } from 'heif-js/detect';
import type { Clap, ImageItem, NclxColor } from 'heif-js/detect';
import decodeHeic from 'heif-js/heic';
import decodeAvif from 'heif-js/avif';

declare const input: BinaryInput;
const options: DecodeOptions = { maxPixels: DEFAULT_DECODE_LIMITS.maxPixels };
const decoded: DecodedImage = decode(input, options);
const decodedDefault: DecodedImage = decodeDefault(input);
const decodedHeic: DecodedImage = decodeHeic(input);
const decodedAvif: DecodedImage = decodeAvif(input);
const format: ReturnType<typeof detectFormat> = detectFormat(new Uint8Array(0));
const file: HeifFile = new HeifFile();

void [decoded, decodedDefault, decodedHeic, decodedAvif, format, file];
void (null as Clap | ImageItem | NclxColor | null);
