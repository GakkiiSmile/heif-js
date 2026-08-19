// Lightweight container-only entry point.  Importing this module avoids
// evaluating the AV1/HEVC entropy tables when callers only need format
// detection or HEIF metadata inspection.
export { detectFormat, HeifFile } from './bmff.ts';
export type { Clap, ImageItem, NclxColor } from './bmff.ts';
export { DEFAULT_DECODE_LIMITS } from './limits.ts';
export type { DecodeOptions } from './limits.ts';
