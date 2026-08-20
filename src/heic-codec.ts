import { frameToAlpha, frameToRgba } from './color.ts';
import type { FrameCrop } from './color.ts';
import { DecodeError } from './decode-core.ts';
import type { DecodedItemImage, ItemCodecDecoder } from './decode-core.ts';
import { HevcDecoder } from './hevc/decode.ts';
import { nalsFromLengthPrefixed, parseHvcC } from './hevc/nal.ts';

export const decodeHevcItem: ItemCodecDecoder = (
  item, limits, colorManagement, output,
): DecodedItemImage => {
  if (!item.config) throw new DecodeError('MISSING_CONFIG', 'HEVC image item has no hvcC configuration');
  const config = parseHvcC(item.config, limits.maxNals);
  const itemNals = nalsFromLengthPrefixed(item.data, config.lengthSize, limits.maxNals);
  const decoder = new HevcDecoder(limits);
  decoder.registerParamSets(config.paramSets);
  decoder.registerParamSets(itemNals);
  const slices = itemNals.filter(nal => nal.type <= 31);
  if (slices.length === 0) throw new DecodeError('DECODE_FAILED', 'HEVC image item contains no slice NAL unit');
  const frame = decoder.decodeFrame(slices);
  const sps = decoder.decodedSps;
  const matrix = item.nclx?.matrixCoefficients ?? sps?.matrixCoefficients ?? 6;
  const fullRange = item.nclx?.fullRangeFlag ?? sps?.fullRange ?? false;
  const primaries = item.nclx?.colourPrimaries ?? sps?.colourPrimaries ?? 2;
  const transfer = item.nclx?.transferCharacteristics ?? sps?.transferCharacteristics ?? 2;
  // Translate HEVC chroma_sample_loc_type to the internal AV1-compatible
  // co-sited/centered modes. Field-specific types 4/5 use the closest
  // progressive-frame location.
  const chromaLocation = [1, 3, 2, 4, 1, 2][sps?.chromaSampleLocation ?? 0] ?? 1;
  let crop: FrameCrop | null = null;
  if (sps?.conformance) {
    const subWidth = sps.chromaFormatIdc === 1 || sps.chromaFormatIdc === 2 ? 2 : 1;
    const subHeight = sps.chromaFormatIdc === 1 ? 2 : 1;
    const left = sps.conformance.left * subWidth;
    const right = sps.conformance.right * subWidth;
    const top = sps.conformance.top * subHeight;
    const bottom = sps.conformance.bottom * subHeight;
    if (left + right >= frame.width || top + bottom >= frame.height) {
      throw new DecodeError('DECODE_FAILED', 'HEVC conformance window is outside the coded picture');
    }
    crop = { left, top, width: frame.width - left - right, height: frame.height - top - bottom };
  }
  const width = crop?.width ?? frame.width, height = crop?.height ?? frame.height;
  return {
    width, height,
    data: colorManagement
      ? frameToRgba(frame, matrix, fullRange, primaries, transfer, chromaLocation, item.icc, true, crop, output ?? null)
      : frameToAlpha(frame, matrix, fullRange, primaries, transfer, chromaLocation, crop),
    channels: colorManagement ? 4 : 1,
  };
};
