import type { ImageItem } from './bmff.ts';
import { frameToAlpha, frameToRgba } from './color.ts';
import { DecodeError } from './decode-core.ts';
import type { DecodedItemImage, ItemCodecDecoder } from './decode-core.ts';
import { Av1Decoder } from './av1/decode.ts';
import { OBU_SEQUENCE_HEADER, parseObus, parseSequenceHeader } from './av1/obu.ts';

export const decodeAv1Item: ItemCodecDecoder = (
  item, limits, colorManagement,
): DecodedItemImage => {
  const decoded = new Av1Decoder(limits).decode(item.data);
  validateAv1Configuration(item, decoded.sequence);
  const matrix = item.nclx?.matrixCoefficients ?? decoded.sequence.matrixCoefficients;
  const fullRange = item.nclx?.fullRangeFlag ?? decoded.sequence.fullRange;
  const primaries = item.nclx?.colourPrimaries ?? decoded.sequence.colorPrimaries;
  const transfer = item.nclx?.transferCharacteristics ?? decoded.sequence.transferCharacteristics;
  return {
    width: decoded.frame.width,
    height: decoded.frame.height,
    data: colorManagement
      ? frameToRgba(
        decoded.frame, matrix, fullRange, primaries, transfer, decoded.sequence.chromaSamplePosition,
        item.icc, true,
      )
      : frameToAlpha(
        decoded.frame, matrix, fullRange, primaries, transfer, decoded.sequence.chromaSamplePosition,
      ),
    channels: colorManagement ? 4 : 1,
  };
};

function validateAv1Configuration(item: ImageItem, sequence: ReturnType<Av1Decoder['decode']>['sequence']): void {
  const configuration = item.config;
  if (!configuration) throw new DecodeError('MISSING_CONFIG', 'AV1 image item has no av1C configuration');
  if (configuration.length < 4 || !(configuration[0]! & 0x80) || (configuration[0]! & 0x7f) !== 1) {
    throw new DecodeError('DECODE_FAILED', 'AV1 image item has an invalid av1C configuration');
  }
  const profile = configuration[1]! >> 5;
  const level = configuration[1]! & 0x1f;
  const tier = configuration[2]! >> 7;
  const highBitDepth = (configuration[2]! >> 6) & 1;
  const twelveBit = (configuration[2]! >> 5) & 1;
  const bitDepth = highBitDepth ? (twelveBit ? 12 : 10) : 8;
  const monochrome = (configuration[2]! >> 4) & 1;
  const subsamplingX = (configuration[2]! >> 3) & 1;
  const subsamplingY = (configuration[2]! >> 2) & 1;
  const chromaSamplePosition = configuration[2]! & 3;
  if (profile !== sequence.profile || level !== sequence.seqLevelIdx[0] || tier !== sequence.seqTier[0] ||
      bitDepth !== sequence.bitDepth ||
      !!monochrome !== sequence.monochrome || subsamplingX !== sequence.subsamplingX ||
      subsamplingY !== sequence.subsamplingY ||
      (subsamplingX && subsamplingY && chromaSamplePosition !== sequence.chromaSamplePosition)) {
    throw new DecodeError('DECODE_FAILED', 'AV1 av1C configuration does not match the sequence header');
  }
  if ((configuration[3]! & 0xe0) || (!(configuration[3]! & 0x10) && (configuration[3]! & 0x0f))) {
    throw new DecodeError('DECODE_FAILED', 'AV1 image item has invalid reserved bits in av1C');
  }
  if (configuration.length > 4) {
    let configSequence: ReturnType<typeof parseSequenceHeader> | undefined;
    try {
      const sequenceHeaders = parseObus(configuration.subarray(4))
        .filter(obu => obu.type === OBU_SEQUENCE_HEADER);
      if (sequenceHeaders.length > 1) throw new Error('av1C contains multiple sequence headers');
      if (sequenceHeaders.length === 1) configSequence = parseSequenceHeader(sequenceHeaders[0]!.payload);
    } catch (error) {
      throw new DecodeError(
        'DECODE_FAILED', error instanceof Error ? `Invalid av1C configOBUs: ${error.message}` : 'Invalid av1C configOBUs', error,
      );
    }
    if (configSequence && JSON.stringify(configSequence) !== JSON.stringify(sequence)) {
      throw new DecodeError('DECODE_FAILED', 'AV1 sequence header in av1C does not match the image item');
    }
  }
}
