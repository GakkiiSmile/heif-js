import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import decodeDefault, { decode } from '../src/index.ts';
import { decode as decodeHeic } from '../src/heic.ts';
import { decode as decodeAvif } from '../src/avif.ts';
import { detectFormat, HeifFile } from '../src/bmff.ts';
import { DecodeError } from '../src/decode-core.ts';
import { Av1Decoder, nextMvScanPosition } from '../src/av1/decode.ts';
import { getDcSignContext } from '../src/av1/coeff.ts';
import { decodePng } from './png.ts';
import { DecodedFrame, CHROMA_420, CHROMA_444 } from '../src/frame.ts';
import { frameToAlpha, frameToRgba } from '../src/color.ts';
import { HevcDecoder } from '../src/hevc/decode.ts';
import { parsePps, parseSps } from '../src/hevc/pps.ts';
import { BitReader } from '../src/hevc/bitreader.ts';
import { checkedSliceQp, sliceSubstreamStarts, validateSaoOffsetScale } from '../src/hevc/guards.ts';
import { applySao } from '../src/hevc/sao.ts';
import { ResourceLimitError } from '../src/limits.ts';
import {
  countSkippedBytesInRange, nalsFromAnnexB, nalsFromLengthPrefixed, parseHvcC, rbspOffsetToEbsp,
} from '../src/hevc/nal.ts';

const heicCases = [
  ['a', 320, 240],
  ['b', 320, 240],
  ['c', 256, 256],
] as const;

assert.equal(decodeDefault, decode);
assert.deepEqual(Object.keys(await import('../src/index.ts')).sort(), ['decode', 'default']);
assert.deepEqual(Object.keys(await import('../src/heic.ts')).sort(), ['decode', 'default']);
assert.deepEqual(Object.keys(await import('../src/avif.ts')).sort(), ['decode', 'default']);
console.log('ok public decoder entries only export decode');

for (const [name, width, height] of heicCases) {
  const encoded = new Uint8Array(readFileSync(`testimages/heic_${name}.heic`));
  assert.equal(detectFormat(encoded), 'heic');

  // Exercise a non-zero byteOffset view as well as a plain Uint8Array.
  const padded = new Uint8Array(encoded.length + 7);
  padded.set(encoded, 3);
  const decoded = decode(padded.subarray(3, 3 + encoded.length));
  assert.deepEqual(Object.keys(decoded), ['width', 'height', 'data']);
  assert.equal(decoded.width, width);
  assert.equal(decoded.height, height);
  assert.equal(decoded.data.length, width * height * 4);
  assert.equal(decoded.data[3], 255);

  const reference = await decodePng(`testimages/gt_heic_${name}.png`);
  const psnr = rgbPsnr(decoded.data, reference.rgba);
  assert.ok(psnr > 22, `HEIC ${name} RGB PSNR is too low: ${psnr.toFixed(2)} dB`);
  console.log(`ok HEIC ${name}: ${width}x${height}, RGB PSNR ${psnr.toFixed(2)} dB`);
}

const outputReuseEncoded = new Uint8Array(readFileSync('testimages/heic_a.heic'));
const outputReuseExpected = decode(outputReuseEncoded);
const reusableOutput = new Uint8ClampedArray(outputReuseExpected.data.length);
const outputReuseDecoded = decode(outputReuseEncoded, { output: reusableOutput });
assert.equal(outputReuseDecoded.data, reusableOutput);
assert.deepEqual(outputReuseDecoded.data, outputReuseExpected.data);
assert.throws(
  () => decode(outputReuseEncoded, { output: new Uint8ClampedArray(reusableOutput.length - 1) }),
  (error: unknown) => error instanceof DecodeError && error.code === 'INVALID_INPUT',
);
const overlappingStorage = new ArrayBuffer(reusableOutput.length);
const overlappingInput = new Uint8Array(overlappingStorage, 0, outputReuseEncoded.length);
overlappingInput.set(outputReuseEncoded);
assert.throws(
  () => decode(overlappingInput, { output: new Uint8ClampedArray(overlappingStorage) }),
  (error: unknown) => error instanceof DecodeError && error.code === 'INVALID_INPUT',
);
console.log('ok caller-provided RGBA output buffer reuse + overlap/size validation');

const zeroCopyBytes = new Uint8Array(readFileSync('testimages/heic_a.heic'));
const zeroCopyFile = new HeifFile().parse(zeroCopyBytes);
const zeroCopyData = zeroCopyFile.primary.data;
assert.equal(zeroCopyData.buffer, zeroCopyBytes.buffer);
assert.ok(zeroCopyData.byteOffset >= zeroCopyBytes.byteOffset);
assert.ok(zeroCopyData.byteOffset + zeroCopyData.byteLength <= zeroCopyBytes.byteOffset + zeroCopyBytes.byteLength);
console.log('ok HEIF single-extent item data remains a zero-copy input view');

const reusableFile = new HeifFile();
const retainedInput = new Uint8Array(readFileSync('testimages/heic_a.heic'));
const retainedItem = reusableFile.parse(retainedInput).primary;
reusableFile.parse(new Uint8Array(readFileSync('testimages/heic_b.heic')));
const retainedData = retainedItem.data;
const retainedExpected = new HeifFile().parse(retainedInput).primary.data;
assert.equal(retainedData.buffer, retainedInput.buffer);
assert.deepEqual(retainedData, retainedExpected);
console.log('ok HEIF lazy item data survives HeifFile parser reuse');

if (typeof SharedArrayBuffer !== 'undefined') {
  const source = new Uint8Array(readFileSync('testimages/heic_b.heic'));
  const expectedFile = new HeifFile().parse(source);
  const expectedMetadata = [expectedFile.primary.type, expectedFile.primary.width, expectedFile.primary.height] as const;
  const expectedData = new Uint8Array(expectedFile.primary.data);
  const shared = new SharedArrayBuffer(source.length + 9);
  const sharedView = new Uint8Array(shared, 5, source.length);
  sharedView.set(source);
  const sharedFile = new HeifFile().parse(sharedView);
  const retainedSharedItem = sharedFile.primary;
  sharedView.fill(0);
  assert.deepEqual(
    [retainedSharedItem.type, retainedSharedItem.width, retainedSharedItem.height],
    expectedMetadata,
  );
  assert.deepEqual(retainedSharedItem.data, expectedData);
  assert.ok(retainedSharedItem.data.buffer instanceof ArrayBuffer);

  const decodeShared = new SharedArrayBuffer(source.length + 7);
  const decodeSharedView = new Uint8Array(decodeShared, 3, source.length);
  decodeSharedView.set(source);
  assert.deepEqual(decode(decodeSharedView), decode(source));
  console.log('ok SharedArrayBuffer input is snapshotted for parsing, lazy data, and decode');
}

// Sparse RBSP escape tracking must retain exact EBSP offsets without the old
// full-size Uint32Array map, and escape-free NAL units should remain zero-copy.
const plainNalPacket = new Uint8Array([4, 0x28, 0x01, 0x04, 0x80]);
const plainNal = nalsFromLengthPrefixed(plainNalPacket, 1)[0]!;
assert.equal(plainNal.rbsp.buffer, plainNalPacket.buffer);
assert.equal(plainNal.rbsp.byteOffset, plainNalPacket.byteOffset + 1);
const escapedNalPacket = new Uint8Array([11, 0x28, 0x01, 0, 0, 3, 1, 0, 0, 3, 3, 0x80]);
const escapedNal = nalsFromLengthPrefixed(escapedNalPacket, 1)[0]!;
assert.deepEqual(Array.from(escapedNal.rbsp), [0x28, 0x01, 0, 0, 1, 0, 0, 3, 0x80]);
assert.deepEqual(escapedNal.skippedBytes, [4, 8]);
assert.equal(rbspOffsetToEbsp(4, escapedNal.skippedBytes!), 5);
assert.equal(rbspOffsetToEbsp(7, escapedNal.skippedBytes!), 9);
assert.equal(countSkippedBytesInRange(escapedNal.skippedBytes!, 4, 8), 2);
const ignoredNalPacket = new Uint8Array([7, 0x46, 0x01, 0, 0, 3, 1, 0x80]); // AUD, type 35
const ignoredNal = nalsFromLengthPrefixed(ignoredNalPacket, 1)[0]!;
assert.equal(ignoredNal.rbsp.buffer, ignoredNalPacket.buffer);
assert.deepEqual(Array.from(ignoredNal.rbsp), [0x46, 0x01, 0, 0, 3, 1, 0x80]);
assert.deepEqual(ignoredNal.skippedBytes, []);
console.log('ok HEVC sparse RBSP mapping + zero-copy used/ignored NAL fast paths');

if (typeof SharedArrayBuffer !== 'undefined') {
  const sharedPacket = new Uint8Array(new SharedArrayBuffer(plainNalPacket.length));
  sharedPacket.set(plainNalPacket);
  const sharedNal = nalsFromLengthPrefixed(sharedPacket, 1)[0]!;
  sharedPacket.fill(0);
  assert.ok(sharedNal.rbsp.buffer instanceof ArrayBuffer);
  assert.deepEqual(Array.from(sharedNal.rbsp), [0x28, 0x01, 0x04, 0x80]);

  const annexBytes = new Uint8Array([0, 0, 1, 0x46, 0x01, 0x04, 0x80]);
  const sharedAnnex = new Uint8Array(new SharedArrayBuffer(annexBytes.length));
  sharedAnnex.set(annexBytes);
  const annexNal = nalsFromAnnexB(sharedAnnex)[0]!;
  sharedAnnex.fill(0);
  assert.ok(annexNal.rbsp.buffer instanceof ArrayBuffer);
  assert.deepEqual(Array.from(annexNal.rbsp), [0x46, 0x01, 0x04, 0x80]);

  const config = new HeifFile().parse(new Uint8Array(readFileSync('testimages/heic_a.heic'))).primary.config!;
  const sharedConfig = new Uint8Array(new SharedArrayBuffer(config.length));
  sharedConfig.set(config);
  const parsedSharedConfig = parseHvcC(sharedConfig);
  sharedConfig.fill(0);
  assert.ok(parsedSharedConfig.paramSets.every(nal => nal.rbsp.buffer instanceof ArrayBuffer));
  assert.ok(parsedSharedConfig.paramSets.every(nal => nal.rbsp[0] !== 0 || nal.rbsp[1] !== 0));
  console.log('ok HEVC direct NAL APIs snapshot SharedArrayBuffer input');
}

const oddHeic = decode(readBase64Fixture('testimages/heic_odd_conformance.heic.b64'));
assert.deepEqual([oddHeic.width, oddHeic.height], [321, 239]);
assert.equal(fnv1a(oddHeic.data), 3265260004);
console.log('ok HEIC SPS conformance window + clap: 321x239');

const wppMultiSliceEncoded = readBase64Fixture('testimages/heic_wpp_multislice.heic.b64');
const wppMultiSliceFile = new HeifFile().parse(wppMultiSliceEncoded);
const wppMultiSliceConfig = parseHvcC(wppMultiSliceFile.primary.config!);
const wppMultiSliceNals = nalsFromLengthPrefixed(
  wppMultiSliceFile.primary.data, wppMultiSliceConfig.lengthSize,
);
const wppMultiSlices = wppMultiSliceNals.filter(nal => nal.type <= 21);
assert.equal(wppMultiSlices.length, 2);
assert.equal(parsePps(wppMultiSliceConfig.paramSets.find(nal => nal.type === 34)!.rbsp).entropyCodingSync, true);
const wppMultiSlice = decode(wppMultiSliceEncoded);
assert.deepEqual([wppMultiSlice.width, wppMultiSlice.height], [96, 66]);
assert.equal(fnv1a(wppMultiSlice.data), 3125811013);
const partialWppDecoder = new HevcDecoder();
partialWppDecoder.registerParamSets(wppMultiSliceConfig.paramSets);
partialWppDecoder.registerParamSets(wppMultiSliceNals);
assert.throws(
  () => partialWppDecoder.decodeFrame([wppMultiSlices[0]!]),
  /slice segments do not cover CTB/,
);
const wppSps = parseSps(wppMultiSliceConfig.paramSets.find(nal => nal.type === 33)!.rbsp);
const wppCtbSize = 1 << wppSps.log2CtbSize;
const wppCtbCols = Math.ceil(wppSps.width / wppCtbSize);
const wppCtbRows = Math.ceil(wppSps.height / wppCtbSize);
const incompleteSaoFrame = new DecodedFrame(
  wppSps.width, wppSps.height, wppSps.bitDepthLuma, wppSps.chromaFormatIdc, wppSps.bitDepthChroma,
);
assert.throws(
  () => applySao(
    incompleteSaoFrame, [], [0, wppCtbCols], [0, wppCtbRows], wppCtbCols, wppCtbRows,
    wppSps, true, false, true,
  ),
  /missing SAO parameters/,
);
console.log('ok HEIC WPP multi-slice subset bounds + incomplete-slice rejection');

const heic12File = new HeifFile().parse(readBase64Fixture('testimages/heic_12bit.heic.b64'));
const heic12Config = parseHvcC(heic12File.primary.config!);
const heic12Nals = nalsFromLengthPrefixed(heic12File.primary.data, heic12Config.lengthSize);
const heic12Decoder = new HevcDecoder();
heic12Decoder.registerParamSets(heic12Config.paramSets);
heic12Decoder.registerParamSets(heic12Nals);
const heic12 = heic12Decoder.decodeFrame(heic12Nals.filter(nal => nal.type <= 31));
assert.equal(heic12.bitDepth, 12);
assert.equal(heic12.chromaBitDepth, 12);
assert.ok(heic12.planes.every(plane => plane.data instanceof Uint16Array));
assert.equal(fnv1aPlanes(heic12.planes), 2876371804);
console.log('ok HEIC 12-bit: 64x64, raw planes exact');

for (const [name, type, width, height, checksum] of [
  ['grid', 'grid', 320, 240, 986625946],
  ['overlay', 'iovl', 96, 64, 2720896015],
  ['prem', 'hvc1', 48, 40, 3748630200],
] as const) {
  const encoded = readBase64Fixture(`testimages/heif_${name}.heic.b64`);
  const file = new HeifFile().parse(encoded);
  assert.equal(file.primary.type, type);
  const decoded = decode(encoded);
  assert.deepEqual([decoded.width, decoded.height], [width, height]);
  assert.equal(fnv1a(decoded.data), checksum);
  const reused = new Uint8ClampedArray(decoded.data.length);
  const decodedInto = decode(encoded, { output: reused });
  assert.equal(decodedInto.data, reused);
  assert.deepEqual(decodedInto.data, decoded.data);
  if (name === 'grid') assert.equal(file.primary.references.dimg?.length, 6);
  if (name === 'prem') {
    let transparent = 0;
    for (let offset = 3; offset < decoded.data.length; offset += 4) transparent += +(decoded.data[offset] !== 255);
    assert.ok(transparent > 0, 'premultiplied-alpha fixture lost its auxiliary alpha plane');
  }
  console.log(`ok HEIF ${name}: ${width}x${height}, ${type}`);
}

// Make every grid cell reference the same coded tile. The cumulative budget is
// deliberately large enough for one unique tile plus the output grid, but not
// six redundant tile decodes; this exercises the item/includeAlpha cache.
const repeatedGrid = readBase64Fixture('testimages/heif_grid.heic.b64');
const dimgOffset = findAscii(repeatedGrid, 'dimg');
assert.ok(dimgOffset >= 0, 'grid fixture lost its dimg reference box');
const repeatedGridView = new DataView(repeatedGrid.buffer, repeatedGrid.byteOffset, repeatedGrid.byteLength);
const referenceCount = repeatedGridView.getUint16(dimgOffset + 6);
const firstTileId = repeatedGridView.getUint16(dimgOffset + 8);
for (let index = 1; index < referenceCount; index++) {
  repeatedGridView.setUint16(dimgOffset + 8 + index * 2, firstTileId);
}
const repeatedGridExpected = decode(repeatedGrid);
const repeatedGridLimited = decode(repeatedGrid, { maxTotalPixels: 100_000 });
assert.deepEqual(repeatedGridLimited, repeatedGridExpected);
console.log('ok HEIF repeated item references: immutable decode cache + cumulative pixel budget');

const cumulativeItemBytes = readBase64Fixture('testimages/heif_grid.heic.b64');
assert.throws(
  () => decode(cumulativeItemBytes, { maxItemBytes: 7_000, maxTotalItemBytes: 6_000 }),
  (error: unknown) => error instanceof DecodeError && error.code === 'RESOURCE_LIMIT' && /cumulative/.test(error.message),
);
console.log('ok HEIF cumulative assembled-item byte limit');

for (const path of [
  'testimages/heic_a.heic',
  'testimages/heic_odd_conformance.heic.b64',
  'testimages/heif_grid.heic.b64',
  'testimages/heif_overlay.heic.b64',
  'testimages/heif_prem.heic.b64',
]) {
  const encoded = path.endsWith('.b64') ? readBase64Fixture(path) : new Uint8Array(readFileSync(path));
  assert.deepEqual(decodeHeic(encoded), decode(encoded));
}
for (const path of [
  'testimages/avif_a_8.avif',
  'testimages/avif_intrabc.avif.b64',
  'testimages/avif_multitile_multigroup.avif.b64',
  'testimages/avif_restoration.avif.b64',
]) {
  const encoded = path.endsWith('.b64') ? readBase64Fixture(path) : new Uint8Array(readFileSync(path));
  assert.deepEqual(decodeAvif(encoded), decode(encoded));
}
assert.throws(
  () => decodeHeic(new Uint8Array(readFileSync('testimages/avif_a_8.avif'))),
  (error: unknown) => error instanceof DecodeError && error.code === 'UNSUPPORTED_CODEC' && /AV1/.test(error.message),
);
assert.throws(
  () => decodeAvif(new Uint8Array(readFileSync('testimages/heic_a.heic'))),
  (error: unknown) => error instanceof DecodeError && error.code === 'UNSUPPORTED_CODEC' && /HEVC/.test(error.message),
);
console.log('ok codec-specific entries: exact derived images + explicit cross-codec errors');

const colourFrame = new DecodedFrame(1, 1, 8, CHROMA_444);
assert.ok(colourFrame.planes.every(plane => plane.data instanceof Uint8Array));
const mixedDepthFrame = new DecodedFrame(2, 2, 8, CHROMA_444, 10);
assert.ok(mixedDepthFrame.luma.data instanceof Uint8Array);
assert.ok(mixedDepthFrame.cb.data instanceof Uint16Array);
assert.ok(mixedDepthFrame.cr.data instanceof Uint16Array);
// Identity matrix stores planes as G, B, R.
colourFrame.luma.data[0] = 150;
colourFrame.cb.data[0] = 200;
colourFrame.cr.data[0] = 100;
assert.deepEqual(Array.from(frameToRgba(colourFrame, 0, true, 1, 13)), [100, 150, 200, 255]);
assert.deepEqual(Array.from(frameToRgba(colourFrame, 0, true, 12, 13)), [83, 152, 205, 255]);
assert.deepEqual(
  Array.from(frameToRgba(colourFrame, 0, true, 2, 2, 0, makeDisplayP3Icc())),
  [83, 152, 205, 255],
);
console.log('ok NCLX/ICC Display-P3 to sRGB conversion');

const fastColourFrame = new DecodedFrame(17, 13, 8, CHROMA_420);
for (let plane = 0; plane < fastColourFrame.planes.length; plane++) {
  const data = fastColourFrame.planes[plane]!.data;
  for (let index = 0; index < data.length; index++) data[index] = (index * (plane * 17 + 29) + plane * 53) & 255;
}
const fastRgba = frameToRgba(fastColourFrame, 6, false, 2, 2, 4, null, false);
const fastAlpha = frameToAlpha(fastColourFrame, 6, false, 2, 2, 4);
for (let pixel = 0; pixel < fastAlpha.length; pixel++) assert.equal(fastAlpha[pixel], fastRgba[pixel * 4]);
const crop = { left: 2, top: 1, width: 11, height: 9 };
const croppedRgba = frameToRgba(fastColourFrame, 6, false, 2, 2, 4, null, false, crop);
for (let y = 0; y < crop.height; y++) {
  const expectedStart = ((crop.top + y) * fastColourFrame.width + crop.left) * 4;
  assert.deepEqual(
    croppedRgba.subarray(y * crop.width * 4, (y + 1) * crop.width * 4),
    fastRgba.subarray(expectedStart, expectedStart + crop.width * 4),
  );
}
console.log('ok scalar colour path: shared chroma sampling, alpha channel, source crop');

// The 8-bit co-sited 4:2:0 loop is specialized, while the equivalent 10-bit
// frame follows the scalar path. Limited-range codes scaled by four describe
// exactly the same samples, so the two paths must remain byte-identical.
const coSited10 = new DecodedFrame(fastColourFrame.width, fastColourFrame.height, 10, CHROMA_420);
for (let plane = 0; plane < fastColourFrame.planes.length; plane++) {
  const source = fastColourFrame.planes[plane]!.data;
  const destination = coSited10.planes[plane]!.data;
  for (let index = 0; index < source.length; index++) destination[index] = source[index]! << 2;
}
const coSitedCrop = { left: 2, top: 2, width: 11, height: 9 };
const coSited8Rgba = frameToRgba(fastColourFrame, 6, false, 2, 2, 1, null, true, coSitedCrop);
assert.deepEqual(coSited8Rgba, frameToRgba(coSited10, 6, false, 2, 2, 1, null, true, coSitedCrop));
assert.deepEqual(
  frameToAlpha(fastColourFrame, 6, false, 2, 2, 1, coSitedCrop),
  frameToAlpha(coSited10, 6, false, 2, 2, 1, coSitedCrop),
);
assert.deepEqual(
  frameToRgba(fastColourFrame, 6, false, 1, 13, 1, null, true, coSitedCrop),
  frameToRgba(fastColourFrame, 6, false, 1, 13, 1, null, false, coSitedCrop),
);
console.log('ok co-sited 4:2:0 fast path + exact sRGB NCLX identity');

const avifCases = [
  ['avif_a_8', 320, 240, 35],
  ['avif_a_cdef', 320, 240, 40],
  ['avif_a_nocdef', 320, 240, 45],
  ['avif_c_10', 256, 256, 28],
] as const;

for (const [name, width, height, minimumPsnr] of avifCases) {
  const encoded = new Uint8Array(readFileSync(`testimages/${name}.avif`));
  assert.equal(detectFormat(encoded), 'avif');
  const decoded = decode(encoded);
  assert.equal(decoded.width, width);
  assert.equal(decoded.height, height);
  assert.equal(decoded.data.length, width * height * 4);
  const reference = await decodePng(`testimages/gt_${name}.png`);
  const psnr = rgbPsnr(decoded.data, reference.rgba);
  assert.ok(psnr > minimumPsnr, `AVIF ${name} RGB PSNR is too low: ${psnr.toFixed(2)} dB`);
  console.log(`ok AVIF ${name}: ${width}x${height}, RGB PSNR ${psnr.toFixed(2)} dB`);
}

const avif12File = new HeifFile().parse(readBase64Fixture('testimages/avif_12bit.avif.b64'));
const avif12 = new Av1Decoder().decode(avif12File.primary.data);
assert.equal(avif12.sequence.bitDepth, 12);
assert.deepEqual([avif12.frame.width, avif12.frame.height], [64, 64]);
assert.equal(fnv1aPlanes(avif12.frame.planes), 3759341237);
console.log('ok AVIF 12-bit Profile 2: 64x64, raw planes exact');

const monoFile = new HeifFile().parse(readBase64Fixture('testimages/avif_monochrome.avif.b64'));
const mono = new Av1Decoder().decode(monoFile.primary.data);
assert.equal(mono.sequence.monochrome, true);
assert.equal(mono.frame.planes.length, 1);
const monoBytes = new Uint8Array(mono.frame.width * mono.frame.height);
for (let y = 0, offset = 0; y < mono.frame.height; y++) {
  for (let x = 0; x < mono.frame.width; x++) monoBytes[offset++] = mono.frame.luma.data[y * mono.frame.luma.stride + x]!;
}
assert.equal(fnv1a(monoBytes), 1777522025);
console.log('ok AVIF monochrome: 64x48, raw plane exact');

const fullHeaderFile = new HeifFile().parse(readBase64Fixture('testimages/avif_full_header.avif.b64'));
const fullHeader = new Av1Decoder().decode(fullHeaderFile.primary.data);
assert.equal(fullHeader.sequence.reducedStillPictureHeader, false);
assert.equal(fnv1aPlanes8(fullHeader.frame.planes), 119134900);
console.log('ok AVIF full frame/sequence headers: raw planes exact');

for (const [fixture, checksum, levels] of [
  ['seg', 2007910247, [7, 8, 8]],
  ['qm', 958273082, [0, 0, 0]],
] as const) {
  const file = new HeifFile().parse(readBase64Fixture(`testimages/avif_${fixture}.avif.b64`));
  const decoded = new Av1Decoder().decode(file.primary.data);
  assert.equal(decoded.header.usingQmatrix, true);
  assert.deepEqual([decoded.header.qmY, decoded.header.qmU, decoded.header.qmV], levels);
  assert.equal(fnv1aPlanes8(decoded.frame.planes), checksum);
}
console.log('ok AVIF quantization matrices: raw planes exact');

const intrabcEncoded = new Uint8Array(Buffer.from(
  readFileSync('testimages/avif_intrabc.avif.b64', 'utf8').replace(/\s/g, ''), 'base64',
));
const intrabcDecoded = decode(intrabcEncoded);
assert.equal(intrabcDecoded.width, 240);
assert.equal(intrabcDecoded.height, 320);
const intrabcOutput = new Uint8ClampedArray(intrabcDecoded.data.length);
const intrabcDecodedInto = decode(intrabcEncoded, { output: intrabcOutput });
assert.equal(intrabcDecodedInto.data, intrabcOutput);
assert.deepEqual(intrabcDecodedInto.data, intrabcDecoded.data);
const intrabcReference = await decodePng(Buffer.from(
  readFileSync('testimages/gt_avif_intrabc.png.b64', 'utf8').replace(/\s/g, ''), 'base64',
));
const intrabcPsnr = rgbPsnr(intrabcDecoded.data, intrabcReference.rgba);
assert.ok(intrabcPsnr > 40, `AVIF IntraBC RGB PSNR is too low: ${intrabcPsnr.toFixed(2)} dB`);
console.log(`ok AVIF IntraBC: 240x320, RGB PSNR ${intrabcPsnr.toFixed(2)} dB`);

const multiTileEncoded = new Uint8Array(Buffer.from(
  readFileSync('testimages/avif_multitile_multigroup.avif.b64', 'utf8').replace(/\s/g, ''), 'base64',
));
const multiTileDecoded = decode(multiTileEncoded);
assert.equal(multiTileDecoded.width, 512);
assert.equal(multiTileDecoded.height, 384);
const multiTileReference = Buffer.from(
  readFileSync('testimages/gt_avif_multitile_multigroup.rgb.b64', 'utf8').replace(/\s/g, ''), 'base64',
);
let multiTileSquaredError = 0, multiTileSamples = 0, referenceIndex = 0;
for (let y = 0; y < 24; y++) {
  for (let x = 0; x < 32; x++) {
    const pixel = ((y * 16) * 512 + x * 16) * 4;
    for (let channel = 0; channel < 3; channel++) {
      const delta = multiTileDecoded.data[pixel + channel]! - multiTileReference[referenceIndex++]!;
      multiTileSquaredError += delta * delta;
      multiTileSamples++;
    }
  }
}
const multiTilePsnr = 10 * Math.log10(255 * 255 / (multiTileSquaredError / multiTileSamples));
assert.ok(multiTilePsnr > 30, `AVIF multi-tile RGB PSNR is too low: ${multiTilePsnr.toFixed(2)} dB`);
console.log(`ok AVIF 2x2 tiles / 4 tile groups: 512x384, sampled RGB PSNR ${multiTilePsnr.toFixed(2)} dB`);

const superResEncoded = new Uint8Array(Buffer.from(
  readFileSync('testimages/avif_superres.avif.b64', 'utf8').replace(/\s/g, ''), 'base64',
));
const superResFile = new HeifFile().parse(superResEncoded);
assert.equal(fnv1aPlanes8(new Av1Decoder().decode(superResFile.primary.data).frame.planes), 95560061);
const superResDecoded = decode(superResEncoded);
assert.equal(superResDecoded.width, 512);
assert.equal(superResDecoded.height, 384);
const superResReference = Buffer.from(
  readFileSync('testimages/gt_avif_superres.rgb.b64', 'utf8').replace(/\s/g, ''), 'base64',
);
const superResPsnr = sampledRgbPsnr(superResDecoded.data, 512, superResReference, 32, 24, 16);
assert.ok(superResPsnr > 40, `AVIF super-res RGB PSNR is too low: ${superResPsnr.toFixed(2)} dB`);
console.log(`ok AVIF super-res 341->512: 512x384, sampled RGB PSNR ${superResPsnr.toFixed(2)} dB`);

for (const [layout, minimumPsnr] of [['422', 44], ['444', 46]] as const) {
  const encoded = readBase64Fixture(`testimages/avif_${layout}.avif.b64`);
  const decoded = decode(encoded);
  assert.equal(decoded.width, 320);
  assert.equal(decoded.height, 240);
  const reference = readBase64Fixture(`testimages/gt_avif_${layout}.rgb.b64`);
  const psnr = sampledRgbPsnr(decoded.data, 320, reference, 20, 15, 16);
  assert.ok(psnr > minimumPsnr, `AVIF 4:${layout[1]}:${layout[2]} RGB PSNR is too low: ${psnr.toFixed(2)} dB`);
  console.log(`ok AVIF 4:${layout[1]}:${layout[2]}: 320x240, sampled RGB PSNR ${psnr.toFixed(2)} dB`);
}

const filmGrainEncoded = readBase64Fixture('testimages/avif_filmgrain.avif.b64');
const filmGrainFile = new HeifFile().parse(filmGrainEncoded);
assert.equal(fnv1aPlanes8(new Av1Decoder().decode(filmGrainFile.primary.data).frame.planes), 2940198444);
const filmGrainDecoded = decode(filmGrainEncoded);
const filmGrainReference = readBase64Fixture('testimages/gt_avif_filmgrain.rgb.b64');
const filmGrainPsnr = sampledRgbPsnr(filmGrainDecoded.data, 512, filmGrainReference, 32, 24, 16);
assert.ok(filmGrainPsnr > 40, `AVIF film-grain RGB PSNR is too low: ${filmGrainPsnr.toFixed(2)} dB`);
console.log(`ok AVIF film grain: 512x384, sampled RGB PSNR ${filmGrainPsnr.toFixed(2)} dB`);

const restorationEncoded = readBase64Fixture('testimages/avif_restoration.avif.b64');
const restorationFile = new HeifFile().parse(restorationEncoded);
const restorationSyntax = new Av1Decoder().decode(restorationFile.primary.data);
assert.equal(fnv1aPlanes8(restorationSyntax.frame.planes), 2785726492);
assert.equal(restorationSyntax.blocks.length, 1835);
assert.equal(restorationSyntax.finalRange, 37051);
assert.deepEqual(
  [0, 2, 3].map(type => restorationSyntax.restorationUnits.filter(unit => unit.type === type).length),
  [37, 31, 4],
);
const restorationDecoded = decode(restorationEncoded);
const restorationReference = readBase64Fixture('testimages/gt_avif_restoration.rgb.b64');
const restorationPsnr = sampledRgbPsnr(restorationDecoded.data, 2048, restorationReference, 32, 24, 64);
assert.ok(restorationPsnr > 15, `AVIF restoration RGB PSNR is too low: ${restorationPsnr.toFixed(2)} dB`);
console.log(`ok AVIF restoration syntax/range: 2048x1536, sampled RGB PSNR ${restorationPsnr.toFixed(2)} dB`);

assert.throws(
  () => decode(new Uint8Array(32)),
  (error: unknown) => error instanceof DecodeError && error.code === 'UNSUPPORTED_FORMAT',
);
console.log('ok invalid input error');

assert.throws(
  () => decode(new Uint8Array(readFileSync('testimages/avif_a_8.avif')), { maxPixels: 100 }),
  (error: unknown) => error instanceof DecodeError && error.code === 'RESOURCE_LIMIT',
);
console.log('ok resource limit error');

assert.throws(
  () => decode(new Uint8Array(readFileSync('testimages/avif_a_8.avif')), { maxPixels: 0 }),
  (error: unknown) => error instanceof DecodeError && error.code === 'INVALID_INPUT',
);
console.log('ok invalid decode options error');

const invalidAv1C = new Uint8Array(readFileSync('testimages/avif_a_8.avif'));
const invalidAv1CFile = new HeifFile().parse(invalidAv1C);
const invalidAv1COffset = invalidAv1CFile.primary.config!.byteOffset - invalidAv1C.byteOffset;
invalidAv1C[invalidAv1COffset + 3] |= 0x80;
assert.throws(
  () => decode(invalidAv1C),
  (error: unknown) => error instanceof DecodeError && error.code === 'DECODE_FAILED' && /av1C/.test(error.message),
);
console.log('ok invalid av1C error');


/** MSB-first syntax writer used to craft parameter sets for error-path tests. */
class MsbBitWriter {
  private bytes: number[] = [];
  private current = 0;
  private filled = 0;
  u(value: number, count: number): void {
    for (let i = count - 1; i >= 0; i--) {
      this.current = (this.current << 1) | ((value >> i) & 1);
      this.filled++;
      if (this.filled === 8) { this.bytes.push(this.current); this.current = 0; this.filled = 0; }
    }
  }
  u1(value: number): void { this.u(value, 1); }
  ue(value: number): void {
    const code = value + 1;
    const length = 32 - Math.clz32(code);
    for (let i = 0; i < length - 1; i++) this.u(0, 1);
    this.u(code, length);
  }
  se(value: number): void { this.ue(value <= 0 ? -2 * value : 2 * value - 1); }
  finish(): Uint8Array {
    if (this.filled) this.bytes.push((this.current << (8 - this.filled)) & 0xff);
    return new Uint8Array(this.bytes);
  }
}

/** Minimal lossless-key-frame AV1 elementary stream for the given dimensions. */
function minimalAv1Stream(width: number, height: number): Uint8Array {
  const stream = new MsbBitWriter();
  stream.u(0, 3); stream.u1(1); stream.u1(1); stream.u(0, 5); // profile 0, still, reduced header, level
  stream.u(15, 4); stream.u(15, 4);                            // 16-bit frame size fields
  stream.u(width - 1, 16); stream.u(height - 1, 16);
  stream.u1(0); stream.u1(0); stream.u1(0);                    // sb128, filter intra, intra edge
  stream.u1(0); stream.u1(0); stream.u1(0);                    // superres, cdef, restoration
  stream.u1(0); stream.u1(0);                                  // high bitdepth, monochrome
  stream.u1(0);                                                // color description absent
  stream.u1(0); stream.u(0, 2); stream.u1(0); stream.u1(0);    // range, chroma position, uv delta, grain

  const frame = new MsbBitWriter();
  frame.u1(0); frame.u1(0); frame.u1(0);                       // cdf update, screen tools, render size
  frame.u1(1);                                                  // uniform tiles
  const ceilLog2 = (value: number): number => { let n = 0; while (1 << n < value) n++; return n; };
  if (ceilLog2(Math.min(Math.ceil(width / 64), 64)) > 0) frame.u1(0);
  if (ceilLog2(Math.min(Math.ceil(height / 64), 64)) > 0) frame.u1(0);
  frame.u(0, 8);                                                // base_q_idx = 0 (lossless)
  for (let i = 0; i < 6; i++) frame.u1(0);                     // q deltas, qmatrix, segmentation
  frame.u1(0);                                                  // reduced transform set
  const headerBytes = frame.finish();
  const payload = new Uint8Array(headerBytes.length + 1);
  payload.set(headerBytes); payload[headerBytes.length] = 0;    // one empty tile

  const obu = (type: number, body: Uint8Array): Uint8Array => {
    const out = new Uint8Array(2 + body.length);
    out[0] = (type << 3) | 0x02;                                  // has_size_field = 1
    out[1] = body.length & 0x7f;                                  // LEB128 (payloads stay < 128)
    out.set(body, 2);
    return out;
  };
  const parts = [obu(1, stream.finish()), obu(6, payload)];
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(total);
  let position = 0;
  for (const part of parts) { out.set(part, position); position += part.length; }
  return out;
}

function boxBytes(type: string, payload: Uint8Array): Uint8Array {
  const out = new Uint8Array(8 + payload.length);
  new DataView(out.buffer).setUint32(0, out.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(payload, 8);
  return out;
}

function fullBoxBytes(type: string, payload: Uint8Array, version = 0): Uint8Array {
  const body = new Uint8Array(4 + payload.length);
  body[0] = version;
  body.set(payload, 4);
  return boxBytes(type, body);
}

const u16Bytes = (value: number): Uint8Array => {
  const out = new Uint8Array(2);
  new DataView(out.buffer).setUint16(0, value);
  return out;
};
const u32Bytes = (value: number): Uint8Array => {
  const out = new Uint8Array(4);
  new DataView(out.buffer).setUint32(0, value >>> 0);
  return out;
};
const ispeBox = (width: number, height: number): Uint8Array =>
  boxBytes('ispe', new Uint8Array([...u32Bytes(0), ...u32Bytes(width), ...u32Bytes(height)]));

const concatBytes = (parts: Uint8Array[]): Uint8Array => {
  const out = new Uint8Array(parts.reduce((n, part) => n + part.length, 0));
  let position = 0;
  for (const part of parts) { out.set(part, position); position += part.length; }
  return out;
};

/** Build a minimal HEIF file with length-prefixed iloc extents over one mdat. */
function buildHeif(structure: {
  primary: number;
  items: { id: number; type: string; props: Uint8Array[]; data: Uint8Array }[];
  refs: { type: string; from: number; to: number[] }[];
}): Uint8Array {
  const { primary, items, refs } = structure;
  const typeCode = (type: string): Uint8Array =>
    new Uint8Array(type.charCodeAt(0) ? type.split('').map(c => c.charCodeAt(0) & 0xff) : [0x30, 0x30, 0x30, 0x30]);
  const infes = concatBytes(items.map(item =>
    fullBoxBytes('infe', concatBytes([u16Bytes(item.id), u16Bytes(0), typeCode(item.type)]), 2)));
  const iinf = fullBoxBytes('iinf', concatBytes([u16Bytes(items.length), infes]));
  const pitm = fullBoxBytes('pitm', u16Bytes(primary));
  const irefBox = fullBoxBytes('iref', concatBytes(refs.map(ref =>
    boxBytes(ref.type, concatBytes([u16Bytes(ref.from), u16Bytes(ref.to.length), ...ref.to.map(u16Bytes)])))));
  const ipco = boxBytes('ipco', concatBytes(items.flatMap(item => item.props)));
  const ipma = fullBoxBytes('ipma', (() => {
    const entries: number[] = [];
    let index = 1;
    for (const item of items) {
      const indices = item.props.map(() => index++);
      entries.push(item.id >> 8, item.id & 0xff, indices.length, ...indices);
    }
    return concatBytes([u32Bytes(items.length), new Uint8Array(entries)]);
  })());
  const iprp = boxBytes('iprp', concatBytes([ipco, ipma]));
  const ftyp = boxBytes('ftyp', concatBytes([
    new Uint8Array([0x6d, 0x69, 0x66, 0x31]), u32Bytes(0), new Uint8Array([0x6d, 0x69, 0x66, 0x31]),
  ]));

  // iloc has fixed-width fields, so its serialized length is known before the
  // extent offsets inside it are: build a placeholder, measure, then rebuild
  // with the real (same-width) offsets so mdat positions stay valid.
  const ilocEntry = (offset: number): Uint8Array =>
    fullBoxBytes('iloc', concatBytes([
      u16Bytes((4 << 12) | (4 << 8)), u16Bytes(items.length),
      concatBytes(items.map((item, index) => concatBytes([
        u16Bytes(item.id), u16Bytes(0), u16Bytes(1),
        u32Bytes(offset + items.slice(0, index).reduce((n, previous) => n + previous.data.length, 0)),
        u32Bytes(item.data.length),
      ]))),
    ]));
  const placeholder = ilocEntry(0);
  const mdatStart = ftyp.length + fullBoxBytes('meta', concatBytes([pitm, iinf, placeholder, irefBox, iprp])).length + 8;
  const iloc = ilocEntry(mdatStart);
  assert.equal(iloc.length, placeholder.length);
  const mdatPayload = concatBytes(items.map(item => item.data));
  return concatBytes([
    ftyp,
    fullBoxBytes('meta', concatBytes([pitm, iinf, iloc, irefBox, iprp])),
    boxBytes('mdat', mdatPayload),
  ]);
}

// ---------- audit regression: AV1 partition tree at non-multiple-of-8 sizes ----------
// A partition region smaller than one 8x8 block used to recurse past the
// 5-level partition CDF table and throw a TypeError on fully conforming
// dimensions (4x4, 65x65, 68x68, ...). Synthesized key-frame streams with an
// empty tile exercise exactly the tree-floor decisions.
{
  const partitionSizes: [number, number][] = [
    [4, 4], [65, 65], [68, 68], [100, 100], [132, 132], [129, 129], [65, 1], [1, 65],
    [64, 64], [68, 64], [64, 68], [13, 7],
  ];
  for (const [width, height] of partitionSizes) {
    const result = new Av1Decoder().decode(minimalAv1Stream(width, height));
    assert.equal(result.frame.width, width, `synthetic AV1 ${width}x${height} width`);
    assert.equal(result.frame.height, height, `synthetic AV1 ${width}x${height} height`);
    assert.ok(result.blocks.length > 0);
    for (const block of result.blocks) {
      assert.ok(block.x4 < Math.ceil(width / 4) && block.y4 < Math.ceil(height / 4));
    }
  }
  // End-to-end through the public entry: wrap the 65x65 stream in a real AVIF
  // container (av1C with matching sequence fields) and decode RGBA.
  const avif65 = buildHeif({
    primary: 1,
    items: [{
      id: 1, type: 'av01',
      props: [ispeBox(65, 65), boxBytes('av1C', new Uint8Array([0x81, 0x00, 0x0c, 0x00]))],
      data: minimalAv1Stream(65, 65),
    }],
    refs: [],
  });
  const avifBranded = concatBytes([
    boxBytes('ftyp', concatBytes([
      new Uint8Array([0x61, 0x76, 0x69, 0x66]), u32Bytes(0), new Uint8Array([0x61, 0x76, 0x69, 0x66]),
    ])),
    avif65.subarray(8 + 12), // drop the mif1 ftyp, keep meta+mdat (offsets shift by 0)
  ]);
  const decoded65 = decode(avifBranded);
  assert.equal(decoded65.width, 65);
  assert.equal(decoded65.height, 65);
  assert.equal(decoded65.data.length, 65 * 65 * 4);
  console.log(`ok AV1 partition tree floor: ${partitionSizes.length} odd/aligned sizes decode (+container 65x65)`);
}

// ---------- audit regression: HEVC scaling-list delta referencing a missing matrix ----------
{
  const writer = new MsbBitWriter();
  writer.ue(0); writer.ue(0);            // pps/sps id
  writer.u1(0); writer.u1(0);            // dependent slices, output flag
  writer.u(0, 3);                        // extra slice header bits
  writer.u1(0); writer.u1(0);            // sign data hiding, cabac init
  writer.ue(0); writer.ue(0);            // ref idx defaults
  writer.se(0);                          // init_qp_minus26
  writer.u1(0); writer.u1(0); writer.u1(0); // constrained, transform skip, cu qp delta
  writer.se(0); writer.se(0);            // cb/cr qp offsets
  writer.u1(0);                          // chroma qp offsets present
  writer.u1(0); writer.u1(0);            // weighted pred/bipred
  writer.u1(0); writer.u1(0); writer.u1(0); // transquant bypass, tiles, entropy sync
  writer.u1(0);                          // loop filter across slices
  writer.u1(0);                          // deblocking control absent
  writer.u1(1);                          // pps_scaling_list_data_present
  // sizeId 0, matrixId 0: pred_mode delta with delta=1 -> refId -1
  writer.u1(0); writer.ue(1);
  const rbsp = new Uint8Array([0x44, 0x01, ...writer.finish()]);
  assert.throws(
    () => parsePps(rbsp),
    (error: unknown) => error instanceof Error && /scaling-list delta/.test(error.message),
  );
  console.log('ok HEVC scaling-list delta rejected with a clean parse error');
}

// ---------- audit regression: NAL-count resource limit ----------
{
  const nal = (size: number): Uint8Array => new Uint8Array(size).fill(0x28, 0, 1).fill(0x01, 1, 2);
  const prefixed = new Uint8Array((2 + 1) * 3);
  for (let n = 0; n < 3; n++) {
    prefixed[n * 3] = 2;
    prefixed.set(nal(2), n * 3 + 1);
  }
  assert.equal(nalsFromLengthPrefixed(prefixed, 1).length, 3);
  assert.throws(
    () => nalsFromLengthPrefixed(prefixed, 1, 2),
    (error: unknown) => error instanceof ResourceLimitError,
  );
  const annexB = new Uint8Array([
    0, 0, 1, 0x28, 0x01,
    0, 0, 1, 0x28, 0x01,
    0, 0, 1, 0x28, 0x01,
  ]);
  assert.equal(nalsFromAnnexB(annexB).length, 3);
  assert.throws(() => nalsFromAnnexB(annexB, 2), (error: unknown) => error instanceof ResourceLimitError);
  const hvcC = new Uint8Array(23 + 3 + 3 * 4);
  hvcC[0] = 1; hvcC[21] = 0x05; hvcC[22] = 1; // version, lengthSize=2, one array
  hvcC[23] = 33; hvcC[24] = 0; hvcC[25] = 3;  // SPS array with three NALs
  for (let n = 0; n < 3; n++) {
    const at = 26 + n * 4;
    hvcC[at] = 0; hvcC[at + 1] = 2; hvcC[at + 2] = 0x42; hvcC[at + 3] = 0x01;
  }
  assert.deepEqual([parseHvcC(hvcC).lengthSize, parseHvcC(hvcC).paramSets.length], [2, 3]);
  assert.throws(() => parseHvcC(hvcC, 2), (error: unknown) => error instanceof ResourceLimitError);
  assert.throws(
    () => decode(new Uint8Array(readFileSync('testimages/heic_a.heic')), { maxNals: 1 }),
    (error: unknown) => error instanceof DecodeError && error.code === 'RESOURCE_LIMIT',
  );
  assert.throws(
    () => decode(new Uint8Array(readFileSync('testimages/heic_a.heic')), { maxNals: 0 }),
    (error: unknown) => error instanceof DecodeError && error.code === 'INVALID_INPUT',
  );
  console.log('ok HEVC NAL count limit (length-prefixed, hvcC, decode option)');
}

// ---------- audit regression: 32-bit syntax reads stay unsigned ----------
{
  const reader = new BitReader(new Uint8Array([0xff, 0xff, 0xff, 0xff]));
  const value = reader.u(32);
  assert.equal(value, 0xffffffff);
  console.log('ok HEVC BitReader u(32) is unsigned');
}

// ---------- audit regression: codec edge guards retain normative values ----------
{
  assert.equal(checkedSliceQp(8, -26, 0), 0);
  assert.equal(checkedSliceQp(10, -38, 0), -12);
  assert.equal(checkedSliceQp(12, -50, 0), -24);
  assert.throws(() => checkedSliceQp(12, -51, 0), /init_qp_minus26/);
  assert.throws(() => checkedSliceQp(12, 25, 1), /slice_qp_delta/);

  assert.deepEqual(sliceSubstreamStarts(5, [], 20), [5]);
  assert.deepEqual(sliceSubstreamStarts(5, [3, 9], 20), [5, 8, 14]);
  assert.throws(() => sliceSubstreamStarts(5, [3, 3], 20), /entry-point offset/);
  assert.throws(() => sliceSubstreamStarts(5, [15], 20), /entry-point offset/);

  validateSaoOffsetScale(0, 8, 'luma');
  validateSaoOffsetScale(2, 12, 'luma');
  validateSaoOffsetScale(6, 16, 'chroma');
  assert.throws(() => validateSaoOffsetScale(1, 8, 'luma'), /SAO offset scale/);
  assert.throws(() => validateSaoOffsetScale(3, 12, 'luma'), /SAO offset scale/);
  assert.throws(() => validateSaoOffsetScale(7, 16, 'chroma'), /SAO offset scale/);

  assert.equal(nextMvScanPosition(7, 4, 12), 11);
  assert.equal(nextMvScanPosition(8, 4, 12), null);
  assert.equal(getDcSignContext(new Uint8Array([0x40]), new Uint8Array([0x40]), 2, 2), 0);

  const padded = new DecodedFrame(5, 3, 8, CHROMA_420, 8, 4);
  assert.equal(padded.luma.stride, 9);
  padded.luma.data[5] = 123;
  assert.equal(padded.luma.data[padded.luma.stride], 0, 'right-edge padding wrapped into the next visible row');
  console.log('ok codec edge guards: negative QP, subset count, SAO scale, MV stop, DC context, plane padding');
}

// ---------- audit regression: hostile derived-item containers are rejected cleanly ----------
{
  const cyclicGrid = buildHeif({
    primary: 1,
    items: [
      { id: 1, type: 'grid', props: [ispeBox(64, 64)], data: new Uint8Array([0, 0, 0, 0, 0, 64, 0, 64]) },
      { id: 2, type: 'hvc1', props: [ispeBox(64, 64)], data: new Uint8Array(0) },
    ],
    refs: [{ type: 'dimg', from: 1, to: [1, 2] }],
  });
  assert.throws(
    () => decode(cyclicGrid),
    (error: unknown) => error instanceof DecodeError && error.code === 'DECODE_FAILED' && /Cyclic/.test(error.message),
  );

  const chainIds = Array.from({ length: 200 }, (_, index) => index + 1);
  const deepIdentity = buildHeif({
    primary: 1,
    items: chainIds.map(id => ({ id, type: 'iden', props: [ispeBox(64, 64)], data: new Uint8Array(0) })),
    refs: chainIds.slice(0, -1).map((id, index) => ({ type: 'dimg', from: id, to: [chainIds[index + 1]!] })),
  });
  assert.throws(
    () => decode(deepIdentity),
    (error: unknown) => error instanceof DecodeError && error.code === 'RESOURCE_LIMIT' && /depth/.test(error.message),
  );

  const gigantic = buildHeif({
    primary: 1,
    items: [{ id: 1, type: 'hvc1', props: [ispeBox(0xffffffff, 0xffffffff)], data: new Uint8Array(0) }],
    refs: [],
  });
  assert.throws(
    () => decode(gigantic),
    (error: unknown) => error instanceof DecodeError && error.code === 'RESOURCE_LIMIT',
  );

  const shortGrid = buildHeif({
    primary: 1,
    items: [
      { id: 1, type: 'grid', props: [ispeBox(64, 64)], data: new Uint8Array([0, 0, 2, 2, 0, 64, 0, 64]) },
      { id: 2, type: 'hvc1', props: [ispeBox(64, 64)], data: new Uint8Array(0) },
    ],
    refs: [{ type: 'dimg', from: 1, to: [2] }],
  });
  assert.throws(
    () => decode(shortGrid),
    (error: unknown) => error instanceof DecodeError && error.code === 'DECODE_FAILED' && /tile references/.test(error.message),
  );

  const essentialMissing = buildHeif({
    primary: 1,
    items: [{ id: 1, type: 'hvc1', props: [ispeBox(32, 32), boxBytes('unkn', new Uint8Array(0))], data: new Uint8Array(0) }],
    refs: [],
  });
  // Mark the second (unknown) property essential via a raw ipma patch.
  const essentialText = Buffer.from(essentialMissing.buffer, essentialMissing.byteOffset, essentialMissing.byteLength).toString('latin1');
  const ipmaAt = essentialText.indexOf('ipma');
  assert.ok(ipmaAt >= 0);
  // ipma content: version/flags(4) + entryCount(4) + per entry itemId(2) count(1) indices...
  const indexOffset = ipmaAt + 4 + 4 + 4 + 2 + 1;
  essentialMissing[indexOffset] = 0x82; // essential + property index 2
  assert.throws(
    () => decode(essentialMissing),
    (error: unknown) => error instanceof DecodeError && error.code === 'UNSUPPORTED_CODEC',
  );
  console.log('ok hostile HEIF derived-item containers rejected (cycle/depth/size/tiles/essential)');
}

function rgbPsnr(actual: Uint8Array, expected: Uint8Array): number {
  assert.equal(actual.length, expected.length);
  let squaredError = 0;
  let count = 0;
  for (let i = 0; i < actual.length; i += 4) {
    for (let channel = 0; channel < 3; channel++) {
      const delta = actual[i + channel]! - expected[i + channel]!;
      squaredError += delta * delta;
      count++;
    }
  }
  if (squaredError === 0) return Infinity;
  return 10 * Math.log10(255 * 255 / (squaredError / count));
}

function sampledRgbPsnr(
  actual: Uint8Array, actualWidth: number, expected: Uint8Array,
  sampleWidth: number, sampleHeight: number, step: number,
): number {
  let squaredError = 0, count = 0, expectedIndex = 0;
  for (let y = 0; y < sampleHeight; y++) {
    for (let x = 0; x < sampleWidth; x++) {
      const pixel = ((y * step) * actualWidth + x * step) * 4;
      for (let channel = 0; channel < 3; channel++) {
        const delta = actual[pixel + channel]! - expected[expectedIndex++]!;
        squaredError += delta * delta;
        count++;
      }
    }
  }
  return squaredError ? 10 * Math.log10(255 * 255 / (squaredError / count)) : Infinity;
}

function readBase64Fixture(path: string): Uint8Array {
  return new Uint8Array(Buffer.from(readFileSync(path, 'utf8').replace(/\s/g, ''), 'base64'));
}

function findAscii(bytes: Uint8Array, text: string): number {
  for (let offset = 0; offset <= bytes.length - text.length; offset++) {
    let matches = true;
    for (let index = 0; index < text.length; index++) matches &&= bytes[offset + index] === text.charCodeAt(index);
    if (matches) return offset;
  }
  return -1;
}

function fnv1a(bytes: Uint8Array): number {
  let hash = 2166136261;
  for (const byte of bytes) hash = Math.imul(hash ^ byte, 16777619) >>> 0;
  return hash;
}

function fnv1aPlanes(planes: DecodedFrame['planes']): number {
  let hash = 2166136261;
  for (const plane of planes) {
    for (let y = 0; y < plane.height; y++) {
      for (let x = 0; x < plane.width; x++) {
        const value = plane.data[y * plane.stride + x]!;
        hash = Math.imul(hash ^ (value & 0xff), 16777619) >>> 0;
        hash = Math.imul(hash ^ (value >> 8), 16777619) >>> 0;
      }
    }
  }
  return hash;
}

function fnv1aPlanes8(planes: DecodedFrame['planes']): number {
  let hash = 2166136261;
  for (const plane of planes) {
    for (let y = 0; y < plane.height; y++) {
      for (let x = 0; x < plane.width; x++) {
        hash = Math.imul(hash ^ plane.data[y * plane.stride + x]!, 16777619) >>> 0;
      }
    }
  }
  return hash;
}

function makeDisplayP3Icc(): Uint8Array {
  const bytes = new Uint8Array(296);
  const view = new DataView(bytes.buffer);
  const ascii = (offset: number, text: string): void => {
    for (let index = 0; index < text.length; index++) bytes[offset + index] = text.charCodeAt(index);
  };
  const fixed = (offset: number, value: number): void => view.setInt32(offset, Math.round(value * 65_536));
  view.setUint32(0, bytes.length);
  ascii(12, 'mntr'); ascii(16, 'RGB '); ascii(20, 'XYZ '); ascii(36, 'acsp');
  const tags = [
    ['rXYZ', 204, 20], ['gXYZ', 224, 20], ['bXYZ', 244, 20],
    ['rTRC', 264, 32], ['gTRC', 264, 32], ['bTRC', 264, 32],
  ] as const;
  view.setUint32(128, tags.length);
  tags.forEach(([signature, offset, size], index) => {
    const entry = 132 + index * 12;
    ascii(entry, signature); view.setUint32(entry + 4, offset); view.setUint32(entry + 8, size);
  });
  const xyz = [
    [0.51512146, 0.24119568, -0.00105286],
    [0.29197693, 0.69224548, 0.04188538],
    [0.15710449, 0.06657410, 0.78407288],
  ];
  xyz.forEach((values, index) => {
    const offset = 204 + index * 20;
    ascii(offset, 'XYZ ');
    values.forEach((value, component) => fixed(offset + 8 + component * 4, value));
  });
  ascii(264, 'para');
  view.setUint16(272, 3);
  [2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045]
    .forEach((value, index) => fixed(276 + index * 4, value));
  return bytes;
}
