import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { decodeToRgba, DecodeError, detectFormat } from '../src/index.ts';
import { HeifFile } from '../src/bmff.ts';
import { Av1Decoder } from '../src/av1/decode.ts';
import { decodePng } from './png.ts';
import { DecodedFrame, CHROMA_444 } from '../src/frame.ts';
import { frameToRgba } from '../src/color.ts';
import { HevcDecoder } from '../src/hevc/decode.ts';
import { nalsFromLengthPrefixed, parseHvcC } from '../src/hevc/nal.ts';

const heicCases = [
  ['a', 320, 240],
  ['b', 320, 240],
  ['c', 256, 256],
] as const;

for (const [name, width, height] of heicCases) {
  const encoded = new Uint8Array(readFileSync(`testimages/heic_${name}.heic`));
  assert.equal(detectFormat(encoded), 'heic');

  // Exercise a non-zero byteOffset view as well as a plain Uint8Array.
  const padded = new Uint8Array(encoded.length + 7);
  padded.set(encoded, 3);
  const decoded = decodeToRgba(padded.subarray(3, 3 + encoded.length));
  assert.equal(decoded.width, width);
  assert.equal(decoded.height, height);
  assert.equal(decoded.data.length, width * height * 4);
  assert.equal(decoded.data[3], 255);

  const reference = await decodePng(`testimages/gt_heic_${name}.png`);
  const psnr = rgbPsnr(decoded.data, reference.rgba);
  assert.ok(psnr > 22, `HEIC ${name} RGB PSNR is too low: ${psnr.toFixed(2)} dB`);
  console.log(`ok HEIC ${name}: ${width}x${height}, RGB PSNR ${psnr.toFixed(2)} dB`);
}

const oddHeic = decodeToRgba(readBase64Fixture('testimages/heic_odd_conformance.heic.b64'));
assert.deepEqual([oddHeic.width, oddHeic.height], [321, 239]);
assert.equal(fnv1a(oddHeic.data), 3265260004);
console.log('ok HEIC SPS conformance window + clap: 321x239');

const heic12File = new HeifFile().parse(readBase64Fixture('testimages/heic_12bit.heic.b64'));
const heic12Config = parseHvcC(heic12File.primary.config!);
const heic12Nals = nalsFromLengthPrefixed(heic12File.primary.data, heic12Config.lengthSize);
const heic12Decoder = new HevcDecoder();
heic12Decoder.registerParamSets(heic12Config.paramSets);
heic12Decoder.registerParamSets(heic12Nals);
const heic12 = heic12Decoder.decodeFrame(heic12Nals.filter(nal => nal.type <= 31));
assert.equal(heic12.bitDepth, 12);
assert.equal(heic12.chromaBitDepth, 12);
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
  const decoded = decodeToRgba(encoded);
  assert.deepEqual([decoded.width, decoded.height], [width, height]);
  assert.equal(fnv1a(decoded.data), checksum);
  if (name === 'grid') assert.equal(file.primary.references.dimg?.length, 6);
  if (name === 'prem') {
    let transparent = 0;
    for (let offset = 3; offset < decoded.data.length; offset += 4) transparent += +(decoded.data[offset] !== 255);
    assert.ok(transparent > 0, 'premultiplied-alpha fixture lost its auxiliary alpha plane');
  }
  console.log(`ok HEIF ${name}: ${width}x${height}, ${type}`);
}

const colourFrame = new DecodedFrame(1, 1, 8, CHROMA_444);
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

const avifCases = [
  ['avif_a_8', 320, 240, 35],
  ['avif_a_cdef', 320, 240, 40],
  ['avif_a_nocdef', 320, 240, 45],
  ['avif_c_10', 256, 256, 28],
] as const;

for (const [name, width, height, minimumPsnr] of avifCases) {
  const encoded = new Uint8Array(readFileSync(`testimages/${name}.avif`));
  assert.equal(detectFormat(encoded), 'avif');
  const decoded = decodeToRgba(encoded);
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
const intrabcDecoded = decodeToRgba(intrabcEncoded);
assert.equal(intrabcDecoded.width, 240);
assert.equal(intrabcDecoded.height, 320);
const intrabcReference = await decodePng(Buffer.from(
  readFileSync('testimages/gt_avif_intrabc.png.b64', 'utf8').replace(/\s/g, ''), 'base64',
));
const intrabcPsnr = rgbPsnr(intrabcDecoded.data, intrabcReference.rgba);
assert.ok(intrabcPsnr > 40, `AVIF IntraBC RGB PSNR is too low: ${intrabcPsnr.toFixed(2)} dB`);
console.log(`ok AVIF IntraBC: 240x320, RGB PSNR ${intrabcPsnr.toFixed(2)} dB`);

const multiTileEncoded = new Uint8Array(Buffer.from(
  readFileSync('testimages/avif_multitile_multigroup.avif.b64', 'utf8').replace(/\s/g, ''), 'base64',
));
const multiTileDecoded = decodeToRgba(multiTileEncoded);
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
const superResDecoded = decodeToRgba(superResEncoded);
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
  const decoded = decodeToRgba(encoded);
  assert.equal(decoded.width, 320);
  assert.equal(decoded.height, 240);
  const reference = readBase64Fixture(`testimages/gt_avif_${layout}.rgb.b64`);
  const psnr = sampledRgbPsnr(decoded.data, 320, reference, 20, 15, 16);
  assert.ok(psnr > minimumPsnr, `AVIF 4:${layout[1]}:${layout[2]} RGB PSNR is too low: ${psnr.toFixed(2)} dB`);
  console.log(`ok AVIF 4:${layout[1]}:${layout[2]}: 320x240, sampled RGB PSNR ${psnr.toFixed(2)} dB`);
}

const filmGrainEncoded = readBase64Fixture('testimages/avif_filmgrain.avif.b64');
const filmGrainDecoded = decodeToRgba(filmGrainEncoded);
const filmGrainReference = readBase64Fixture('testimages/gt_avif_filmgrain.rgb.b64');
const filmGrainPsnr = sampledRgbPsnr(filmGrainDecoded.data, 512, filmGrainReference, 32, 24, 16);
assert.ok(filmGrainPsnr > 40, `AVIF film-grain RGB PSNR is too low: ${filmGrainPsnr.toFixed(2)} dB`);
console.log(`ok AVIF film grain: 512x384, sampled RGB PSNR ${filmGrainPsnr.toFixed(2)} dB`);

const restorationEncoded = readBase64Fixture('testimages/avif_restoration.avif.b64');
const restorationFile = new HeifFile().parse(restorationEncoded);
const restorationSyntax = new Av1Decoder().decode(restorationFile.primary.data);
assert.equal(restorationSyntax.blocks.length, 1835);
assert.equal(restorationSyntax.finalRange, 37051);
assert.deepEqual(
  [0, 2, 3].map(type => restorationSyntax.restorationUnits.filter(unit => unit.type === type).length),
  [37, 31, 4],
);
const restorationDecoded = decodeToRgba(restorationEncoded);
const restorationReference = readBase64Fixture('testimages/gt_avif_restoration.rgb.b64');
const restorationPsnr = sampledRgbPsnr(restorationDecoded.data, 2048, restorationReference, 32, 24, 64);
assert.ok(restorationPsnr > 15, `AVIF restoration RGB PSNR is too low: ${restorationPsnr.toFixed(2)} dB`);
console.log(`ok AVIF restoration syntax/range: 2048x1536, sampled RGB PSNR ${restorationPsnr.toFixed(2)} dB`);

assert.throws(
  () => decodeToRgba(new Uint8Array(32)),
  (error: unknown) => error instanceof DecodeError && error.code === 'UNSUPPORTED_FORMAT',
);
console.log('ok invalid input error');

assert.throws(
  () => decodeToRgba(new Uint8Array(readFileSync('testimages/avif_a_8.avif')), { maxPixels: 100 }),
  (error: unknown) => error instanceof DecodeError && error.code === 'RESOURCE_LIMIT',
);
console.log('ok resource limit error');

assert.throws(
  () => decodeToRgba(new Uint8Array(readFileSync('testimages/avif_a_8.avif')), { maxPixels: 0 }),
  (error: unknown) => error instanceof DecodeError && error.code === 'INVALID_INPUT',
);
console.log('ok invalid decode options error');

const invalidAv1C = new Uint8Array(readFileSync('testimages/avif_a_8.avif'));
const invalidAv1CFile = new HeifFile().parse(invalidAv1C);
const invalidAv1COffset = invalidAv1CFile.primary.config!.byteOffset - invalidAv1C.byteOffset;
invalidAv1C[invalidAv1COffset + 3] |= 0x80;
assert.throws(
  () => decodeToRgba(invalidAv1C),
  (error: unknown) => error instanceof DecodeError && error.code === 'DECODE_FAILED' && /av1C/.test(error.message),
);
console.log('ok invalid av1C error');

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
