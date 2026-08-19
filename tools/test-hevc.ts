/** HEIC decode test: container -> HEVC decode -> color convert -> PNG + PSNR. */
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { HeifFile } from '../src/bmff.ts';
import { parseHvcC, nalsFromLengthPrefixed } from '../src/hevc/nal.ts';
import { HevcDecoder } from '../src/hevc/decode.ts';
import { frameToRgba } from '../src/color.ts';
import { encodePng, decodePng } from './png.ts';

const file = process.argv[2] ?? 'testimages/heic_b.heic';
const gtFile = process.argv[3] ?? file.replace(/heic_[a-z0-9]+/, 'gt_heic_$&'.replace('gt_heic_heic_', 'gt_heic_'));

const u8 = new Uint8Array(readFileSync(file));
const heif = new HeifFile().parse(u8);
const item = heif.primary;
console.log(`item: ${item.type} ${item.width}x${item.height} nclx=${JSON.stringify(item.nclx)}`);

const hvcC = item.config ? parseHvcC(item.config) : { lengthSize: 4, paramSets: [] };
const sliceNals = nalsFromLengthPrefixed(item.data, hvcC.lengthSize);

const dec = new HevcDecoder();
dec.registerParamSets(hvcC.paramSets);
// also register any in-band parameter sets
dec.registerParamSets(sliceNals.filter(n => n.type === 33 || n.type === 34));

const slice = sliceNals.filter(n => n.type < 32 || (n.type >= 16 && n.type <= 21));
console.log('slice NALs:', slice.map(n => n.type).join(','));

const t0 = performance.now();
const frame = dec.decodeFrame(slice);
console.log(`decoded in ${(performance.now() - t0).toFixed(1)}ms: ${frame.width}x${frame.height} bd=${frame.bitDepth} cf=${frame.chromaFormat}`);

const rgba = frameToRgba(frame, item.nclx?.matrixCoefficients ?? 2, item.nclx?.fullRangeFlag ?? false);
mkdirSync('out', { recursive: true });
const outName = `out/${file.split('/').pop()!.replace(/\.[a-z0-9]+$/, '')}.png`;
writeFileSync(outName, encodePng(frame.width, frame.height, rgba));
console.log('wrote', outName);

// raw YUV dump for pixel-level diffing against dec265 output
if (process.env.DUMP_YUV) {
  const { Plane } = await import('../src/frame.ts');
  void Plane;
  const yuv = new Uint8Array(frame.width * frame.height * 3 / 2);
  let o = 0;
  for (const p of frame.planes) {
    for (let y = 0; y < p.height; y++)
      for (let x = 0; x < p.width; x++) yuv[o++] = p.data[y * p.stride + x]!;
  }
  writeFileSync(process.env.DUMP_YUV, yuv);
  console.log('dumped YUV to', process.env.DUMP_YUV);
}

// PSNR vs ground truth (if exists)
try {
  const base = file.split('/').pop()!.replace('.heic', '');
  const gt = await decodePng(`testimages/gt_${base}.png`);
  let mse = 0;
  for (let i = 0; i < rgba.length; i += 4) {
    mse += (rgba[i]! - gt.rgba[i]!) ** 2;
  }
  mse /= rgba.length / 4;
  const psnr = 10 * Math.log10(255 * 255 / mse);
  console.log(`PSNR vs ground truth: ${psnr.toFixed(2)} dB`);
} catch (e) {
  console.log('(no ground truth comparison)', (e as Error).message);
}
