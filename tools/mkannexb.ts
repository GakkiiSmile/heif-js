/** HEIC -> Annex-B .hevc elementary stream (for reference decoder tracing). */
import { readFileSync, writeFileSync } from 'node:fs';
import { HeifFile } from '../src/bmff.ts';
import { parseHvcC, nalsFromLengthPrefixed } from '../src/hevc/nal.ts';

const file = process.argv[2] ?? 'testimages/heic_b.heic';
const out = process.argv[3] ?? 'out/stream.hevc';
const u8 = new Uint8Array(readFileSync(file));
const heif = new HeifFile().parse(u8);
const item = heif.primary;
const hvcC = item.config ? parseHvcC(item.config) : { lengthSize: 4, paramSets: [] };
const nals = [...hvcC.paramSets, ...nalsFromLengthPrefixed(item.data, hvcC.lengthSize)];
const chunks: Uint8Array[] = [];
for (const n of nals) {
  // rebuild NAL from rbsp is lossy; use original bytes: re-extract with raw payload instead
  void n;
}
// need raw NAL bytes (before unescape) — re-extract without unescaping
{
  const d = item.data;
  let pos = 0;
  const ls = hvcC.lengthSize;
  while (pos + ls <= d.length) {
    let len = 0;
    for (let i = 0; i < ls; i++) len = len * 256 + d[pos + i];
    pos += ls;
    if (len === 0 || pos + len > d.length) break;
    chunks.push(new Uint8Array([0, 0, 0, 1]), d.subarray(pos, pos + len));
    pos += len;
  }
  for (const p of hvcC.paramSets) {
    void p;
  }
}
// param sets come from hvcC — re-extract raw from config
{
  const cfg = item.config!;
  let pos = 23;
  const numArrays = cfg[22]!;
  const out2: Uint8Array[] = [];
  for (let a = 0; a < numArrays; a++) {
    pos += 1;
    const numNals = (cfg[pos]! << 8) | cfg[pos + 1]!;
    pos += 2;
    for (let n = 0; n < numNals; n++) {
      const length = (cfg[pos]! << 8) | cfg[pos + 1]!;
      pos += 2;
      out2.push(new Uint8Array([0, 0, 0, 1]), cfg.subarray(pos, pos + length));
      pos += length;
    }
  }
  chunks.unshift(...out2);
}
const total = chunks.reduce((s, c) => s + c.length, 0);
const buf = new Uint8Array(total);
let o = 0;
for (const c of chunks) { buf.set(c, o); o += c.length; }
writeFileSync(out, buf);
console.log('wrote', out, buf.length, 'bytes,', chunks.length / 2, 'NALs');
