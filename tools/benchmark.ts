import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { decode } from '../src/index.ts';
import { HeifFile } from '../src/bmff.ts';
import { Av1Decoder } from '../src/av1/decode.ts';
import { frameToRgba } from '../src/color.ts';

const iterations = positiveInteger(process.env.BENCH_ITERATIONS, 9);
const warmups = positiveInteger(process.env.BENCH_WARMUPS, 3);
const memoryMode = process.env.BENCH_MEMORY === '1';
let checksum = 0;

interface Result {
  name: string;
  medianMs: number;
  p90Ms: number;
  megapixelsPerSecond: number;
  dimensions: string;
  outputMiB?: number;
  rssMiB?: number;
}

const fixtures = [
  ['HEVC 8-bit', 'testimages/heic_a.heic', 320, 240],
  ['AV1 8-bit', 'testimages/avif_a_8.avif', 320, 240],
  ['AV1 10-bit directional', 'testimages/avif_c_10.avif', 256, 256],
  ['AV1 quant matrices', 'testimages/avif_qm.avif.b64', 512, 384],
  ['AV1 IntraBC + irot', 'testimages/avif_intrabc.avif.b64', 240, 320],
  ['AV1 multi-tile', 'testimages/avif_multitile_multigroup.avif.b64', 512, 384],
  ['AV1 super-resolution', 'testimages/avif_superres.avif.b64', 512, 384],
  ['AV1 film grain', 'testimages/avif_filmgrain.avif.b64', 512, 384],
  ['HEIF alpha + prem', 'testimages/heif_prem.heic.b64', 48, 40],
  ['AV1 restoration 3MP', 'testimages/avif_restoration.avif.b64', 2048, 1536],
] as const;

const results: Result[] = [];
for (const [name, path, width, height] of fixtures) {
  const encoded = readFixture(path);
  results.push(benchmark(name, width, height, () => decode(encoded).data));
}

// Isolate the hot YUV-to-RGBA stage from entropy decoding and loop filters.
const restorationBytes = readFixture('testimages/avif_restoration.avif.b64');
const restorationFile = new HeifFile().parse(restorationBytes);
const restoration = new Av1Decoder().decode(restorationFile.primary.data);
const sequence = restoration.sequence;
const item = restorationFile.primary;
results.push(benchmark('RGBA conversion 3MP', restoration.frame.width, restoration.frame.height, () =>
  frameToRgba(
    restoration.frame,
    item.nclx?.matrixCoefficients ?? sequence.matrixCoefficients,
    item.nclx?.fullRangeFlag ?? sequence.fullRange,
    item.nclx?.colourPrimaries ?? sequence.colorPrimaries,
    item.nclx?.transferCharacteristics ?? sequence.transferCharacteristics,
    sequence.chromaSamplePosition,
    item.icc,
  ),
));

if (memoryMode) {
  for (let index = 0; index < fixtures.length; index++) {
    Object.assign(results[index]!, measureDecodeMemory(fixtures[index]![1]));
  }
  Object.assign(results[results.length - 1]!, measureColourMemory());
}

console.table(results.map(result => ({
  case: result.name,
  dimensions: result.dimensions,
  'median ms': result.medianMs.toFixed(2),
  'p90 ms': result.p90Ms.toFixed(2),
  'MP/s': result.megapixelsPerSecond.toFixed(1),
  ...(memoryMode ? {
    'output buffer MiB': result.outputMiB!.toFixed(2),
    'RSS delta MiB': result.rssMiB!.toFixed(2),
  } : {}),
})));
console.log(`warmups=${warmups} iterations=${iterations} checksum=${checksum >>> 0}`);

function benchmark(
  name: string, width: number, height: number, operation: () => Uint8ClampedArray,
): Result {
  for (let index = 0; index < warmups; index++) consume(operation());
  const samples: number[] = [];
  for (let index = 0; index < iterations; index++) {
    const start = performance.now();
    const output = operation();
    samples.push(performance.now() - start);
    consume(output);
  }
  samples.sort((left, right) => left - right);
  const medianMs = percentile(samples, 0.5);
  const result: Result = {
    name,
    medianMs,
    p90Ms: percentile(samples, 0.9),
    megapixelsPerSecond: width * height / medianMs / 1000,
    dimensions: `${width}x${height}`,
  };
  return result;
}

function measureDecodeMemory(path: string): Pick<Result, 'outputMiB' | 'rssMiB'> {
  const script = `
    import { readFileSync } from 'node:fs';
    import { decode } from './src/index.ts';
    const path = ${JSON.stringify(path)};
    const raw = readFileSync(path);
    const bytes = path.endsWith('.b64')
      ? new Uint8Array(Buffer.from(raw.toString('utf8').replace(/\\s/g, ''), 'base64'))
      : new Uint8Array(raw);
    globalThis.__benchmarkRetained = { raw, bytes };
    globalThis.gc();
    const before = process.memoryUsage();
    const image = decode(bytes);
    globalThis.__benchmarkRetained = { raw, bytes, image };
    globalThis.gc();
    const after = process.memoryUsage();
    console.log(JSON.stringify({
      outputMiB: image.data.byteLength / 1048576,
      rssMiB: (after.rss - before.rss) / 1048576,
      retained: image.data.byteLength,
    }));
  `;
  return runMemoryChild(script);
}

function measureColourMemory(): Pick<Result, 'outputMiB' | 'rssMiB'> {
  const script = `
    import { readFileSync } from 'node:fs';
    import { HeifFile } from './src/bmff.ts';
    import { Av1Decoder } from './src/av1/decode.ts';
    import { frameToRgba } from './src/color.ts';
    const raw = readFileSync('testimages/avif_restoration.avif.b64', 'utf8');
    const file = new HeifFile().parse(new Uint8Array(Buffer.from(raw.replace(/\\s/g, ''), 'base64')));
    const decoded = new Av1Decoder().decode(file.primary.data);
    const sequence = decoded.sequence, item = file.primary;
    globalThis.__benchmarkRetained = { raw, file, decoded };
    globalThis.gc();
    const before = process.memoryUsage();
    const pixels = frameToRgba(
      decoded.frame,
      item.nclx?.matrixCoefficients ?? sequence.matrixCoefficients,
      item.nclx?.fullRangeFlag ?? sequence.fullRange,
      item.nclx?.colourPrimaries ?? sequence.colorPrimaries,
      item.nclx?.transferCharacteristics ?? sequence.transferCharacteristics,
      sequence.chromaSamplePosition,
      item.icc,
    );
    globalThis.__benchmarkRetained = { raw, file, decoded, pixels };
    globalThis.gc();
    const after = process.memoryUsage();
    console.log(JSON.stringify({
      outputMiB: pixels.byteLength / 1048576,
      rssMiB: (after.rss - before.rss) / 1048576,
      retained: pixels.byteLength,
    }));
  `;
  return runMemoryChild(script);
}

function runMemoryChild(script: string): Pick<Result, 'outputMiB' | 'rssMiB'> {
  const child = spawnSync(process.execPath, [
    '--expose-gc', '--experimental-strip-types', '--input-type=module', '-e', script,
  ], { cwd: process.cwd(), encoding: 'utf8' });
  if (child.status !== 0) throw new Error(child.stderr || `Memory child exited with status ${child.status}`);
  return JSON.parse(child.stdout.trim()) as Pick<Result, 'outputMiB' | 'rssMiB'>;
}

function percentile(sorted: readonly number[], percentileValue: number): number {
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * percentileValue) - 1)]!;
}

function consume(output: Uint8ClampedArray): void {
  if (!output.length) return;
  checksum = Math.imul(checksum ^ output[0]! ^ output[output.length - 1]!, 16_777_619) >>> 0;
}

function readFixture(path: string): Uint8Array {
  const bytes = readFileSync(path);
  return path.endsWith('.b64')
    ? new Uint8Array(Buffer.from(bytes.toString('utf8').replace(/\s/g, ''), 'base64'))
    : new Uint8Array(bytes);
}

function positiveInteger(value: string | undefined, fallback: number): number {
  const parsed = Number(value ?? fallback);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) throw new Error(`Expected a positive integer, received ${value}`);
  return parsed;
}
