import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { isMainThread, parentPort, Worker } from 'node:worker_threads';
import { decodeToRgba } from '../src/index.ts';

const ITERATIONS = 300;

if (isMainThread) {
  const worker = new Worker(new URL(import.meta.url), { workerData: null });
  const timeout = setTimeout(() => {
    void worker.terminate();
    process.stderr.write('fuzz smoke test exceeded 30 seconds\n');
    process.exitCode = 1;
  }, 30_000);
  worker.on('message', result => {
    clearTimeout(timeout);
    console.log(`ok fuzz smoke: ${result.cases} mutated inputs (${result.decoded} still decoded)`);
  });
  worker.on('error', error => {
    clearTimeout(timeout);
    throw error;
  });
  worker.on('exit', code => {
    clearTimeout(timeout);
    if (code && process.exitCode === undefined) process.exitCode = code;
  });
} else {
  const corpus = [
    new Uint8Array(readFileSync('testimages/heic_a.heic')),
    new Uint8Array(readFileSync('testimages/avif_a_8.avif')),
    new Uint8Array(readFileSync('testimages/avif_c_10.avif')),
  ];
  let state = 0x9e3779b9;
  const random = (): number => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return state >>> 0;
  };
  let cases = 0, decoded = 0;
  for (const seed of corpus) {
    for (let iteration = 0; iteration < ITERATIONS; iteration++, cases++) {
      let bytes = seed.slice();
      const strategy = iteration % 4;
      if (strategy === 0) {
        const length = 12 + random() % Math.max(1, bytes.length - 11);
        bytes = bytes.slice(0, length);
      } else {
        const mutations = 1 + random() % 6;
        for (let mutation = 0; mutation < mutations; mutation++) {
          // Preserve enough of ftyp to exercise deep container/codec paths.
          const position = 16 + random() % Math.max(1, bytes.length - 16);
          if (strategy === 1) bytes[position] ^= 1 << (random() & 7);
          else if (strategy === 2) bytes[position] = random() & 0xff;
          else {
            const value = mutation & 1 ? 0xff : 0;
            bytes.fill(value, position, Math.min(bytes.length, position + 4));
          }
        }
      }
      try {
        const image = decodeToRgba(bytes, {
          maxDimension: 4096,
          maxPixels: 16 * 1024 * 1024,
          maxTotalPixels: 32 * 1024 * 1024,
          maxItemBytes: 32 * 1024 * 1024,
        });
        assert.ok(image.width > 0 && image.height > 0);
        assert.equal(image.data.length, image.width * image.height * 4);
        decoded++;
      } catch (error) {
        assert.ok(error instanceof Error);
      }
    }
  }
  parentPort!.postMessage({ cases, decoded });
}
