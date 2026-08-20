import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const entries = [
  ['heif-js', ['decode', 'default']],
  ['heif-js/detect', ['DEFAULT_DECODE_LIMITS', 'HeifFile', 'detectFormat']],
  ['heif-js/heic', ['decode', 'default']],
  ['heif-js/avif', ['decode', 'default']],
];
let esmRoot;
let cjsRoot;

for (const [specifier, expectedExports] of entries) {
  const esm = await import(specifier);
  const cjs = require(specifier);
  for (const name of expectedExports) {
    assert.ok(name in esm, `${specifier} ESM is missing ${name}`);
    assert.ok(name in cjs, `${specifier} CommonJS is missing ${name}`);
  }
  if ('decode' in esm) {
    assert.equal(esm.default, esm.decode);
    assert.equal(cjs.default, cjs.decode);
  }
  if (specifier === 'heif-js') {
    esmRoot = esm;
    cjsRoot = cjs;
  }
}

const umdContext = vm.createContext({});
vm.runInContext(readBundle('heif-js.umd.js'), umdContext, { filename: 'heif-js.umd.js' });
assertDecoderNamespace(umdContext.HeifJS, 'UMD');

let amdNamespace;
function define(name, dependencies, factory) {
  assert.equal(name, 'heif-js');
  assert.ok(Array.isArray(dependencies));
  assert.equal(dependencies.length, 0);
  amdNamespace = factory();
}
define.amd = {};
const amdContext = vm.createContext({ define });
vm.runInContext(readBundle('heif-js.amd.js'), amdContext, { filename: 'heif-js.amd.js' });
assertDecoderNamespace(amdNamespace, 'AMD');

for (const fixture of ['heic_a.heic', 'avif_a_8.avif']) {
  const encoded = new Uint8Array(readFileSync(new URL(`../testimages/${fixture}`, import.meta.url)));
  const expected = esmRoot.decode(encoded);
  for (const [format, namespace] of [
    ['CommonJS', cjsRoot],
    ['AMD', amdNamespace],
    ['UMD', umdContext.HeifJS],
  ]) {
    const actual = namespace.decode(encoded);
    assert.equal(actual.width, expected.width, `${format} ${fixture} width differs`);
    assert.equal(actual.height, expected.height, `${format} ${fixture} height differs`);
    assert.equal(checksum(actual.data), checksum(expected.data), `${format} ${fixture} pixels differ`);
  }
}

console.log('ok packaged ESM, CommonJS, AMD, and UMD entry points');

function readBundle(name) {
  return readFileSync(new URL(`../dist/browser/${name}`, import.meta.url), 'utf8');
}

function assertDecoderNamespace(namespace, format) {
  assert.ok(namespace, `${format} namespace is missing`);
  assert.equal(typeof namespace.decode, 'function');
  assert.equal(namespace.default, namespace.decode);
}

function checksum(bytes) {
  let hash = 2166136261;
  for (const byte of bytes) hash = Math.imul(hash ^ byte, 16777619) >>> 0;
  return hash;
}
