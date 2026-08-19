import { readdirSync, readFileSync, statSync, unlinkSync, writeFileSync } from 'node:fs';
import { dirname, join, normalize } from 'node:path';

const declarations = new Map();
for (const file of walk('dist')) {
  if (!file.endsWith('.d.ts')) continue;
  const source = readFileSync(file, 'utf8');
  const rewritten = source.replace(/(from\s+|import\()(['"])((?:\.\.?\/)[^'"]+)\.ts\2/g,
    (_match, prefix, quote, specifier) => `${prefix}${quote}${specifier}.js${quote}`);
  declarations.set(normalize(file), rewritten);
  if (rewritten !== source) writeFileSync(file, rewritten);
}

// Retain the transitive type graph of every public package entry. A fixed
// allow-list silently breaks as soon as a new entry re-exports a shared public
// type, so derive roots from package.json and follow relative .d.ts imports.
const manifest = JSON.parse(readFileSync('package.json', 'utf8'));
const roots = new Set();
if (typeof manifest.types === 'string') roots.add(packagePath(manifest.types));
collectExportTypes(manifest.exports, roots);

const retained = new Set();
const queue = [...roots];
while (queue.length) {
  const file = queue.pop();
  if (!file || retained.has(file)) continue;
  const source = declarations.get(file);
  if (source === undefined) throw new Error(`Public declaration entry is missing: ${file}`);
  retained.add(file);
  const imports = /(?:from\s+|import\()\s*['"](\.\.?\/[^'"]+)\.js['"]/g;
  for (let match; (match = imports.exec(source));) {
    const dependency = normalize(join(dirname(file), `${match[1]}.d.ts`));
    if (!declarations.has(dependency)) {
      throw new Error(`Declaration dependency ${dependency} referenced by ${file} is missing`);
    }
    if (!retained.has(dependency)) queue.push(dependency);
  }
}

for (const file of declarations.keys()) if (!retained.has(file)) unlinkSync(file);

function collectExportTypes(value, output) {
  if (!value || typeof value !== 'object') return;
  if (typeof value.types === 'string') output.add(packagePath(value.types));
  for (const child of Object.values(value)) collectExportTypes(child, output);
}

function packagePath(path) {
  return normalize(path.startsWith('./') ? path.slice(2) : path);
}

function* walk(directory) {
  for (const name of readdirSync(directory)) {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) yield* walk(path);
    else yield path;
  }
}
