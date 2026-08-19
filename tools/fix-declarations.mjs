import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

for (const file of walk('dist')) {
  if (!file.endsWith('.d.ts')) continue;
  const source = readFileSync(file, 'utf8');
  const rewritten = source.replace(/(from\s+|import\()(['"])((?:\.\.?\/)[^'"]+)\.ts\2/g,
    (_match, prefix, quote, specifier) => `${prefix}${quote}${specifier}.js${quote}`);
  if (rewritten !== source) writeFileSync(file, rewritten);
}

function* walk(directory) {
  for (const name of readdirSync(directory)) {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) yield* walk(path);
    else yield path;
  }
}
