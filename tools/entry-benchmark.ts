import { readFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { gzipSync } from 'node:zlib';
import ts from 'typescript';

const built = process.argv.includes('--dist') || process.env.BENCH_DIST === '1';
const directory = built ? 'dist' : 'src';
const extension = built ? 'js' : 'ts';
const entries = ['detect', 'async', 'heic', 'avif', 'index'].map(name => `${directory}/${name}.${extension}`);
const runs = Number(process.env.BENCH_IMPORT_RUNS ?? 9);
if (!Number.isSafeInteger(runs) || runs < 3) throw new Error('BENCH_IMPORT_RUNS must be an integer >= 3');

const rows = entries.map(entry => {
  const files = staticGraph(entry);
  const sources = [...files].map(file => readFileSync(file));
  const samples: { milliseconds: number; rssMiB: number }[] = [];
  for (let run = 0; run < runs; run++) samples.push(coldImport(entry));
  samples.sort((left, right) => left.milliseconds - right.milliseconds);
  const median = samples[Math.floor(samples.length / 2)]!;
  return {
    entry,
    modules: files.size,
    'source KiB': (sources.reduce((sum, source) => sum + source.byteLength, 0) / 1024).toFixed(1),
    'gzip KiB': (gzipSync(Buffer.concat(sources)).byteLength / 1024).toFixed(1),
    'median import ms': median.milliseconds.toFixed(2),
    'RSS MiB': median.rssMiB.toFixed(2),
  };
});

const heicFiles = staticGraph(`${directory}/heic.${extension}`);
const avifFiles = staticGraph(`${directory}/avif.${extension}`);
if ([...heicFiles].some(file => file.includes('/av1/'))) throw new Error('HEIC entry statically reaches AV1');
if ([...avifFiles].some(file => file.includes('/hevc/'))) throw new Error('AVIF entry statically reaches HEVC');

console.table(rows);
console.log(`runs=${runs}; codec-specific static graphs are disjoint`);

function staticGraph(entry: string): Set<string> {
  const seen = new Set<string>();
  const visit = (path: string): void => {
    path = resolve(path);
    if (seen.has(path)) return;
    seen.add(path);
    const source = readFileSync(path, 'utf8');
    const file = ts.createSourceFile(path, source, ts.ScriptTarget.ESNext, false, ts.ScriptKind.TS);
    for (const statement of file.statements) {
      let specifier: ts.Expression | undefined;
      if (ts.isImportDeclaration(statement)) {
        if (statement.importClause?.isTypeOnly || importIsTypesOnly(statement.importClause)) continue;
        specifier = statement.moduleSpecifier;
      } else if (ts.isExportDeclaration(statement)) {
        if (statement.isTypeOnly || exportIsTypesOnly(statement.exportClause)) continue;
        specifier = statement.moduleSpecifier;
      }
      if (!specifier || !ts.isStringLiteral(specifier) || !specifier.text.startsWith('.')) continue;
      visit(resolve(dirname(path), specifier.text));
    }
  };
  visit(entry);
  return seen;
}

function importIsTypesOnly(clause: ts.ImportClause | undefined): boolean {
  if (!clause || clause.name || !clause.namedBindings || !ts.isNamedImports(clause.namedBindings)) return false;
  return clause.namedBindings.elements.length > 0 && clause.namedBindings.elements.every(element => element.isTypeOnly);
}

function exportIsTypesOnly(clause: ts.NamedExportBindings | undefined): boolean {
  return !!clause && ts.isNamedExports(clause) && clause.elements.length > 0 &&
    clause.elements.every(element => element.isTypeOnly);
}

function coldImport(entry: string): { milliseconds: number; rssMiB: number } {
  const code = `
    const start = performance.now();
    await import(${JSON.stringify(`./${entry}`)});
    console.log(JSON.stringify({
      milliseconds: performance.now() - start,
      rssMiB: process.memoryUsage().rss / 1048576,
    }));
  `;
  const args = built
    ? ['--input-type=module', '-e', code]
    : ['--experimental-strip-types', '--input-type=module', '-e', code];
  const child = spawnSync(process.execPath, args, { cwd: process.cwd(), encoding: 'utf8' });
  if (child.status !== 0) throw new Error(child.stderr || `Import child exited with ${child.status}`);
  return JSON.parse(child.stdout.trim()) as { milliseconds: number; rssMiB: number };
}
