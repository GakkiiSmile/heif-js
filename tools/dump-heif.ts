import { readFileSync } from 'node:fs';
import { HeifFile } from '../src/bmff.ts';

const file = process.argv[2] ?? 'testimages/heic_a.heic';
const u8 = new Uint8Array(readFileSync(file));
const heif = new HeifFile().parse(u8);
console.log('brands:', heif.brands.join(' '), 'primary:', heif.primaryItemId);
for (const [id, it] of heif.items) {
  console.log(`item ${id}: type=${it.type} ${it.width}x${it.height} bitDepth=${it.bitDepth} data=${it.data.length}B config=${it.config?.length ?? '-'}B`,
    it.nclx ? `nclx(${it.nclx.colourPrimaries}/${it.nclx.transferCharacteristics}/${it.nclx.matrixCoefficients} range=${it.nclx.fullRangeFlag ? 'full' : 'limited'})` : '',
    it.irot ? `irot=${it.irot}` : '', it.clap ? 'clap' : '',
    it.gridTiles ? `grid ${it.gridRows}x${it.gridCols} tiles=${it.gridTiles.join(',')}` : '');
}
