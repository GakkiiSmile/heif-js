/** Minimal PNG writer (RGB, no compression dependencies beyond node zlib). */
import { deflateSync } from 'node:zlib';

export function encodePng(width: number, height: number, rgba: Uint8Array, rgbOut = true): Buffer {
  const bpp = rgbOut ? 3 : 4;
  const raw = Buffer.alloc((width * bpp + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width * bpp + 1)] = 0; // filter none
    for (let x = 0; x < width; x++) {
      const si = (y * width + x) * 4, di = y * (width * bpp + 1) + 1 + x * bpp;
      raw[di] = rgba[si]; raw[di + 1] = rgba[si + 1]; raw[di + 2] = rgba[si + 2];
      if (!rgbOut) raw[di + 3] = rgba[si + 3];
    }
  }
  const idat = deflateSync(raw, { level: 6 });
  const out: Buffer[] = [];
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  out.push(sig);
  const ihdr = Buffer.alloc(13);
  writeU32(ihdr, 0, width); writeU32(ihdr, 4, height);
  ihdr[8] = 8; ihdr[9] = rgbOut ? 2 : 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  out.push(chunk('IHDR', ihdr));
  out.push(chunk('IDAT', idat));
  out.push(chunk('IEND', Buffer.alloc(0)));
  return Buffer.concat(out);
}

function chunk(type: string, data: Buffer): Buffer {
  const len = Buffer.alloc(4); writeU32(len, 0, data.length);
  const typeB = Buffer.from(type, 'ascii');
  const crcBuf = Buffer.concat([typeB, data]);
  const crc = Buffer.alloc(4); writeU32(crc, 0, crc32(crcBuf));
  return Buffer.concat([len, typeB, data, crc]);
}

let crcTable: Uint32Array | null = null;
function crc32(buf: Buffer): number {
  if (!crcTable) {
    crcTable = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
      crcTable[n] = c >>> 0;
    }
  }
  let c = 0xFFFFFFFF;
  for (let i = 0; i < buf.length; i++) c = crcTable[(c ^ buf[i]) & 0xFF]! ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}

function writeU32(b: Buffer, off: number, v: number) {
  b[off] = v >>> 24 & 255; b[off + 1] = v >>> 16 & 255; b[off + 2] = v >>> 8 & 255; b[off + 3] = v & 255;
}

/** Decode a PNG (8-bit RGB/RGBA, non-interlaced) to RGBA for ground-truth comparison. */
export async function decodePng(file: string | Uint8Array): Promise<{ width: number; height: number; rgba: Uint8Array }> {
  const { readFile } = await import('node:fs/promises');
  const buf = typeof file === 'string' ? await readFile(file) : Buffer.from(file);
  let pos = 8;
  let width = 0, height = 0, colorType = 0;
  const idat: Buffer[] = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos); const type = buf.toString('ascii', pos + 4, pos + 8);
    if (type === 'IHDR') {
      width = buf.readUInt32BE(pos + 8); height = buf.readUInt32BE(pos + 12);
      colorType = buf[pos + 17];
    } else if (type === 'IDAT') idat.push(buf.subarray(pos + 8, pos + 8 + len));
    else if (type === 'IEND') break;
    pos += 12 + len;
  }
  const raw = (await import('node:zlib')).inflateSync(Buffer.concat(idat));
  const bpp = colorType === 6 ? 4 : colorType === 2 ? 3 : 1;
  const rgba = new Uint8Array(width * height * 4);
  let rp = 0;
  let previousRow: Buffer | null = null;
  for (let y = 0; y < height; y++) {
    const filter = raw[rp++];
    const row = raw.subarray(rp, rp + width * bpp); rp += width * bpp;
    // un-filter (support 0..4)
    const out = Buffer.alloc(width * bpp);
    for (let i = 0; i < width * bpp; i++) {
      const a = i >= bpp ? out[i - bpp] : 0;
      const b = previousRow ? previousRow[i] : 0;
      const c = i >= bpp && previousRow ? previousRow[i - bpp] : 0;
      let v = row[i];
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      out[i] = v & 255;
    }
    for (let x = 0; x < width; x++) {
      const s = x * bpp, d = (y * width + x) * 4;
      rgba[d] = out[s]; rgba[d + 1] = out[s + 1]; rgba[d + 2] = out[s + 2];
      rgba[d + 3] = bpp === 4 ? out[s + 3] : 255;
    }
    previousRow = out;
  }
  return { width, height, rgba };
}
