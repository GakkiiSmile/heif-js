/**
 * ISOBMFF / HEIF container parsing (ISO/IEC 14496-12 + 23008-12).
 * Extracts image items (hvc1 / av01), their properties (hvcC, av1C, ispe,
 * colr/nclx, pixi, clap, irot, imir) and raw coded data from iloc/mdat/idat.
 */

import { DEFAULT_DECODE_LIMITS, ResourceLimitError, resolveDecodeLimits } from './limits.ts';
import type { DecodeOptions, ResolvedDecodeLimits } from './limits.ts';

export interface NclxColor {
  colourPrimaries: number;
  transferCharacteristics: number;
  matrixCoefficients: number;
  fullRangeFlag: boolean;
}

export interface Clap {
  cleanApertureWidthN: number; cleanApertureWidthD: number;
  cleanApertureHeightN: number; cleanApertureHeightD: number;
  horizOffN: number; horizOffD: number;
  vertOffN: number; vertOffD: number;
}

export interface ImageItem {
  itemId: number;
  type: string;            // 'hvc1' | 'av01' | 'grid' | ...
  /**
   * Coded data (for hvc1: length-prefixed NAL units; for av01: OBU stream).
   * Extents are assembled and validated lazily on first access.
   */
  data: Uint8Array;
  /** hvcC / av1C property payload (null if absent) */
  config: Uint8Array | null;
  width: number;           // from ispe
  height: number;
  bitDepth: number;        // from pixi (0 = unknown)
  nclx: NclxColor | null;
  icc: Uint8Array | null;
  /** Auxiliary-image type from auxC (for example an alpha-plane URN). */
  auxType: string | null;
  /** Item references keyed by four-character type (dimg, auxl, prem, ...). */
  references: Record<string, number[]>;
  /** Transformative properties in their ipma association order. */
  transformations: ('clap' | 'irot' | 'imir')[];
  irot: number;            // 0..3, anti-clockwise quarter turns
  imir: number;            // 0=vertical flip, 1=horizontal flip
  clap: Clap | null;
  /** for grid items: tile ids in row-major order */
  gridTiles: number[] | null;
  gridRows: number;
  gridCols: number;
  /** Essential property types unknown to this decoder. */
  unsupportedEssentialProperties: string[];
}

class Reader {
  view: DataView;
  pos = 0;
  end: number;
  u8: Uint8Array;
  constructor(u8: Uint8Array, start = 0, end = u8.length) {
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start || end > u8.length) {
      throw new Error('HEIF: invalid reader bounds');
    }
    this.u8 = u8;
    this.view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    this.pos = start; this.end = end;
  }
  get remaining() { return this.end - this.pos; }
  u8v(): number { this.ensure(1); return this.u8[this.pos++]!; }
  u16(): number { this.ensure(2); const v = this.view.getUint16(this.pos); this.pos += 2; return v; }
  u24(): number {
    this.ensure(3);
    const v = (this.u8[this.pos]! * 0x10000) + (this.u8[this.pos + 1]! << 8) + this.u8[this.pos + 2]!;
    this.pos += 3;
    return v;
  }
  u32(): number { this.ensure(4); const v = this.view.getUint32(this.pos); this.pos += 4; return v; }
  u64(): number {
    this.ensure(8);
    const hi = this.view.getUint32(this.pos), lo = this.view.getUint32(this.pos + 4);
    this.pos += 8;
    const value = hi * 4294967296 + lo;
    if (!Number.isSafeInteger(value)) throw new Error('HEIF: 64-bit integer exceeds JavaScript safe range');
    return value;
  }
  i16(): number { this.ensure(2); const v = this.view.getInt16(this.pos); this.pos += 2; return v; }
  i32(): number { this.ensure(4); const v = this.view.getInt32(this.pos); this.pos += 4; return v; }
  bytes(n: number): Uint8Array { this.ensure(n); const v = this.u8.subarray(this.pos, this.pos + n); this.pos += n; return v; }
  skip(n: number) { this.ensure(n); this.pos += n; }
  private ensure(n: number): void {
    if (!Number.isSafeInteger(n) || n < 0 || this.pos + n > this.end) {
      throw new Error('HEIF: truncated box payload');
    }
  }
}

interface RawBox { type: string; start: number; end: number; contentStart: number }

function parseBoxes(
  u8: Uint8Array, start: number, end: number, maxBoxes = DEFAULT_DECODE_LIMITS.maxBoxes,
): RawBox[] {
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end < start || end > u8.length) {
    throw new Error('HEIF: invalid box-list bounds');
  }
  const out: RawBox[] = [];
  let pos = start;
  const v = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  while (pos < end) {
    if (end - pos < 8) {
      for (let index = pos; index < end; index++) {
        if (u8[index] !== 0) throw new Error('HEIF: truncated box header');
      }
      break;
    }
    let size = v.getUint32(pos);
    const type = String.fromCharCode(u8[pos + 4], u8[pos + 5], u8[pos + 6], u8[pos + 7]);
    let hdr = 8;
    if (size === 1) {
      if (end - pos < 16) throw new Error('HEIF: truncated extended box header');
      size = v.getUint32(pos + 8) * 4294967296 + v.getUint32(pos + 12);
      if (!Number.isSafeInteger(size)) throw new Error('HEIF: box size exceeds JavaScript safe range');
      hdr = 16;
    }
    else if (size === 0) { size = end - pos; }
    if (type === 'uuid') hdr += 16;
    if (size < hdr || pos + size > end) throw new Error(`HEIF: invalid ${type} box size`);
    if (out.length >= maxBoxes) throw new ResourceLimitError(`HEIF: box count exceeds configured limit ${maxBoxes}`);
    out.push({ type, start: pos, end: pos + size, contentStart: pos + hdr });
    pos += size;
  }
  return out;
}

function typeStr(b: Uint8Array, off: number) {
  if (!Number.isSafeInteger(off) || off < 0 || off + 4 > b.length) throw new Error('HEIF: truncated four-character code');
  return String.fromCharCode(b[off]!, b[off + 1]!, b[off + 2]!, b[off + 3]!);
}

interface Infe { itemId: number; itemType: string; }
interface LocEntry {
  itemId: number;
  constructionMethod: number;
  dataReferenceIndex: number;
  extents: { index: number; indexExplicit: boolean; offset: number; length: number }[];
}

export class HeifFile {
  primaryItemId = -1;
  items = new Map<number, ImageItem>();
  brands: string[] = [];

  parse(u8: Uint8Array, options: DecodeOptions | ResolvedDecodeLimits = {}): this {
    // A SharedArrayBuffer can be modified by another agent while box bounds are
    // being validated. Snapshot it once so all lazy item views remain stable.
    if (!(u8.buffer instanceof ArrayBuffer)) u8 = new Uint8Array(u8);
    const limits = resolveDecodeLimits(options);
    this.primaryItemId = -1;
    this.items.clear();
    this.brands.length = 0;
    const boxes = parseBoxes(u8, 0, u8.length, limits.maxBoxes);
    let metaBox: RawBox | null = null;
    for (const b of boxes) {
      if (b.type === 'ftyp') {
        const r = new Reader(u8, b.contentStart, b.end);
        this.brands.push(typeStr(u8, r.pos));
        r.skip(8); // major_brand + minor_version
        const n = Math.floor((r.end - r.pos) / 4);
        for (let i = 0; i < n; i++) this.brands.push(typeStr(u8, r.pos)), r.skip(4);
      } else if (b.type === 'meta') {
        metaBox = b;
      }
    }
    if (!metaBox) throw new Error('HEIF: no meta box (not a HEIF/AVIF file?)');

    // HEIF's meta is a version-0 FullBox.
    const metaHeader = new Reader(u8, metaBox.contentStart, metaBox.end);
    const metaVersion = metaHeader.u8v();
    metaHeader.u24();
    if (metaVersion !== 0) throw new Error(`HEIF: unsupported meta box version ${metaVersion}`);
    const metaChildrenStart = metaHeader.pos;

    const infeList: Infe[] = [];
    const locList: LocEntry[] = [];
    const props: Map<number, { type: string; payload: Uint8Array }> = new Map();
    const assoc = new Map<number, { index: number; essential: boolean }[]>();
    const references = new Map<number, Record<string, number[]>>();
    let idat: Uint8Array | null = null;
    let ipcoSeen = false;

    for (const b of parseBoxes(u8, metaChildrenStart, metaBox.end, limits.maxBoxes)) {
      switch (b.type) {
        case 'pitm': {
          if (this.primaryItemId >= 0) throw new Error('HEIF: duplicate primary-item box');
          const version = new Reader(u8, b.contentStart, b.end).u8v();
          if (version > 1) throw new Error(`HEIF: unsupported pitm version ${version}`);
          const rr = new Reader(u8, b.contentStart + 4, b.end);
          this.primaryItemId = version === 0 ? rr.u16() : rr.u32();
          break;
        }
        case 'iinf': {
          const ver = new Reader(u8, b.contentStart, b.end).u8v();
          if (ver > 1) throw new Error(`HEIF: unsupported iinf version ${ver}`);
          const rr = new Reader(u8, b.contentStart + 4, b.end);
          const count = ver === 0 ? rr.u16() : rr.u32();
          if (count > limits.maxItems) throw new ResourceLimitError(`HEIF: item count exceeds configured limit ${limits.maxItems}`);
          const entries = parseBoxes(u8, rr.pos, b.end, limits.maxBoxes);
          if (entries.length < count) throw new Error('HEIF: iinf contains fewer entries than declared');
          for (const infe of entries) {
            if (infe.type !== 'infe') continue;
            const iv = new Reader(u8, infe.contentStart, infe.end).u8v();
            const ir = new Reader(u8, infe.contentStart + 4, infe.end);
            if (iv === 2 || iv === 3) {
              const itemId = iv === 2 ? ir.u16() : ir.u32();
              ir.u16(); // item_protection_index (present in both v2 and v3)
              const itemType = typeStr(u8, ir.pos); ir.skip(4);
              infeList.push({ itemId, itemType });
            }
          }
          if (infeList.length > limits.maxItems) {
            throw new ResourceLimitError(`HEIF: item count exceeds configured limit ${limits.maxItems}`);
          }
          break;
        }
        case 'iloc': {
          const ver = new Reader(u8, b.contentStart, b.end).u8v();
          if (ver > 2) throw new Error(`HEIF: unsupported iloc version ${ver}`);
          const rr = new Reader(u8, b.contentStart + 4, b.end);
          const sizes = rr.u16();
          const offsetSize = sizes >> 12, lengthSize = (sizes >> 8) & 15;
          const baseOffsetSize = (sizes >> 4) & 15, indexSize = sizes & 15;
          const itemCount = ver < 2 ? rr.u16() : rr.u32();
          if (itemCount > limits.maxItems) throw new ResourceLimitError(`HEIF: iloc item count exceeds configured limit ${limits.maxItems}`);
          for (let i = 0; i < itemCount; i++) {
            const itemId = ver < 2 ? rr.u16() : rr.u32();
            let constructionMethod = 0;
            if (ver === 1 || ver === 2) { rr.u16(); constructionMethod = rr.view.getUint16(rr.pos - 2) & 0xF; }
            const dataReferenceIndex = rr.u16();
            const baseOffset = readUint(rr, baseOffsetSize);
            const extentCount = rr.u16();
            if (extentCount > limits.maxExtentsPerItem) {
              throw new ResourceLimitError(`HEIF: extent count exceeds configured limit ${limits.maxExtentsPerItem}`);
            }
            const extents: { index: number; indexExplicit: boolean; offset: number; length: number }[] = [];
            for (let e = 0; e < extentCount; e++) {
              const indexExplicit = ver > 0 && indexSize > 0;
              const index = indexExplicit ? readUint(rr, indexSize) : 0;
              const off = readUint(rr, offsetSize);
              const len = readUint(rr, lengthSize);
              const offset = baseOffset + off;
              if (!Number.isSafeInteger(offset)) throw new Error('HEIF: iloc extent offset exceeds JavaScript safe range');
              extents.push({ index, indexExplicit, offset, length: len });
            }
            locList.push({ itemId, constructionMethod, dataReferenceIndex, extents });
          }
          break;
        }
        case 'idat': {
          if (idat) throw new Error('HEIF: duplicate idat box');
          idat = u8.subarray(b.contentStart, b.end);
          break;
        }
        case 'iref': {
          const version = new Reader(u8, b.contentStart, b.end).u8v();
          if (version > 1) throw new Error(`HEIF: unsupported iref version ${version}`);
          for (const reference of parseBoxes(u8, b.contentStart + 4, b.end, limits.maxBoxes)) {
            const rr = new Reader(u8, reference.contentStart, reference.end);
            const from = version === 0 ? rr.u16() : rr.u32();
            const count = rr.u16();
            if (count > limits.maxItems) throw new ResourceLimitError(`HEIF: reference count exceeds configured limit ${limits.maxItems}`);
            const to: number[] = [];
            for (let i = 0; i < count; i++) {
              to.push(version === 0 ? rr.u16() : rr.u32());
            }
            const byType = references.get(from) ?? {};
            const combined = [...(byType[reference.type] ?? []), ...to];
            if (combined.length > limits.maxItems) {
              throw new ResourceLimitError(`HEIF: reference count exceeds configured limit ${limits.maxItems}`);
            }
            byType[reference.type] = combined;
            references.set(from, byType);
          }
          break;
        }
        case 'iprp': {
          for (const sub of parseBoxes(u8, b.contentStart, b.end, limits.maxBoxes)) {
            if (sub.type === 'ipco') {
              if (ipcoSeen) throw new Error('HEIF: duplicate ipco property container');
              ipcoSeen = true;
              let idx = 1;
              for (const p of parseBoxes(u8, sub.contentStart, sub.end, limits.maxProperties)) {
                if (idx > limits.maxProperties) {
                  throw new ResourceLimitError(`HEIF: property count exceeds configured limit ${limits.maxProperties}`);
                }
                props.set(idx++, { type: p.type, payload: u8.subarray(p.contentStart, p.end) });
              }
            } else if (sub.type === 'ipma') {
              const full = new Reader(u8, sub.contentStart, sub.end);
              const ver = full.u8v();
              const flags = full.u24();
              if (ver > 1) throw new Error(`HEIF: unsupported ipma version ${ver}`);
              const rr = new Reader(u8, sub.contentStart + 4, sub.end);
              const entryCount = rr.u32();
              if (entryCount > limits.maxItems) {
                throw new ResourceLimitError(`HEIF: property-association count exceeds configured limit ${limits.maxItems}`);
              }
              for (let i = 0; i < entryCount; i++) {
                const itemId = ver < 1 ? rr.u16() : rr.u32();
                const n = rr.u8v();
                const list: { index: number; essential: boolean }[] = [];
                for (let j = 0; j < n; j++) {
                  const raw = flags & 1 ? rr.u16() : rr.u8v();
                  list.push({
                    index: raw & (flags & 1 ? 0x7fff : 0x7f),
                    essential: !!(raw & (flags & 1 ? 0x8000 : 0x80)),
                  });
                }
                assoc.set(itemId, [...(assoc.get(itemId) ?? []), ...list]);
              }
            }
          }
          break;
        }
      }
    }

    // assemble items
    const itemIds = new Set<number>();
    for (const { itemId, itemType } of infeList) {
      if (itemIds.has(itemId)) throw new Error(`HEIF: duplicate item id ${itemId}`);
      itemIds.add(itemId);
      const item: ImageItem = {
        itemId, type: itemType, data: new Uint8Array(0), config: null,
        width: 0, height: 0, bitDepth: 0, nclx: null, icc: null,
        auxType: null, references: references.get(itemId) ?? {}, transformations: [],
        irot: 0, imir: 0, clap: null, gridTiles: null, gridRows: 0, gridCols: 0,
        unsupportedEssentialProperties: [],
      };
      const propsIdx = assoc.get(itemId) ?? [];
      for (const association of propsIdx) {
        const p = props.get(association.index);
        if (!p) {
          if (association.index !== 0 && association.essential) {
            throw new Error(`HEIF: essential property index ${association.index} is missing`);
          }
          continue;
        }
        const pr = new Reader(p.payload, 0, p.payload.length);
        let recognized = true;
        switch (p.type) {
          case 'ispe': pr.skip(4); item.width = pr.u32(); item.height = pr.u32(); break;
          case 'pixi': pr.skip(4); {
            const numChannels = pr.u8v();
            let depth = 0;
            for (let channel = 0; channel < numChannels; channel++) depth = Math.max(depth, pr.u8v());
            item.bitDepth = depth;
            break;
          }
          case 'hvcC': case 'av1C': item.config = p.payload; break;
          case 'auxC': {
            // auxC is a FullBox followed by a NUL-terminated UTF-8 type string.
            const bytes = p.payload.subarray(4);
            const end = bytes.indexOf(0);
            item.auxType = new TextDecoder().decode(end < 0 ? bytes : bytes.subarray(0, end));
            break;
          }
          case 'colr': {
            const kind = typeStr(p.payload, 0);
            pr.skip(4);
            if (kind === 'nclx') {
              item.nclx = {
                colourPrimaries: pr.u16(),
                transferCharacteristics: pr.u16(),
                matrixCoefficients: pr.u16(),
                fullRangeFlag: !!(pr.u8v() & 0x80),
              };
            } else if (kind === 'prof' || kind === 'rICC') {
              item.icc = p.payload.subarray(4);
            } else recognized = false;
            break;
          }
          case 'irot': item.irot = pr.u8v() & 3; item.transformations.push('irot'); break;
          case 'imir': item.imir = pr.u8v() & 1; item.transformations.push('imir'); break;
          case 'clap': {
            item.clap = {
              cleanApertureWidthN: pr.u32(), cleanApertureWidthD: pr.u32(),
              cleanApertureHeightN: pr.u32(), cleanApertureHeightD: pr.u32(),
              horizOffN: pr.i32(), horizOffD: pr.u32(),
              vertOffN: pr.i32(), vertOffD: pr.u32(),
            };
            item.transformations.push('clap');
            break;
          }
          default: recognized = false; break;
        }
        if (!recognized && association.essential) item.unsupportedEssentialProperties.push(p.type);
      }
      this.items.set(itemId, item);
    }

    const locById = new Map<number, LocEntry>();
    for (const loc of locList) {
      if (locById.has(loc.itemId)) throw new Error(`HEIF: duplicate iloc entry for item ${loc.itemId}`);
      locById.set(loc.itemId, loc);
    }
    // Keep item payload assembly lazy.  Real-world HEIF files commonly carry
    // thumbnails and alternate representations that are never referenced by
    // the primary image; eagerly concatenating all of them wastes both memory
    // bandwidth and peak storage.  Capture this parse's item map so retained
    // ImageItem objects remain valid even if the HeifFile instance is reused.
    const parsedItems = new Map(this.items);
    const resolving = new Set<number>();
    const resolvedData = new Map<number, Uint8Array>();
    let totalResolvedBytes = 0;
    const resolveData = (itemId: number): Uint8Array => {
      const cached = resolvedData.get(itemId);
      if (cached) return cached;
      const item = parsedItems.get(itemId);
      if (!item) throw new Error(`HEIF: item ${itemId} referenced by iloc is missing`);
      if (resolving.has(itemId)) throw new Error('HEIF: cyclic iloc item construction');
      if (resolving.size >= limits.maxReferenceDepth) {
        throw new ResourceLimitError(`HEIF: iloc reference depth exceeds configured limit ${limits.maxReferenceDepth}`);
      }
      resolving.add(itemId);
      try {
        const loc = locById.get(itemId);
        let data: Uint8Array = new Uint8Array(0);
        if (loc) {
          if (loc.dataReferenceIndex !== 0) throw new Error('HEIF: external data references are not supported');
          if (loc.extents.length === 0) throw new Error(`HEIF: item ${itemId} has no iloc extents`);
          const chunks: Uint8Array[] = [];
          for (const extent of loc.extents) {
            let source: Uint8Array;
            const offset = extent.offset;
            if (loc.constructionMethod === 0) source = u8;
            else if (loc.constructionMethod === 1) {
              if (!idat) throw new Error(`HEIF: item ${itemId} references missing idat data`);
              source = idat;
            } else if (loc.constructionMethod === 2) {
              const targets = item.references.iloc ?? [];
              if (extent.indexExplicit && extent.index === 0) {
                throw new Error(`HEIF: item ${itemId} uses the reserved iloc extent index 0`);
              }
              const referenceIndex = extent.index || 1;
              const targetId = targets[referenceIndex - 1];
              if (targetId === undefined) throw new Error(`HEIF: item ${itemId} has an invalid iloc extent index`);
              source = resolveData(targetId);
            } else throw new Error(`HEIF: unsupported iloc construction method ${loc.constructionMethod}`);
            if (extent.length === 0 && loc.extents.length !== 1) {
              throw new Error(`HEIF: item ${itemId} has an implicit length with multiple extents`);
            }
            const length = extent.length || Math.max(0, source.length - offset);
            if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) ||
                offset < 0 || length < 0 || offset > source.length - length) {
              throw new Error(`HEIF: item ${itemId} extent is out of bounds`);
            }
            chunks.push(source.subarray(offset, offset + length));
          }
          data = concat(chunks, limits.maxItemBytes);
        }
        if (data.length > limits.maxTotalItemBytes - totalResolvedBytes) {
          throw new ResourceLimitError(
            `HEIF: cumulative assembled item data exceeds configured limit ${limits.maxTotalItemBytes}`,
          );
        }
        totalResolvedBytes += data.length;
        resolvedData.set(itemId, data);
        return data;
      } finally {
        resolving.delete(itemId);
      }
    };
    for (const [itemId, item] of parsedItems) {
      Object.defineProperty(item, 'data', {
        configurable: true,
        enumerable: true,
        get: () => resolveData(itemId),
        set: (value: Uint8Array) => { resolvedData.set(itemId, value); },
      });
    }

    for (const item of this.items.values()) {
      if (item.type !== 'grid' || item.data.length < 8) continue;
      const gr = new Reader(item.data, 0, item.data.length);
      const version = gr.u8v(), flags = gr.u8v();
      if (version !== 0) throw new Error(`HEIF: unsupported grid version ${version}`);
      item.gridRows = gr.u8v() + 1;
      item.gridCols = gr.u8v() + 1;
      if (item.gridRows * item.gridCols > limits.maxItems) {
        throw new ResourceLimitError(`HEIF: grid tile count exceeds configured limit ${limits.maxItems}`);
      }
      if (flags & 1) { item.width = gr.u32(); item.height = gr.u32(); }
      else { item.width = gr.u16(); item.height = gr.u16(); }
      item.gridTiles = item.references.dimg?.slice() ?? [];
    }
    return this;
  }

  get primary(): ImageItem {
    const it = this.items.get(this.primaryItemId);
    if (!it) throw new Error('HEIF: primary item missing');
    return it;
  }
}

function readUint(r: Reader, size: number): number {
  switch (size) {
    case 0: return 0;
    case 1: return r.u8v();
    case 2: return r.u16();
    case 4: return r.u32();
    case 8: return r.u64();
    default: throw new Error(`iloc: unsupported field size ${size}`);
  }
}

function concat(chunks: Uint8Array[], maximumLength: number): Uint8Array {
  let len = 0;
  for (const c of chunks) {
    if (c.length > maximumLength - len) {
      throw new ResourceLimitError(`HEIF: assembled item exceeds configured limit ${maximumLength} bytes`);
    }
    len += c.length;
  }
  // The overwhelmingly common HEIF layout stores an item in one contiguous
  // extent.  Keep that extent as a view of the input instead of copying the
  // complete compressed payload before the codec immediately scans it again.
  if (chunks.length === 1) return chunks[0]!;
  const out = new Uint8Array(len);
  let pos = 0;
  for (const c of chunks) { out.set(c, pos); pos += c.length; }
  return out;
}

/** Check whether a brand marks HEIC/HEIF/AVIF-ish files. */
export function detectFormat(u8: Uint8Array): 'heic' | 'heif' | 'avif' | 'unknown' {
  if (u8.length < 12) return 'unknown';
  if (typeStr(u8, 4) !== 'ftyp') return 'unknown';
  const boxSize = Math.min(new DataView(u8.buffer, u8.byteOffset, u8.byteLength).getUint32(0), u8.length);
  const brands = [typeStr(u8, 8)];
  for (let offset = 16; offset + 4 <= boxSize; offset += 4) brands.push(typeStr(u8, offset));
  if (brands.some(brand => brand === 'avif' || brand === 'avis')) return 'avif';
  if (brands.some(brand => brand.startsWith('hei') || brand.startsWith('hev'))) return 'heic';
  if (brands.some(brand => brand === 'mif1' || brand === 'msf1' || brand === 'mif2')) return 'heif';
  return 'unknown';
}
