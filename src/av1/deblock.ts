import type { DecodedFrame, Plane } from '../frame.ts';
import type { Av1DecodedBlock } from './decode.ts';
import type { Av1FrameHeader, Av1SequenceHeader } from './obu.ts';
import { block_dimensions } from './tables_data.ts';
import { transformSizes } from './tables.ts';

interface FilterMap {
  width4: number;
  height4: number;
  block: Int32Array;
  txW: Uint8Array;
  txH: Uint8Array;
  verticalTx: Uint8Array;
  horizontalTx: Uint8Array;
  levelVertical: Uint8Array;
  levelHorizontal: Uint8Array;
}

/** Apply AV1's in-loop deblocking filter for key/intra still frames. */
export function applyDeblock(
  frame: DecodedFrame, blocks: readonly Av1DecodedBlock[],
  sequence: Av1SequenceHeader, header: Av1FrameHeader,
): void {
  if (!header.loopFilterLevels.some(Boolean)) return;
  const maps = frame.planes.map(plane => createMap(plane));
  const ssX = sequence.subsamplingX, ssY = sequence.subsamplingY;

  blocks.forEach((block, blockId) => {
    const dimensions = block_dimensions[block.blockSize]!;
    fillBlockMap(maps[0]!, block, blockId, block.x4, block.y4,
      Math.min(dimensions[0]!, maps[0]!.width4 - block.x4),
      Math.min(dimensions[1]!, maps[0]!.height4 - block.y4), false, 0, 0, header);

    if (frame.planes.length === 1) return;
    const hasChroma = (dimensions[0]! > ssX || !!(block.x4 & ssX)) &&
      (dimensions[1]! > ssY || !!(block.y4 & ssY));
    if (!hasChroma) return;
    const x4 = block.x4 >> ssX, y4 = block.y4 >> ssY;
    const width4 = Math.min((dimensions[0]! + ssX) >> ssX, maps[1]!.width4 - x4);
    const height4 = Math.min((dimensions[1]! + ssY) >> ssY, maps[1]!.height4 - y4);
    fillBlockMap(maps[1]!, block, blockId, x4, y4, width4, height4, true, 1, 2, header);
    fillBlockMap(maps[2]!, block, blockId, x4, y4, width4, height4, true, 2, 3, header);
  });

  for (let planeIndex = 0; planeIndex < frame.planes.length; planeIndex++) {
    const plane = frame.planes[planeIndex]!;
    const map = maps[planeIndex]!;
    const chroma = planeIndex !== 0;
    // AV1 applies filters on column boundaries before row boundaries.
    for (let x4 = 1; x4 < map.width4; x4++) {
      for (let y4 = 0; y4 < map.height4; y4++) {
        const index = y4 * map.width4 + x4;
        const previous = index - 1;
        const blockEdge = map.block[index] !== map.block[previous];
        if (!blockEdge && !map.verticalTx[index]) continue;
        const category = Math.min(map.txW[index]!, map.txW[previous]!, chroma ? 1 : 2);
        const width = chroma ? 4 + 2 * category : 4 << category;
        const level = map.levelVertical[index] || map.levelVertical[previous]!;
        if (level) filterEdge(plane, x4 * 4, y4 * 4, true, width, level,
          header.loopFilterSharpness, sequence.bitDepth);
      }
    }
    for (let y4 = 1; y4 < map.height4; y4++) {
      for (let x4 = 0; x4 < map.width4; x4++) {
        const index = y4 * map.width4 + x4;
        const previous = index - map.width4;
        const blockEdge = map.block[index] !== map.block[previous];
        if (!blockEdge && !map.horizontalTx[index]) continue;
        const category = Math.min(map.txH[index]!, map.txH[previous]!, chroma ? 1 : 2);
        const width = chroma ? 4 + 2 * category : 4 << category;
        const level = map.levelHorizontal[index] || map.levelHorizontal[previous]!;
        if (level) filterEdge(plane, x4 * 4, y4 * 4, false, width, level,
          header.loopFilterSharpness, sequence.bitDepth);
      }
    }
  }
}

function createMap(plane: Plane): FilterMap {
  const width4 = Math.ceil(plane.width / 4), height4 = Math.ceil(plane.height / 4);
  const size = width4 * height4;
  return {
    width4, height4,
    block: new Int32Array(size).fill(-1),
    txW: new Uint8Array(size),
    txH: new Uint8Array(size),
    verticalTx: new Uint8Array(size),
    horizontalTx: new Uint8Array(size),
    levelVertical: new Uint8Array(size),
    levelHorizontal: new Uint8Array(size),
  };
}

function fillBlockMap(
  map: FilterMap, block: Av1DecodedBlock, blockId: number,
  x4: number, y4: number, width4: number, height4: number,
  chroma: boolean, planeIndex: number, levelSlot: number, header: Av1FrameHeader,
): void {
  if (width4 <= 0 || height4 <= 0) return;
  const verticalLevel = filterLevel(block, chroma ? levelSlot : 0, header);
  const horizontalLevel = filterLevel(block, chroma ? levelSlot : 1, header);
  const tx = chroma ? block.uvTx : block.tx;
  const txInfo = transformSizes[tx]!;
  for (let y = 0; y < height4; y++) {
    for (let x = 0; x < width4; x++) {
      const index = (y4 + y) * map.width4 + x4 + x;
      map.block[index] = blockId;
      map.txW[index] = Math.min(txInfo.logW, chroma ? 1 : 2);
      map.txH[index] = Math.min(txInfo.logH, chroma ? 1 : 2);
      map.levelVertical[index] = verticalLevel;
      map.levelHorizontal[index] = horizontalLevel;
    }
  }

  if (!chroma && block.intrabc && block.yCoefficients.length) {
    for (const unit of block.yCoefficients) {
      const info = transformSizes[unit.tx]!;
      const ux = unit.x4, uy = unit.y4;
      for (let y = uy; y < Math.min(map.height4, uy + info.h4); y++) {
        for (let x = ux; x < Math.min(map.width4, ux + info.w4); x++) {
          const index = y * map.width4 + x;
          map.txW[index] = Math.min(info.logW, 2);
          map.txH[index] = Math.min(info.logH, 2);
        }
      }
      if (!block.skip) {
        if (ux > x4) for (let y = uy; y < Math.min(map.height4, uy + info.h4); y++) {
          map.verticalTx[y * map.width4 + ux] = 1;
        }
        if (uy > y4) for (let x = ux; x < Math.min(map.width4, ux + info.w4); x++) {
          map.horizontalTx[uy * map.width4 + x] = 1;
        }
      }
    }
  } else if (!block.intrabc || !block.skip) {
    for (let x = txInfo.w4; x < width4; x += txInfo.w4) {
      for (let y = 0; y < height4; y++) map.verticalTx[(y4 + y) * map.width4 + x4 + x] = 1;
    }
    for (let y = txInfo.h4; y < height4; y += txInfo.h4) {
      for (let x = 0; x < width4; x++) map.horizontalTx[(y4 + y) * map.width4 + x4 + x] = 1;
    }
  }
  void planeIndex;
}

function filterLevel(block: Av1DecodedBlock, slot: number, header: Av1FrameHeader): number {
  const baseLevel = header.loopFilterLevels[slot] ?? 0;
  if (!baseLevel) return 0;
  const deltaIndex = header.deltaLfMulti ? slot : 0;
  const segment = header.segments[block.segmentId]!;
  const segmentDelta = slot === 0 ? segment.deltaLfYVertical :
    slot === 1 ? segment.deltaLfYHorizontal : slot === 2 ? segment.deltaLfU : segment.deltaLfV;
  let level = clamp(baseLevel + block.deltaLf[deltaIndex]!, 0, 63);
  level = clamp(level + segmentDelta, 0, 63);
  if (header.loopFilterModeRefDeltaEnabled) {
    const shift = level >= 32 ? 1 : 0;
    level = clamp(level + header.loopFilterRefDeltas[0]! * (1 << shift), 0, 63);
  }
  return level;
}

function filterEdge(
  plane: Plane, x0: number, y0: number, vertical: boolean, requestedWidth: number,
  level: number, sharpness: number, bitDepth: number,
): void {
  const availableBefore = vertical ? x0 : y0;
  const availableAfter = vertical ? plane.width - x0 : plane.height - y0;
  let width = requestedWidth;
  if (width >= 16 && (availableBefore < 7 || availableAfter < 7)) width = 8;
  if (width >= 8 && (availableBefore < 4 || availableAfter < 4)) width = 6;
  if (width >= 6 && (availableBefore < 3 || availableAfter < 3)) width = 4;
  if (availableBefore < 2 || availableAfter < 2) return;

  let interiorLimit = level;
  if (sharpness > 0) {
    interiorLimit >>= (sharpness + 3) >> 2;
    interiorLimit = Math.min(interiorLimit, 9 - sharpness);
  }
  interiorLimit = Math.max(interiorLimit, 1);
  const edgeLimit = 2 * (level + 2) + interiorLimit;
  const hevThreshold = level >> 4;
  const shift = bitDepth - 8;
  const maximum = (1 << bitDepth) - 1;
  const scale = 1 << shift;
  const sample = (along: number, across: number): number => {
    const x = vertical ? x0 + across : x0 + along;
    const y = vertical ? y0 + along : y0 + across;
    return plane.data[y * plane.stride + x]!;
  };
  const store = (along: number, across: number, value: number): void => {
    const x = vertical ? x0 + across : x0 + along;
    const y = vertical ? y0 + along : y0 + across;
    plane.data[y * plane.stride + x] = clamp(value, 0, maximum);
  };

  const count = Math.min(4, vertical ? plane.height - y0 : plane.width - x0);
  for (let along = 0; along < count; along++) {
    const p1 = sample(along, -2), p0 = sample(along, -1);
    const q0 = sample(along, 0), q1 = sample(along, 1);
    let filterMask = Math.abs(p1 - p0) <= interiorLimit * scale &&
      Math.abs(q1 - q0) <= interiorLimit * scale &&
      Math.abs(p0 - q0) * 2 + (Math.abs(p1 - q1) >> 1) <= edgeLimit * scale;
    let p2 = 0, q2 = 0, p3 = 0, q3 = 0;
    if (width > 4) {
      p2 = sample(along, -3); q2 = sample(along, 2);
      filterMask &&= Math.abs(p2 - p1) <= interiorLimit * scale &&
        Math.abs(q2 - q1) <= interiorLimit * scale;
      if (width > 6) {
        p3 = sample(along, -4); q3 = sample(along, 3);
        filterMask &&= Math.abs(p3 - p2) <= interiorLimit * scale &&
          Math.abs(q3 - q2) <= interiorLimit * scale;
      }
    }
    if (!filterMask) continue;

    let flatInner = false, flatOuter = false;
    if (width >= 6) {
      flatInner = Math.abs(p2 - p0) <= scale && Math.abs(p1 - p0) <= scale &&
        Math.abs(q1 - q0) <= scale && Math.abs(q2 - q0) <= scale;
    }
    if (width >= 8) flatInner &&= Math.abs(p3 - p0) <= scale && Math.abs(q3 - q0) <= scale;
    let p4 = 0, p5 = 0, p6 = 0, q4 = 0, q5 = 0, q6 = 0;
    if (width >= 16) {
      p4 = sample(along, -5); p5 = sample(along, -6); p6 = sample(along, -7);
      q4 = sample(along, 4); q5 = sample(along, 5); q6 = sample(along, 6);
      flatOuter = Math.abs(p6 - p0) <= scale && Math.abs(p5 - p0) <= scale &&
        Math.abs(p4 - p0) <= scale && Math.abs(q4 - q0) <= scale &&
        Math.abs(q5 - q0) <= scale && Math.abs(q6 - q0) <= scale;
    }

    if (width >= 16 && flatOuter && flatInner) {
      store(along, -6, (7 * p6 + 2 * p5 + 2 * p4 + p3 + p2 + p1 + p0 + q0 + 8) >> 4);
      store(along, -5, (5 * p6 + 2 * p5 + 2 * p4 + 2 * p3 + p2 + p1 + p0 + q0 + q1 + 8) >> 4);
      store(along, -4, (4 * p6 + p5 + 2 * p4 + 2 * p3 + 2 * p2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4);
      store(along, -3, (3 * p6 + p5 + p4 + 2 * p3 + 2 * p2 + 2 * p1 + p0 + q0 + q1 + q2 + q3 + 8) >> 4);
      store(along, -2, (2 * p6 + p5 + p4 + p3 + 2 * p2 + 2 * p1 + 2 * p0 + q0 + q1 + q2 + q3 + q4 + 8) >> 4);
      store(along, -1, (p6 + p5 + p4 + p3 + p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + q2 + q3 + q4 + q5 + 8) >> 4);
      store(along, 0, (p5 + p4 + p3 + p2 + p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + q3 + q4 + q5 + q6 + 8) >> 4);
      store(along, 1, (p4 + p3 + p2 + p1 + p0 + 2 * q0 + 2 * q1 + 2 * q2 + q3 + q4 + q5 + 2 * q6 + 8) >> 4);
      store(along, 2, (p3 + p2 + p1 + p0 + q0 + 2 * q1 + 2 * q2 + 2 * q3 + q4 + q5 + 3 * q6 + 8) >> 4);
      store(along, 3, (p2 + p1 + p0 + q0 + q1 + 2 * q2 + 2 * q3 + 2 * q4 + q5 + 4 * q6 + 8) >> 4);
      store(along, 4, (p1 + p0 + q0 + q1 + q2 + 2 * q3 + 2 * q4 + 2 * q5 + 5 * q6 + 8) >> 4);
      store(along, 5, (p0 + q0 + q1 + q2 + q3 + 2 * q4 + 2 * q5 + 7 * q6 + 8) >> 4);
    } else if (width >= 8 && flatInner) {
      store(along, -3, (3 * p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3);
      store(along, -2, (2 * p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3);
      store(along, -1, (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3);
      store(along, 0, (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3);
      store(along, 1, (p1 + p0 + q0 + 2 * q1 + q2 + 2 * q3 + 4) >> 3);
      store(along, 2, (p0 + q0 + q1 + 2 * q2 + 3 * q3 + 4) >> 3);
    } else if (width === 6 && flatInner) {
      store(along, -2, (3 * p2 + 2 * p1 + 2 * p0 + q0 + 4) >> 3);
      store(along, -1, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
      store(along, 0, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
      store(along, 1, (p0 + 2 * q0 + 2 * q1 + 3 * q2 + 4) >> 3);
    } else {
      const highEdgeVariance = Math.abs(p1 - p0) > hevThreshold * scale ||
        Math.abs(q1 - q0) > hevThreshold * scale;
      const clipDifference = (value: number): number => clamp(value, -128 * scale, 128 * scale - 1);
      let filter = highEdgeVariance ? clipDifference(p1 - q1) : 0;
      filter = clipDifference(3 * (q0 - p0) + filter);
      const filter1 = Math.min(filter + 4, 128 * scale - 1) >> 3;
      const filter2 = Math.min(filter + 3, 128 * scale - 1) >> 3;
      store(along, -1, p0 + filter2);
      store(along, 0, q0 - filter1);
      if (!highEdgeVariance) {
        const half = (filter1 + 1) >> 1;
        store(along, -2, p1 + half);
        store(along, 1, q1 - half);
      }
    }
  }
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
