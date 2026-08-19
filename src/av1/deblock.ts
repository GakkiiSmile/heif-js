import type { DecodedFrame, Plane } from '../frame.ts';
import type { Av1DecodedBlock } from './decode.ts';
import type { Av1FrameHeader, Av1SequenceHeader } from './obu.ts';
import { block_dimensions } from './tables_data.ts';
import { transformSizes } from './tables.ts';

interface FilterMap {
  width4: number;
  height4: number;
  /** block id + transform dimensions + internal-edge flags, packed per 4x4 cell. */
  info: Uint32Array;
  /** six-bit vertical and horizontal filter levels. */
  levels: Uint16Array;
}

const BLOCK_ID_MASK = 0x00ff_ffff;
const TX_WIDTH_SHIFT = 24, TX_HEIGHT_SHIFT = 26;
const TX_SIZE_MASK = (3 << TX_WIDTH_SHIFT) | (3 << TX_HEIGHT_SHIFT);
const VERTICAL_TX = 1 << 28, HORIZONTAL_TX = 1 << 29;
const LEVEL_MASK = 0x3f, HORIZONTAL_LEVEL_SHIFT = 6;

/** Apply AV1's in-loop deblocking filter for key/intra still frames. */
export function applyDeblock(
  frame: DecodedFrame, blocks: readonly Av1DecodedBlock[],
  sequence: Av1SequenceHeader, header: Av1FrameHeader,
): void {
  if (!header.loopFilterLevels.some(Boolean)) return;
  const mapSizes = frame.planes.map(plane => Math.ceil(plane.width / 4) * Math.ceil(plane.height / 4));
  const totalMapSize = mapSizes.reduce((sum, size) => sum + size, 0);
  // One packed backing store replaces seven separate buffers per plane while
  // remaining short-lived after this frame has been filtered.
  const mapStorage = new ArrayBuffer(totalMapSize * 6);
  const packedInfo = new Uint32Array(mapStorage, 0, totalMapSize);
  const packedLevels = new Uint16Array(mapStorage, totalMapSize * 4, totalMapSize);
  let mapOffset = 0;
  const maps = frame.planes.map((plane, index) => {
    const map = createMap(plane, packedInfo, packedLevels, mapOffset, mapSizes[index]!);
    mapOffset += mapSizes[index]!;
    return map;
  });
  const ssX = sequence.subsamplingX, ssY = sequence.subsamplingY;

  blocks.forEach((block, blockId) => {
    const dimensions = block_dimensions[block.blockSize]!;
    fillBlockMap(maps[0]!, block, blockId, block.x4, block.y4,
      Math.min(dimensions[0]!, maps[0]!.width4 - block.x4),
      Math.min(dimensions[1]!, maps[0]!.height4 - block.y4), false, 0, header);

    if (frame.planes.length === 1) return;
    const hasChroma = (dimensions[0]! > ssX || !!(block.x4 & ssX)) &&
      (dimensions[1]! > ssY || !!(block.y4 & ssY));
    if (!hasChroma) return;
    const x4 = block.x4 >> ssX, y4 = block.y4 >> ssY;
    const width4 = Math.min((dimensions[0]! + ssX) >> ssX, maps[1]!.width4 - x4);
    const height4 = Math.min((dimensions[1]! + ssY) >> ssY, maps[1]!.height4 - y4);
    fillBlockMap(maps[1]!, block, blockId, x4, y4, width4, height4, true, 2, header);
    fillBlockMap(maps[2]!, block, blockId, x4, y4, width4, height4, true, 3, header);
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
        const currentInfo = map.info[index]!, previousInfo = map.info[previous]!;
        const blockEdge = (currentInfo & BLOCK_ID_MASK) !== (previousInfo & BLOCK_ID_MASK);
        if (!blockEdge && !(currentInfo & VERTICAL_TX)) continue;
        const category = Math.min(
          (currentInfo >> TX_WIDTH_SHIFT) & 3, (previousInfo >> TX_WIDTH_SHIFT) & 3, chroma ? 1 : 2,
        );
        const width = chroma ? 4 + 2 * category : 4 << category;
        const level = (map.levels[index]! & LEVEL_MASK) || (map.levels[previous]! & LEVEL_MASK);
        if (level) filterEdge(plane, x4 * 4, y4 * 4, true, width, level,
          header.loopFilterSharpness, sequence.bitDepth);
      }
    }
    for (let y4 = 1; y4 < map.height4; y4++) {
      for (let x4 = 0; x4 < map.width4; x4++) {
        const index = y4 * map.width4 + x4;
        const previous = index - map.width4;
        const currentInfo = map.info[index]!, previousInfo = map.info[previous]!;
        const blockEdge = (currentInfo & BLOCK_ID_MASK) !== (previousInfo & BLOCK_ID_MASK);
        if (!blockEdge && !(currentInfo & HORIZONTAL_TX)) continue;
        const category = Math.min(
          (currentInfo >> TX_HEIGHT_SHIFT) & 3, (previousInfo >> TX_HEIGHT_SHIFT) & 3, chroma ? 1 : 2,
        );
        const width = chroma ? 4 + 2 * category : 4 << category;
        const level = (map.levels[index]! >> HORIZONTAL_LEVEL_SHIFT) ||
          (map.levels[previous]! >> HORIZONTAL_LEVEL_SHIFT);
        if (level) filterEdge(plane, x4 * 4, y4 * 4, false, width, level,
          header.loopFilterSharpness, sequence.bitDepth);
      }
    }
  }
}

function createMap(
  plane: Plane, packedInfo: Uint32Array, packedLevels: Uint16Array, offset: number, size: number,
): FilterMap {
  const width4 = Math.ceil(plane.width / 4), height4 = Math.ceil(plane.height / 4);
  return {
    width4, height4,
    info: packedInfo.subarray(offset, offset + size),
    levels: packedLevels.subarray(offset, offset + size),
  };
}

function fillBlockMap(
  map: FilterMap, block: Av1DecodedBlock, blockId: number,
  x4: number, y4: number, width4: number, height4: number,
  chroma: boolean, levelSlot: number, header: Av1FrameHeader,
): void {
  if (width4 <= 0 || height4 <= 0) return;
  const verticalLevel = filterLevel(block, chroma ? levelSlot : 0, header);
  const horizontalLevel = filterLevel(block, chroma ? levelSlot : 1, header);
  const tx = chroma ? block.uvTx : block.tx;
  const txInfo = transformSizes[tx]!;
  const txWidth = Math.min(txInfo.logW, chroma ? 1 : 2);
  const txHeight = Math.min(txInfo.logH, chroma ? 1 : 2);
  const info = (blockId + 1) | (txWidth << TX_WIDTH_SHIFT) | (txHeight << TX_HEIGHT_SHIFT);
  const levels = verticalLevel | (horizontalLevel << HORIZONTAL_LEVEL_SHIFT);
  for (let y = 0; y < height4; y++) {
    for (let x = 0; x < width4; x++) {
      const index = (y4 + y) * map.width4 + x4 + x;
      map.info[index] = info;
      map.levels[index] = levels;
    }
  }

  if (!chroma && block.intrabc && block.yCoefficients.length) {
    for (const unit of block.yCoefficients) {
      const info = transformSizes[unit.tx]!;
      const ux = unit.x4, uy = unit.y4;
      for (let y = uy; y < Math.min(map.height4, uy + info.h4); y++) {
        for (let x = ux; x < Math.min(map.width4, ux + info.w4); x++) {
          const index = y * map.width4 + x;
          map.info[index] = (map.info[index]! & ~TX_SIZE_MASK) |
            (Math.min(info.logW, 2) << TX_WIDTH_SHIFT) |
            (Math.min(info.logH, 2) << TX_HEIGHT_SHIFT);
        }
      }
      if (!block.skip) {
        if (ux > x4) for (let y = uy; y < Math.min(map.height4, uy + info.h4); y++) {
          map.info[y * map.width4 + ux] |= VERTICAL_TX;
        }
        if (uy > y4) for (let x = ux; x < Math.min(map.width4, ux + info.w4); x++) {
          map.info[uy * map.width4 + x] |= HORIZONTAL_TX;
        }
      }
    }
  } else if (!block.intrabc || !block.skip) {
    for (let x = txInfo.w4; x < width4; x += txInfo.w4) {
      for (let y = 0; y < height4; y++) map.info[(y4 + y) * map.width4 + x4 + x] |= VERTICAL_TX;
    }
    for (let y = txInfo.h4; y < height4; y += txInfo.h4) {
      for (let x = 0; x < width4; x++) map.info[(y4 + y) * map.width4 + x4 + x] |= HORIZONTAL_TX;
    }
  }
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
  const data = plane.data;
  const alongStep = vertical ? plane.stride : 1;
  const acrossStep = vertical ? 1 : plane.stride;
  const base = y0 * plane.stride + x0;
  const differenceMinimum = -128 * scale, differenceMaximum = 128 * scale - 1;

  const count = Math.min(4, vertical ? plane.height - y0 : plane.width - x0);
  for (let along = 0; along < count; along++) {
    const center = base + along * alongStep;
    const p1 = data[center - 2 * acrossStep]!, p0 = data[center - acrossStep]!;
    const q0 = data[center]!, q1 = data[center + acrossStep]!;
    let filterMask = Math.abs(p1 - p0) <= interiorLimit * scale &&
      Math.abs(q1 - q0) <= interiorLimit * scale &&
      Math.abs(p0 - q0) * 2 + (Math.abs(p1 - q1) >> 1) <= edgeLimit * scale;
    let p2 = 0, q2 = 0, p3 = 0, q3 = 0;
    if (width > 4) {
      p2 = data[center - 3 * acrossStep]!; q2 = data[center + 2 * acrossStep]!;
      filterMask &&= Math.abs(p2 - p1) <= interiorLimit * scale &&
        Math.abs(q2 - q1) <= interiorLimit * scale;
      if (width > 6) {
        p3 = data[center - 4 * acrossStep]!; q3 = data[center + 3 * acrossStep]!;
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
      p4 = data[center - 5 * acrossStep]!; p5 = data[center - 6 * acrossStep]!;
      p6 = data[center - 7 * acrossStep]!;
      q4 = data[center + 4 * acrossStep]!; q5 = data[center + 5 * acrossStep]!;
      q6 = data[center + 6 * acrossStep]!;
      flatOuter = Math.abs(p6 - p0) <= scale && Math.abs(p5 - p0) <= scale &&
        Math.abs(p4 - p0) <= scale && Math.abs(q4 - q0) <= scale &&
        Math.abs(q5 - q0) <= scale && Math.abs(q6 - q0) <= scale;
    }

    if (width >= 16 && flatOuter && flatInner) {
      data[center - 6 * acrossStep] = clamp((7 * p6 + 2 * p5 + 2 * p4 + p3 + p2 + p1 + p0 + q0 + 8) >> 4, 0, maximum);
      data[center - 5 * acrossStep] = clamp((5 * p6 + 2 * p5 + 2 * p4 + 2 * p3 + p2 + p1 + p0 + q0 + q1 + 8) >> 4, 0, maximum);
      data[center - 4 * acrossStep] = clamp((4 * p6 + p5 + 2 * p4 + 2 * p3 + 2 * p2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4, 0, maximum);
      data[center - 3 * acrossStep] = clamp((3 * p6 + p5 + p4 + 2 * p3 + 2 * p2 + 2 * p1 + p0 + q0 + q1 + q2 + q3 + 8) >> 4, 0, maximum);
      data[center - 2 * acrossStep] = clamp((2 * p6 + p5 + p4 + p3 + 2 * p2 + 2 * p1 + 2 * p0 + q0 + q1 + q2 + q3 + q4 + 8) >> 4, 0, maximum);
      data[center - acrossStep] = clamp((p6 + p5 + p4 + p3 + p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + q2 + q3 + q4 + q5 + 8) >> 4, 0, maximum);
      data[center] = clamp((p5 + p4 + p3 + p2 + p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + q3 + q4 + q5 + q6 + 8) >> 4, 0, maximum);
      data[center + acrossStep] = clamp((p4 + p3 + p2 + p1 + p0 + 2 * q0 + 2 * q1 + 2 * q2 + q3 + q4 + q5 + 2 * q6 + 8) >> 4, 0, maximum);
      data[center + 2 * acrossStep] = clamp((p3 + p2 + p1 + p0 + q0 + 2 * q1 + 2 * q2 + 2 * q3 + q4 + q5 + 3 * q6 + 8) >> 4, 0, maximum);
      data[center + 3 * acrossStep] = clamp((p2 + p1 + p0 + q0 + q1 + 2 * q2 + 2 * q3 + 2 * q4 + q5 + 4 * q6 + 8) >> 4, 0, maximum);
      data[center + 4 * acrossStep] = clamp((p1 + p0 + q0 + q1 + q2 + 2 * q3 + 2 * q4 + 2 * q5 + 5 * q6 + 8) >> 4, 0, maximum);
      data[center + 5 * acrossStep] = clamp((p0 + q0 + q1 + q2 + q3 + 2 * q4 + 2 * q5 + 7 * q6 + 8) >> 4, 0, maximum);
    } else if (width >= 8 && flatInner) {
      data[center - 3 * acrossStep] = clamp((3 * p3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3, 0, maximum);
      data[center - 2 * acrossStep] = clamp((2 * p3 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3, 0, maximum);
      data[center - acrossStep] = clamp((p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3, 0, maximum);
      data[center] = clamp((p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3, 0, maximum);
      data[center + acrossStep] = clamp((p1 + p0 + q0 + 2 * q1 + q2 + 2 * q3 + 4) >> 3, 0, maximum);
      data[center + 2 * acrossStep] = clamp((p0 + q0 + q1 + 2 * q2 + 3 * q3 + 4) >> 3, 0, maximum);
    } else if (width === 6 && flatInner) {
      data[center - 2 * acrossStep] = clamp((3 * p2 + 2 * p1 + 2 * p0 + q0 + 4) >> 3, 0, maximum);
      data[center - acrossStep] = clamp((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3, 0, maximum);
      data[center] = clamp((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3, 0, maximum);
      data[center + acrossStep] = clamp((p0 + 2 * q0 + 2 * q1 + 3 * q2 + 4) >> 3, 0, maximum);
    } else {
      const highEdgeVariance = Math.abs(p1 - p0) > hevThreshold * scale ||
        Math.abs(q1 - q0) > hevThreshold * scale;
      let filter = highEdgeVariance ? clamp(p1 - q1, differenceMinimum, differenceMaximum) : 0;
      filter = clamp(3 * (q0 - p0) + filter, differenceMinimum, differenceMaximum);
      const filter1 = Math.min(filter + 4, differenceMaximum) >> 3;
      const filter2 = Math.min(filter + 3, differenceMaximum) >> 3;
      data[center - acrossStep] = clamp(p0 + filter2, 0, maximum);
      data[center] = clamp(q0 - filter1, 0, maximum);
      if (!highEdgeVariance) {
        const half = (filter1 + 1) >> 1;
        data[center - 2 * acrossStep] = clamp(p1 + half, 0, maximum);
        data[center + acrossStep] = clamp(q1 - half, 0, maximum);
      }
    }
  }
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
