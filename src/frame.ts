/** Shared decoded picture structure (used by both HEVC and AV1 decoders). */
export const CHROMA_MONO = 0, CHROMA_420 = 1, CHROMA_422 = 2, CHROMA_444 = 3;
export type SampleArray = Uint8Array | Uint16Array;

export class Plane {
  data: SampleArray;
  stride: number;
  width: number;
  height: number;
  /**
   * @param pad extra columns (appended to the stride) and rows allocated
   *   past width/height. Codecs whose coding grid can overhang the visible
   *   picture (HEVC codes partial edge blocks) write the overhang into the
   *   padding instead of wrapping into the next row.
   */
  constructor(width: number, height: number, stride?: number, bitDepth = 8, pad = 0) {
    this.width = width; this.height = height;
    this.stride = (stride ?? width) + pad;
    this.data = bitDepth <= 8
      ? new Uint8Array(this.stride * (height + pad))
      : new Uint16Array(this.stride * (height + pad));
  }
}

export class DecodedFrame {
  planes: Plane[] = [];
  width: number;
  height: number;
  bitDepth: number;
  chromaBitDepth: number;
  chromaFormat: number;
  constructor(
    width: number, height: number, bitDepth: number, chromaFormat: number,
    chromaBitDepth = bitDepth, pad = 0,
  ) {
    this.width = width; this.height = height;
    this.bitDepth = bitDepth; this.chromaBitDepth = chromaBitDepth; this.chromaFormat = chromaFormat;
    this.planes.push(new Plane(width, height, undefined, bitDepth, pad));
    if (chromaFormat !== CHROMA_MONO) {
      const hs = chromaFormat === CHROMA_420 || chromaFormat === CHROMA_422 ? 1 : 0;
      const vs = chromaFormat === CHROMA_420 ? 1 : 0;
      const chromaWidth = (width + (1 << hs) - 1) >> hs;
      const chromaHeight = (height + (1 << vs) - 1) >> vs;
      this.planes.push(new Plane(chromaWidth, chromaHeight, undefined, chromaBitDepth, pad));
      this.planes.push(new Plane(chromaWidth, chromaHeight, undefined, chromaBitDepth, pad));
    }
  }
  get luma() { return this.planes[0]; }
  get cb() { return this.planes[1]; }
  get cr() { return this.planes[2]; }
}
