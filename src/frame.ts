/** Shared decoded picture structure (used by both HEVC and AV1 decoders). */
export const CHROMA_MONO = 0, CHROMA_420 = 1, CHROMA_422 = 2, CHROMA_444 = 3;

export class Plane {
  data: Uint16Array;
  stride: number;
  width: number;
  height: number;
  constructor(width: number, height: number, stride?: number) {
    this.width = width; this.height = height;
    this.stride = stride ?? width;
    this.data = new Uint16Array(this.stride * height);
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
    chromaBitDepth = bitDepth,
  ) {
    this.width = width; this.height = height;
    this.bitDepth = bitDepth; this.chromaBitDepth = chromaBitDepth; this.chromaFormat = chromaFormat;
    this.planes.push(new Plane(width, height));
    if (chromaFormat !== CHROMA_MONO) {
      const hs = chromaFormat === CHROMA_420 || chromaFormat === CHROMA_422 ? 1 : 0;
      const vs = chromaFormat === CHROMA_420 ? 1 : 0;
      const chromaWidth = (width + (1 << hs) - 1) >> hs;
      const chromaHeight = (height + (1 << vs) - 1) >> vs;
      this.planes.push(new Plane(chromaWidth, chromaHeight));
      this.planes.push(new Plane(chromaWidth, chromaHeight));
    }
  }
  get luma() { return this.planes[0]; }
  get cb() { return this.planes[1]; }
  get cr() { return this.planes[2]; }
}
