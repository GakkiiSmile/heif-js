/** MSB-first AV1 uncompressed-header bit reader. */
export class Av1BitReader {
  readonly data: Uint8Array;
  bitPosition = 0;

  constructor(data: Uint8Array) {
    this.data = data;
  }

  get bitsRemaining(): number { return this.data.length * 8 - this.bitPosition; }

  bit(): number { return this.bits(1); }

  bits(count: number): number {
    if (count < 0 || count > 32 || this.bitPosition + count > this.data.length * 8) {
      throw new Error('AV1: truncated uncompressed header');
    }
    let value = 0;
    for (let i = 0; i < count; i++) {
      const position = this.bitPosition++;
      value = value * 2 + ((this.data[position >> 3]! >> (7 - (position & 7))) & 1);
    }
    return value >>> 0;
  }

  signedBits(count: number): number {
    const value = this.bits(count);
    const sign = 2 ** (count - 1);
    return value >= sign ? value - 2 ** count : value;
  }

  uleb128(): number {
    let value = 0;
    let shift = 0;
    for (let i = 0; i < 8; i++) {
      const byte = this.bits(8);
      value += (byte & 0x7f) * 2 ** shift;
      if (!(byte & 0x80)) return value;
      shift += 7;
    }
    throw new Error('AV1: invalid LEB128 value');
  }

  uniform(max: number): number {
    if (max <= 1) return 0;
    const length = Math.floor(Math.log2(max - 1)) + 1;
    const m = 2 ** length - max;
    const value = this.bits(length - 1);
    return value < m ? value : value * 2 - m + this.bit();
  }

  uvlc(): number {
    if (this.bit()) return 0;
    let leadingZeroes = 1;
    while (!this.bit()) {
      if (++leadingZeroes >= 32) throw new Error('AV1: UVLC overflow');
    }
    return 2 ** leadingZeroes - 1 + this.bits(leadingZeroes);
  }

  byteAlign(): void {
    this.bitPosition = (this.bitPosition + 7) & ~7;
  }
}
