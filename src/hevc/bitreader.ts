/** MSB-first bit reader over an RBSP byte stream (H.265 syntax parsing). */
export class BitReader {
  pos = 0;        // bit position
  u8: Uint8Array;
  byteStart: number;
  byteEnd: number;
  constructor(u8: Uint8Array, byteStart = 0, byteEnd = u8.length) {
    this.u8 = u8; this.byteStart = byteStart; this.byteEnd = byteEnd;
    this.pos = byteStart * 8;
  }
  get bitsLeft() { return this.byteEnd * 8 - this.pos; }
  u(n: number): number {
    if (!Number.isSafeInteger(n) || n < 0 || this.pos + n > this.byteEnd * 8) {
      throw new Error('HEVC: truncated RBSP');
    }
    if (n > 32) {
      this.pos += n;
      return 0;
    }
    let v = 0;
    for (let i = 0; i < n; i++) {
      const bitPos = this.pos + i;
      const byteI = bitPos >> 3;
      const bit = (this.u8[byteI]! >> (7 - (bitPos & 7))) & 1;
      // Multiply instead of shifting so a full 32-bit read stays unsigned;
      // entry-point offsets are allowed to use the complete u(32) range.
      v = v * 2 + bit;
    }
    this.pos += n;
    return v;
  }
  u1(): number { return this.u(1); }
  /** unsigned Exp-Golomb */
  ue(): number {
    let leading = 0;
    while (this.u1() === 0) {
      if (++leading >= 32) throw new Error('HEVC: Exp-Golomb value overflow');
    }
    return 2 ** leading - 1 + (leading ? this.u(leading) : 0);
  }
  /** signed Exp-Golomb */
  se(): number {
    const k = this.ue();
    return k & 1 ? (k + 1) >> 1 : -(k >> 1);
  }
  byteAlign() { this.pos = (this.pos + 7) & ~7; }
  /** more_rbsp_data: any non-zero bit left before the trailing stop bit */
  moreRbspData(): boolean {
    let p = this.pos;
    // skip zero bits, find stop bit
    while (p < this.byteEnd * 8) {
      const bit = (this.u8[p >> 3] >> (7 - (p & 7))) & 1;
      if (bit === 1) {
        // stop bit must be last one of the byte and everything after zero
        return (p & 7) !== 7 || p + 8 < this.byteEnd * 8 && this.u8[(p >> 3) + 1] !== 0;
      }
      p++;
    }
    return false;
  }
}
