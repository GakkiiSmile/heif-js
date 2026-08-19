/** AV1 multi-symbol arithmetic decoder (inverse-CDF form used by dav1d). */
export class MsacDecoder {
  private readonly data: Uint8Array;
  private position = 0;
  private difference = 0n;
  range = 0x8000;
  private count = -15;
  private readonly updateCdf: boolean;
  private readonly cdfCounts = new WeakMap<number[], number>();

  constructor(data: Uint8Array, disableCdfUpdate = false) {
    this.data = data;
    this.updateCdf = !disableCdfUpdate;
    this.refill();
  }

  get bytesRead(): number { return this.position; }

  boolEqui(): number {
    const range = this.range;
    let value = ((range >> 8) << 7) + 4;
    const threshold = BigInt(value) << 48n;
    const upper = this.difference >= threshold;
    let difference = this.difference;
    if (upper) difference -= threshold;
    if (upper) value += range - 2 * value;
    this.normalize(difference, value);
    return upper ? 0 : 1;
  }

  bool(probabilityOne: number): number {
    const range = this.range;
    let value = ((range >> 8) * (probabilityOne >> 6) >> 1) + 4;
    const threshold = BigInt(value) << 48n;
    const upper = this.difference >= threshold;
    let difference = this.difference;
    if (upper) difference -= threshold;
    if (upper) value += range - 2 * value;
    this.normalize(difference, value);
    return upper ? 0 : 1;
  }

  boolAdapt(cdf: number[]): number {
    const bit = this.bool(cdf[0]!);
    if (this.updateCdf) {
      const count = this.cdfCounts.get(cdf) ?? 0;
      const rate = 4 + (count >> 4);
      if (bit) cdf[0] += (32768 - cdf[0]!) >> rate;
      else cdf[0] -= cdf[0]! >> rate;
      this.cdfCounts.set(cdf, count + +(count < 32));
    }
    return bit;
  }

  symbol(cdf: number[], symbolCountMinusOne = cdf.length): number {
    const code = Number(this.difference >> 48n);
    const scaledRange = this.range >> 8;
    let upper = this.range;
    let lower = this.range;
    let symbol = -1;
    do {
      symbol++;
      upper = lower;
      lower = (scaledRange * (cdf[symbol]! >> 6) >> 1) + 4 * (symbolCountMinusOne - symbol);
    } while (code < lower);
    this.normalize(this.difference - (BigInt(lower) << 48n), upper - lower);

    if (this.updateCdf) {
      const count = this.cdfCounts.get(cdf) ?? 0;
      const rate = 4 + (count >> 4) + +(symbolCountMinusOne > 2);
      let i = 0;
      for (; i < symbol; i++) cdf[i] += (32768 - cdf[i]!) >> rate;
      for (; i < symbolCountMinusOne; i++) cdf[i] -= cdf[i]! >> rate;
      this.cdfCounts.set(cdf, count + +(count < 32));
    }
    return symbol;
  }

  bools(count: number): number {
    let value = 0;
    for (let i = 0; i < count; i++) value = value * 2 + this.boolEqui();
    return value;
  }

  uniform(max: number): number {
    if (max <= 1) return 0;
    const length = Math.floor(Math.log2(max - 1)) + 1;
    const m = 2 ** length - max;
    const value = this.bools(length - 1);
    return value < m ? value : value * 2 - m + this.boolEqui();
  }

  hiToken(cdf: number[]): number {
    let branch = this.symbol(cdf, 3);
    let token = 3 + branch;
    if (branch === 3) {
      branch = this.symbol(cdf, 3); token = 6 + branch;
      if (branch === 3) {
        branch = this.symbol(cdf, 3); token = 9 + branch;
        if (branch === 3) token = 12 + this.symbol(cdf, 3);
      }
    }
    return token;
  }

  subexp(reference: number, range: number, bits: number): number {
    let offset = 0;
    if (this.boolEqui()) {
      if (this.boolEqui()) bits += this.boolEqui() + 1;
      offset = 1 << bits;
    }
    const value = this.bools(bits) + offset;
    return reference * 2 <= range
      ? inverseRecenter(reference, value)
      : range - 1 - inverseRecenter(range - 1 - reference, value);
  }

  private normalize(difference: bigint, range: number): void {
    const shift = 15 - Math.floor(Math.log2(range));
    const oldCount = this.count;
    this.difference = BigInt.asUintN(64, difference << BigInt(shift));
    this.range = range << shift;
    this.count = oldCount - shift;
    // dav1d uses an unsigned comparison here: once the finite tile payload is
    // exhausted, a negative count must not trigger another virtual refill.
    if (oldCount >= 0 && oldCount < shift) this.refill();
  }

  private refill(): void {
    let shift = 40 - this.count;
    let difference = this.difference;
    do {
      if (this.position >= this.data.length) {
        // Arithmetic decoders conventionally read virtual one bits past EOB.
        const mask = (1n << BigInt(Math.min(64, shift + 8))) - 1n;
        difference |= mask;
        break;
      }
      difference |= BigInt(this.data[this.position++]! ^ 0xff) << BigInt(shift);
      shift -= 8;
    } while (shift >= 0);
    this.difference = BigInt.asUintN(64, difference);
    this.count = 40 - shift;
  }
}

function inverseRecenter(reference: number, value: number): number {
  if (value > reference * 2) return value;
  return value & 1 ? reference - ((value + 1) >> 1) : reference + (value >> 1);
}

export function mutableCdf<T>(value: T): T {
  return structuredClone(value);
}
