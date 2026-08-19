/** AV1 multi-symbol arithmetic decoder (inverse-CDF form used by dav1d). */
const CDF_COUNT = Symbol('cdfCount');
type AdaptiveCdf = number[] & { [CDF_COUNT]?: number };

export class MsacDecoder {
  private readonly data: Uint8Array;
  private position = 0;
  // Unsigned 64-bit arithmetic-decoder state as two uint32 words.  BigInt in
  // this path creates temporaries for every decoded bool/symbol; all operations
  // used by AV1 are comparisons/subtractions in the high 16 bits, short left
  // shifts, and byte refills, which map directly to this representation.
  private differenceHigh = 0;
  private differenceLow = 0;
  range = 0x8000;
  private count = -15;
  private readonly updateCdf: boolean;

  constructor(data: Uint8Array, disableCdfUpdate = false) {
    this.data = data;
    this.updateCdf = !disableCdfUpdate;
    this.refill();
  }

  get bytesRead(): number { return this.position; }

  boolEqui(): number {
    const range = this.range;
    let value = ((range >> 8) << 7) + 4;
    const thresholdHigh = (value << 16) >>> 0;
    const upper = this.differenceHigh >= thresholdHigh;
    let high = this.differenceHigh;
    if (upper) high = (high - thresholdHigh) >>> 0;
    if (upper) value += range - 2 * value;
    this.normalize(high, this.differenceLow, value);
    return upper ? 0 : 1;
  }

  bool(probabilityOne: number): number {
    const range = this.range;
    let value = ((range >> 8) * (probabilityOne >> 6) >> 1) + 4;
    const thresholdHigh = (value << 16) >>> 0;
    const upper = this.differenceHigh >= thresholdHigh;
    let high = this.differenceHigh;
    if (upper) high = (high - thresholdHigh) >>> 0;
    if (upper) value += range - 2 * value;
    this.normalize(high, this.differenceLow, value);
    return upper ? 0 : 1;
  }

  boolAdapt(cdf: number[]): number {
    const bit = this.bool(cdf[0]!);
    if (this.updateCdf) {
      const adaptive = cdf as AdaptiveCdf;
      const count = adaptive[CDF_COUNT] ?? 0;
      const rate = 4 + (count >> 4);
      if (bit) cdf[0] += (32768 - cdf[0]!) >> rate;
      else cdf[0] -= cdf[0]! >> rate;
      adaptive[CDF_COUNT] = count + +(count < 32);
    }
    return bit;
  }

  symbol(cdf: number[], symbolCountMinusOne = cdf.length): number {
    const code = this.differenceHigh >>> 16;
    const scaledRange = this.range >> 8;
    let upper = this.range;
    let lower = this.range;
    let symbol = -1;
    do {
      symbol++;
      upper = lower;
      lower = (scaledRange * (cdf[symbol]! >> 6) >> 1) + 4 * (symbolCountMinusOne - symbol);
    } while (code < lower);
    this.normalize((this.differenceHigh - ((lower << 16) >>> 0)) >>> 0,
      this.differenceLow, upper - lower);

    if (this.updateCdf) {
      const adaptive = cdf as AdaptiveCdf;
      const count = adaptive[CDF_COUNT] ?? 0;
      const rate = 4 + (count >> 4) + +(symbolCountMinusOne > 2);
      let i = 0;
      for (; i < symbol; i++) cdf[i] += (32768 - cdf[i]!) >> rate;
      for (; i < symbolCountMinusOne; i++) cdf[i] -= cdf[i]! >> rate;
      adaptive[CDF_COUNT] = count + +(count < 32);
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

  private normalize(high: number, low: number, range: number): void {
    const shift = Math.clz32(range) - 16;
    const oldCount = this.count;
    if (shift) {
      this.differenceHigh = ((high << shift) | (low >>> (32 - shift))) >>> 0;
      this.differenceLow = (low << shift) >>> 0;
    } else {
      this.differenceHigh = high;
      this.differenceLow = low;
    }
    this.range = range << shift;
    this.count = oldCount - shift;
    // dav1d uses an unsigned comparison here: once the finite tile payload is
    // exhausted, a negative count must not trigger another virtual refill.
    if (oldCount >= 0 && oldCount < shift) this.refill();
  }

  private refill(): void {
    let shift = 40 - this.count;
    let high = this.differenceHigh, low = this.differenceLow;
    do {
      if (this.position >= this.data.length) {
        // Arithmetic decoders conventionally read virtual one bits past EOB.
        const bits = Math.min(64, shift + 8);
        if (bits >= 64) {
          high = 0xffff_ffff;
          low = 0xffff_ffff;
        } else if (bits > 32) {
          low = 0xffff_ffff;
          high = (high | (2 ** (bits - 32) - 1)) >>> 0;
        } else if (bits === 32) {
          low = 0xffff_ffff;
        } else if (bits > 0) {
          low = (low | (2 ** bits - 1)) >>> 0;
        }
        break;
      }
      const byte = this.data[this.position++]! ^ 0xff;
      if (shift >= 32) {
        high = (high | (byte << (shift - 32))) >>> 0;
      } else if (shift <= 24) {
        low = (low | (byte << shift)) >>> 0;
      } else {
        low = (low | (byte << shift)) >>> 0;
        high = (high | (byte >>> (32 - shift))) >>> 0;
      }
      shift -= 8;
    } while (shift >= 0);
    this.differenceHigh = high;
    this.differenceLow = low;
    this.count = 40 - shift;
  }
}

function inverseRecenter(reference: number, value: number): number {
  if (value > reference * 2) return value;
  return value & 1 ? reference - ((value + 1) >> 1) : reference + (value >> 1);
}

export function mutableCdf<T>(value: T): T {
  return cloneCdf(value) as T;
}

function cloneCdf(value: unknown): unknown {
  if (Array.isArray(value)) {
    const output = new Array(value.length);
    for (let index = 0; index < value.length; index++) output[index] = cloneCdf(value[index]);
    return output;
  }
  if (value !== null && typeof value === 'object') {
    const output: Record<string, unknown> = {};
    for (const key of Object.keys(value)) output[key] = cloneCdf((value as Record<string, unknown>)[key]);
    return output;
  }
  return value;
}
