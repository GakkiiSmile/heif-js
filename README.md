# heif-js

纯 TypeScript 的静态 HEIC、HEIF 与 AVIF 解码器。输入 `ArrayBuffer` 或任意
TypedArray/DataView，直接得到包含 RGBA8 像素的 `DecodedImage`。不依赖 WASM、
原生模块、WebCodecs，也不借用浏览器内置的 HEIF/AVIF 解码能力。

## 使用

```ts
import { decode } from 'heif-js';

const response = await fetch('/photo.avif');
const binary = await response.arrayBuffer();
const { width, height, data } = decode(binary);
// data 是 width × height × 4 的 Uint8ClampedArray（straight-alpha sRGB RGBA8）。
```

批量解码相同尺寸图片时，可以复用调用方分配的 RGBA 缓冲区，避免每帧重新分配
输出数组。缓冲区长度必须精确等于 `width * height * 4`，且不能与编码输入重叠：

```ts
const output = new Uint8ClampedArray(width * height * 4);
const image = decode(binary, { output });
// image.data === output
```

`decode` 是同步函数，返回 `{ width, height, data }`。解码入口不创建或返回任何
浏览器专用图像对象；默认导出与具名导出指向同一个 `decode` 函数。

只需要识别格式或读取 HEIF 元数据时，可使用轻量入口，避免加载 AV1/HEVC
解码表：

```ts
import { detectFormat, HeifFile } from 'heif-js/detect';
```

`HeifFile.parse()` 只解析容器元数据；未使用 item 的 extent 会在首次读取
`item.data` 时再组装和校验，从而避免复制缩略图或备用表示。普通 `ArrayBuffer`
输入的单 extent 会保留为零复制 view，因此在读取 lazy `item.data` 前不要修改原输入；
`SharedArrayBuffer` 输入会先建立稳定快照。

已知输入只使用一种编码时，可通过 `heif-js/heic` 或
`heif-js/avif` 使用同一套同步 `decode` API，只加载对应 codec。同步 codec 专用
入口仍完整支持 grid、identity、overlay、辅助 alpha、变换和资源限制；
若交给了另一种 codec，会抛出 `DecodeError`，其 `code` 为 `UNSUPPORTED_CODEC`。

## 解码能力

- HEVC `hvc1` / `hev1` 静态图像
  - 8–16-bit 语法路径（回归语料覆盖 8/10/12-bit）
  - 单色、4:2:0、4:2:2、4:4:4 与独立 colour-plane 图片
  - 多 slice、tiles、WPP、PCM、transform skip、显式/隐式 RDPCM
  - scaling list、range extensions、cross-component prediction
  - deblocking、SAO 与 SPS conformance window
- AV1 `av01` 静态图像
  - 8/10/12-bit，单色、4:2:0、4:2:2、4:4:4
  - reduced/full sequence 和 frame header
  - IntraBC、palette、filter intra、方向预测与全部静态图像逆变换
  - 多 tile / tile group、segmentation、delta Q/LF、quantization matrix
  - super-resolution、deblocking、CDEF、loop restoration、film grain
- HEIF 容器与派生图像
  - `iloc` construction method 0/1/2、多 extent、`idat`、32-bit item id
  - `grid`、`iden`、`iovl`
  - `auxC` alpha、`prem` 预乘关系
  - `clap`、`irot`、`imir`
- 色彩输出
  - H.273 NCLX full/limited range 与矩阵系数 0–15 的适用路径
  - BT.709、Display-P3、BT.2020 等原色到 sRGB 的线性光转换
  - PQ/HLG 到 RGBA8 的 SDR 映射
  - 常见 matrix/TRC 型 `rICC` / `prof` ICC 配置
- 结构化 `DecodeError`，包含格式、配置、解码和资源限制错误码

## 资源限制

解析不可信图片时默认启用尺寸、像素数、item、extent、HEVC NAL 数量、引用深度和累计解码量限制。
处理超大图片时可以按需放宽：

```ts
const image = decode(binary, {
  maxDimension: 100_000,
  maxPixels: 120_000_000,
  maxTotalPixels: 240_000_000,
  maxTotalItemBytes: 768 * 1024 * 1024,
  maxNals: 100_000,
});
```

解码入口导出 `DecodeOptions` 类型；默认限制 `DEFAULT_DECODE_LIMITS` 可从
`heif-js/detect` 导入。

## 范围

本项目面向 HEIF/AVIF 的静态、单层图像 item。以下内容不在当前 API 范围内：

- AVIF/HEIF 动画或视频轨道；
- AV1 多空间层（`a1op` / `lsel` / `a1lx`）的渐进式输出；
- HEIF 中与本项目目标无关的 JPEG、JPEG 2000、VVC 等其他编码格式；
- 需要外部文件的数据引用。

图片 item 中的 HEVC 使用 I-slice，AV1 使用 key/intra frame；这是本项目静态解码器
与完整视频解码器的边界。

## 开发验证

```sh
npm install
npm run build
npm test
npm run test:fuzz
npm run bench
npm run bench:entries
```

回归语料包含 HEIC/AVIF 的多位深和色度格式、IntraBC、多 tile/group、super-res、
film grain、loop restoration、量化矩阵、full header、grid、overlay、alpha、奇数尺寸
裁剪与颜色配置。关键 AV1/HEVC 样本会与 dav1d/FFmpeg/libheif 参考输出进行原始平面
或像素质量校验；浏览器冒烟页还会验证返回的 RGBA8 像素可以正确绘制到画布。
