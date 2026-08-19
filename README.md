# heic-decode-pure-js

纯 TypeScript 的静态 HEIC、HEIF 与 AVIF 解码器。输入 `ArrayBuffer` 或任意
TypedArray/DataView，浏览器中直接得到 `ImageBitmap`。不依赖 WASM、原生模块、
WebCodecs，也不借用浏览器内置的 HEIF/AVIF 解码能力。

## 使用

```ts
import decode, { decodeToRgba } from 'heic-decode-pure-js';

const response = await fetch('/photo.avif');
const binary = await response.arrayBuffer();

const bitmap = await decode(binary);
canvas.getContext('2d')!.drawImage(bitmap, 0, 0);

// 非 DOM 环境或需要直接访问像素时：
const { width, height, data } = decodeToRgba(binary);
// data 是 width × height × 4 的 Uint8ClampedArray（直通 alpha 的 sRGB RGBA8）。
```

也可以使用 `decodeToImageData(binary)`。`decodeToRgba` 是同步函数；默认导出的
`decode` 调用浏览器 `createImageBitmap()`，因此返回 `Promise<ImageBitmap>`。

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
- 结构化 `DecodeError`，包含格式、配置、解码、运行环境和资源限制错误码

## 资源限制

解析不可信图片时默认启用尺寸、像素数、item、extent、引用深度和累计解码量限制。
处理超大图片时可以按需放宽：

```ts
const bitmap = await decode(binary, {
  maxDimension: 100_000,
  maxPixels: 120_000_000,
  maxTotalPixels: 240_000_000,
});
```

所有选项及默认值由 `DecodeOptions` 和 `DEFAULT_DECODE_LIMITS` 导出。

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
```

回归语料包含 HEIC/AVIF 的多位深和色度格式、IntraBC、多 tile/group、super-res、
film grain、loop restoration、量化矩阵、full header、grid、overlay、alpha、奇数尺寸
裁剪与颜色配置。关键 AV1/HEVC 样本会与 dav1d/FFmpeg/libheif 参考输出进行原始平面
或像素质量校验；浏览器冒烟页还会验证结果确实是可绘制的 `ImageBitmap`。
