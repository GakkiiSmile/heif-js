import { defineConfig } from '@rspack/cli';
import { resolve } from 'node:path';

const projectRoot = import.meta.dirname;
const sourceRoot = resolve(projectRoot, 'src');
const outputRoot = resolve(projectRoot, 'dist');
const publicEntries = {
  index: './src/index.ts',
  detect: './src/detect.ts',
  heic: './src/heic.ts',
  avif: './src/avif.ts',
};

function baseConfig(name) {
  return {
    name,
    context: projectRoot,
    mode: 'production',
    devtool: false,
    cache: false,
    performance: { hints: false },
    resolve: { extensions: ['.ts', '.js'] },
    module: {
      rules: [
        {
          test: /\.ts$/,
          exclude: /node_modules/,
          type: 'javascript/auto',
          use: {
            loader: 'builtin:swc-loader',
            options: {
              jsc: {
                parser: { syntax: 'typescript' },
                target: 'es2015',
              },
            },
          },
        },
      ],
    },
  };
}

const esm = {
  ...baseConfig('esm'),
  target: ['web', 'es2015'],
  entry: publicEntries,
  output: {
    path: resolve(outputRoot, 'esm'),
    filename: '[name].js',
    library: {
      type: 'modern-module',
      preserveModules: sourceRoot,
    },
  },
  optimization: {
    minimize: false,
  },
};

const cjs = {
  ...baseConfig('cjs'),
  target: ['node16', 'es2015'],
  entry: publicEntries,
  output: {
    path: resolve(outputRoot, 'cjs'),
    filename: '[name].cjs',
    library: { type: 'commonjs-static' },
  },
  optimization: {
    minimize: false,
    runtimeChunk: false,
    splitChunks: false,
  },
};

const amd = {
  ...baseConfig('amd'),
  target: ['web', 'es2015'],
  entry: { index: './src/index.ts' },
  output: {
    path: resolve(outputRoot, 'browser'),
    filename: 'heif-js.amd.js',
    library: {
      name: 'heif-js',
      type: 'amd',
    },
  },
  optimization: {
    minimize: true,
    runtimeChunk: false,
    splitChunks: false,
  },
};

const umd = {
  ...baseConfig('umd'),
  target: ['web', 'es2015'],
  entry: { index: './src/index.ts' },
  output: {
    path: resolve(outputRoot, 'browser'),
    filename: 'heif-js.umd.js',
    globalObject: 'typeof self !== "undefined" ? self : this',
    library: {
      name: {
        root: 'HeifJS',
        amd: 'heif-js',
        commonjs: 'heif-js',
      },
      type: 'umd',
      umdNamedDefine: true,
    },
  },
  optimization: {
    minimize: true,
    runtimeChunk: false,
    splitChunks: false,
  },
};

export default defineConfig([esm, cjs, amd, umd]);
