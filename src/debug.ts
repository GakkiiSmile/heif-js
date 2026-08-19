/** Optional Node-side diagnostics without introducing a Node runtime dependency. */
interface DebugProcess {
  env?: Record<string, string | undefined>;
  stderr?: { write(text: string): unknown };
}

const debugProcess = (globalThis as typeof globalThis & { process?: DebugProcess }).process;

export function debugEnabled(name: string): boolean {
  return Boolean(debugProcess?.env?.[name]);
}

export function debugValue(name: string): string | undefined {
  return debugProcess?.env?.[name];
}

export function debugWrite(text: string): void {
  debugProcess?.stderr?.write(text);
}
