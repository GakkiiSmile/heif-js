import { rmSync } from 'node:fs';

// Resolve the generated directory relative to this script so the deletion
// target cannot expand with the caller's working directory or environment.
rmSync(new URL('../dist/', import.meta.url), { recursive: true, force: true });
