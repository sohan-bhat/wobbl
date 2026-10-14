// Image sizes, read from the files in public/, so every <img> gets its width and height.
import { readFileSync, readdirSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';
import { url } from './site';

const PUBLIC = resolve(process.cwd(), 'public');

/** Width and height of a WebP or PNG file, from its header. */
export function imageSize(file: string): [number, number] {
  const b = readFileSync(file).subarray(0, 32);
  if (b.toString('latin1', 0, 4) === 'RIFF' && b.toString('latin1', 8, 12) === 'WEBP') {
    const kind = b.toString('latin1', 12, 16);
    if (kind === 'VP8X') return [1 + b.readUIntLE(24, 3), 1 + b.readUIntLE(27, 3)];
    if (kind === 'VP8 ') return [b.readUInt16LE(26) & 0x3fff, b.readUInt16LE(28) & 0x3fff];
    if (kind === 'VP8L') {
      const bits = b.readUInt32LE(21);
      return [(bits & 0x3fff) + 1, ((bits >> 14) & 0x3fff) + 1];
    }
  }
  if (b.readUInt32BE(0) === 0x89504e47) return [b.readUInt32BE(16), b.readUInt32BE(20)];
  throw new Error(`unknown image format: ${file}`);
}

/** Size of a file under public/, given its site path such as /img/x.webp. */
export function publicSize(path: string): [number, number] | null {
  const file = resolve(PUBLIC, path.replace(/^\//, ''));
  return existsSync(file) ? imageSize(file) : null;
}

export interface Pic { src: string; srcset: string; width: number; height: number }

/** public/img/<name>.webp with its smaller copies (name-800.webp and so on) as a srcset. */
export function picture(name: string): Pic {
  const dir = resolve(PUBLIC, 'img');
  const [width, height] = imageSize(resolve(dir, `${name}.webp`));
  const small = readdirSync(dir).filter((f) => new RegExp(`^${name}-\\d+\\.webp$`).test(f));
  const parts = small.map((f) => `${url(`img/${f}`)} ${imageSize(resolve(dir, f))[0]}w`);
  parts.push(`${url(`img/${name}.webp`)} ${width}w`);
  return { src: url(`img/${name}.webp`), srcset: parts.join(', '), width, height };
}

/**
 * public/img/<name>.svg as markup to inline, so its labels use the site's fonts and the page can
 * play it. Every id gets the prefix ns, so two drawings never clash on one page.
 */
export function inlineSvg(name: string, ns: string): string {
  return readFileSync(resolve(PUBLIC, 'img', `${name}.svg`), 'utf-8').trim()
    .replace(/\bid="([^"]+)"/g, `id="${ns}-$1"`)
    .replace(/url\(#([^)]+)\)/g, `url(#${ns}-$1)`)
    .replace(/href="#([^"]+)"/g, `href="#${ns}-$1"`)
    .replace(/<svg /, '<svg focusable="false" ');
}
