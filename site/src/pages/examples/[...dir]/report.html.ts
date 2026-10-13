// Each example's self contained report.html, copied from examples/ when the repository tracks it.
import type { APIRoute } from 'astro';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { examples } from '../../../lib/examples';
import { ROOT } from '../../../lib/site';

export function getStaticPaths() {
  return examples().flatMap((ex) => ex.runs.filter((r) => r.report)
    .map((r) => ({ params: { dir: r.report!.replace(/^examples\//, '').replace(/\/report\.html$/, '') }, props: { file: r.report! } })));
}

export const GET: APIRoute = ({ props }) =>
  new Response(readFileSync(resolve(ROOT, props.file as string)), { headers: { 'Content-Type': 'text/html; charset=utf-8' } });
