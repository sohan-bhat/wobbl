// The validation cases, read from docs/validation-results.json (written by `wobbl validate`), with
// the group and plain line of each from src/data/validation-notes.json. The docs pages quote the
// counts through {{cases}}, {{Cases}} and {{exact}}, filled in here, so they never fall behind
// when a domain adds cases, and the build fails if a page's claim about them stops being true.
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { ROOT } from './site';

export interface Case {
  id: string; title: string; domain: string; source: string; quantity: string; unit: string;
  expected: number; wobbl: number; error: number; tolerance: number; pass: boolean; exact_theory: boolean;
  confidence?: string | null;
}

export const cases: Case[] = JSON.parse(readFileSync(resolve(ROOT, 'docs/validation-results.json'), 'utf-8'));

const notesFile = resolve(process.cwd(), 'src/data/validation-notes.json');
export const notes: Record<string, { group?: string; plain?: string }> = existsSync(notesFile)
  ? JSON.parse(readFileSync(notesFile, 'utf-8')) : {};

/** Groups in table order. A domain without a note or a label here gets a group named after it. */
export const ORDER = ['Textbook formulas', 'Industry benchmarks', 'Real parts from the bake off', 'Heat', 'Circuits', 'Motion'];
const LABEL: Record<string, string> = { heat: 'Heat', circuit: 'Circuits', motion: 'Motion' };

export function groupOf(c: Case): string {
  if (notes[c.id]?.group) return notes[c.id].group!;
  if (c.domain !== 'strength') return LABEL[c.domain] ?? c.domain[0].toUpperCase() + c.domain.slice(1);
  return c.id.startsWith('nafems') ? 'Industry benchmarks' : c.exact_theory ? 'Textbook formulas' : 'Real parts from the bake off';
}

const ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
  'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
const TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];

/** A count in words, as the docs write them: nineteen, twenty five, a hundred and four. */
export function words(n: number): string {
  if (n < 20) return ONES[n];
  if (n < 100) return TENS[Math.floor(n / 10)] + (n % 10 ? ` ${ONES[n % 10]}` : '');
  if (n < 1000) return `${Math.floor(n / 100) === 1 ? 'a' : ONES[Math.floor(n / 100)]} hundred${n % 100 ? ` and ${words(n % 100)}` : ''}`;
  return String(n);
}

const exact = cases.filter((c) => c.exact_theory);

/**
 * Fill the counts into a docs page's text: {{cases}} is the number of cases, {{Cases}} the same in
 * words at the start of a sentence and {{exact}} the cases whose reference is exact, in words.
 */
export function fillCounts(text: string, where: string): string {
  if (!text.includes('{{')) return text;
  // the pages say every case passes and every exact case lands within 2%
  if (!cases.every((c) => c.pass)) throw new Error(`${where}: a validation case no longer passes its tolerance`);
  if (exact.some((c) => c.error > 0.02)) throw new Error(`${where}: an exact validation case is off by more than 2%`);
  const cap = (s: string) => s[0].toUpperCase() + s.slice(1);
  return text.replace(/\{\{(cases|Cases|exact)\}\}/g, (_m, k: string) =>
    k === 'cases' ? String(cases.length) : k === 'Cases' ? cap(words(cases.length)) : words(exact.length));
}
