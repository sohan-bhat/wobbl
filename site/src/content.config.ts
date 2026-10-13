// The docs pages: short Markdown files under src/content/docs, one per page.
// A file's path is its route: docs/guides/inputs/loads.md is /guides/inputs/loads/.
// The validation page also gets the results table from docs/validation-results.json.
import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const docs = defineCollection({
  loader: glob({
    pattern: '**/*.md',
    base: './src/content/docs',
    generateId: ({ entry }) => entry.replace(/\.md$/, '').replace(/(^|\/)index$/, ''),
  }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    lead: z.string().optional(),
    wide: z.boolean().default(false),
    order: z.number().optional(),
    cards: z.array(z.object({ title: z.string(), href: z.string(), text: z.string() })).optional(),
  }),
});

export const collections = { docs };
