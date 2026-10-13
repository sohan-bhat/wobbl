// Old public routes keep working after the docs were split into short pages.
/** Public routes from before the docs were split up, and where each one lives now. */
export const MOVED: Record<string, { to: string[]; old: string }> = {
  'get-started': { to: [], old: 'docs/getting-started.md' },
  guides: { to: ['get-started'], old: 'docs' },
  'guides/strength': { to: ['guides/simulations/strength'], old: 'docs/strength.md' },
  'guides/heat': { to: ['guides/simulations/heat'], old: 'docs/heat.md' },
  'guides/circuits': { to: ['guides/simulations/circuits'], old: 'docs/circuits.md' },
  'guides/fix': { to: ['guides/fixing'], old: 'docs/fix.md' },
  'guides/agents': { to: ['guides/agents/claude-code'], old: 'docs/agents.md' },
  'guides/checks': { to: ['guides/results/checks'], old: 'docs/verification.md' },
  'guides/reference': { to: ['guides/cli'], old: 'docs/cli.md' },
  validation: { to: [], old: 'docs/validation.md' },
};

