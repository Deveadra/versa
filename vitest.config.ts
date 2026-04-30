import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    coverage: {
      provider: 'v8',
      enabled: false,
      reporter: ['text-summary', 'json-summary', 'lcov'],
      reportsDirectory: './coverage/ws24-ts',
      thresholds: {
        'packages/integrations/src/index.ts': {
          lines: 95,
          statements: 95,
          functions: 100,
          branches: 83,
        },
        'packages/memory/src/index.ts': {
          lines: 70,
          statements: 70,
          functions: 63,
          branches: 38,
        },
        'packages/workspaces/src/index.ts': {
          lines: 70,
          statements: 70,
          functions: 60,
          branches: 35,
        },
        'packages/approvals/src/index.ts': {
          lines: 84,
          statements: 84,
          functions: 80,
          branches: 86,
        },
        'packages/environment/src/index.ts': {
          lines: 91,
          statements: 91,
          functions: 80,
          branches: 55,
        },
        'apps/ai/src/server.ts': {
          lines: 87,
          statements: 87,
          functions: 25,
          branches: 56,
        },
      },
    },
  },
});
