import js from '@eslint/js';

export default [
  js.configs.recommended,
  {
    ignores: ['**/dist/**', '**/.next/**', '**/node_modules/**']
  }
];
