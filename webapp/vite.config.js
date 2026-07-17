import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'
import { configDefaults } from 'vitest/config'

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
  // Playwright's e2e/*.spec.js files use @playwright/test's test()/expect(), which
  // vitest's default include glob (**/*.spec.js) would otherwise also try to collect
  // and run as vitest tests, crashing with "Playwright Test did not expect test() to
  // be called here." Keep the two runners scoped to their own directories.
  test: { exclude: [...configDefaults.exclude, 'e2e/**'] },
})
