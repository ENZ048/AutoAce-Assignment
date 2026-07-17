import { expect, test } from '@playwright/test'

// Server must be running with DASHBOARD_STUB_ANALYZE=1 and the SPA built.
// Credentials come from env so real ones never land in the repo.
const USER = process.env.E2E_USER
const PASS = process.env.E2E_PASS

test('login → upload → confirm → live queue → results → download', async ({ page }) => {
  await page.goto('/login')
  await page.getByLabel('Username').fill(USER)
  await page.getByLabel('Password').fill(PASS)
  await page.getByRole('button', { name: 'Sign in' }).click()

  await page.getByRole('button', { name: 'Choose ZIP' }).waitFor()
  const chooser = page.waitForEvent('filechooser')
  await page.getByRole('button', { name: 'Choose ZIP' }).click()
  await (await chooser).setFiles('e2e/fixtures/batch.zip')

  await expect(page.getByText('Validation report')).toBeVisible()
  await expect(page.getByText('2 audio files ready')).toBeVisible()
  await page.getByRole('button', { name: 'Start processing' }).click()

  await expect(page.getByText('call_001.wav')).toBeVisible()
  await expect(page.getByText('Succeeded')).toBeVisible({ timeout: 90000 })

  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: '⬇ results.csv' }).click()
  expect((await download).suggestedFilename()).toBe('results.csv')
})
