import { describe, expect, it } from 'vitest'
import { MODEL_LOADING, STATUS_META, buildQueueRows, isActive, isPreloading, isTerminal, shortId } from './status'

const ALL = ['validating', 'awaiting_confirmation', 'queued', 'running',
  'completed', 'failed', 'interrupted']

describe('status helpers', () => {
  it('covers every job state with label and chip classes', () => {
    for (const s of ALL) {
      expect(STATUS_META[s].label).toBeTruthy()
      expect(STATUS_META[s].chip).toContain('bg-')
    }
  })
  it('isActive only for in-flight states', () => {
    expect(ALL.filter(isActive)).toEqual(['validating', 'queued', 'running'])
  })
  it('shortId takes 8 chars', () => {
    expect(shortId('abcdef0123456789')).toBe('abcdef01')
  })
  it('buildQueueRows marks completed/analyzing/pending', () => {
    const files = ['a.wav', 'b.wav', 'c.wav']
    expect(buildQueueRows(files, 0, 3, 'running').map((r) => r.state))
      .toEqual(['analyzing', 'pending', 'pending'])
    expect(buildQueueRows(files, 1, 3, 'running').map((r) => r.state))
      .toEqual(['completed', 'analyzing', 'pending'])
    expect(buildQueueRows(files, 3, 3, 'running').map((r) => r.state))
      .toEqual(['completed', 'completed', 'completed'])
  })
  it('buildQueueRows marks failed files distinctly from completed ones', () => {
    const files = ['a.wav', 'bad.wav', 'c.wav']
    expect(buildQueueRows(files, 2, 3, 'running', ['bad.wav']).map((r) => r.state))
      .toEqual(['completed', 'failed', 'analyzing'])
  })
  it('buildQueueRows marks every row pending while the job is still queued', () => {
    const files = ['a.wav', 'b.wav', 'c.wav']
    expect(buildQueueRows(files, 0, 3, 'queued').map((r) => r.state))
      .toEqual(['pending', 'pending', 'pending'])
  })
  it('isPreloading only while the worker reports the model-loading sentinel', () => {
    expect(isPreloading({ status: 'running', current_file: MODEL_LOADING, done: 0 })).toBe(true)
    expect(isPreloading({ status: 'running', current_file: 'a.wav', done: 1 })).toBe(false)
    expect(isPreloading({ status: 'running', current_file: null, done: 0 })).toBe(false)
    expect(isPreloading({ status: 'queued', current_file: MODEL_LOADING, done: 0 })).toBe(false)
  })
  it('isTerminal only for completed/failed/interrupted', () => {
    expect(ALL.filter(isTerminal)).toEqual(['completed', 'failed', 'interrupted'])
  })
})
