export const STATUS_META = {
  validating: { label: 'Validating', chip: 'bg-blue-100 text-blue-700' },
  awaiting_confirmation: { label: 'Awaiting confirmation', chip: 'bg-amber-100 text-amber-700' },
  queued: { label: 'Queued', chip: 'bg-blue-100 text-blue-700' },
  running: { label: 'Analyzing', chip: 'bg-blue-100 text-blue-700' },
  completed: { label: 'Completed', chip: 'bg-green-100 text-green-700' },
  failed: { label: 'Failed', chip: 'bg-red-100 text-red-700' },
  interrupted: { label: 'Interrupted', chip: 'bg-red-100 text-red-700' },
}

export const isActive = (s) => ['validating', 'queued', 'running'].includes(s)

export const shortId = (id) => (id || '').slice(0, 8)

export const fmtTime = (iso) => (iso ? new Date(iso).toLocaleString() : '—')

export const buildQueueRows = (files, done, total) =>
  files.map((name, i) => ({
    name,
    state: i < done ? 'completed' : i === done && done < total ? 'analyzing' : 'pending',
  }))
