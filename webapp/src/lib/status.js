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

// Sentinel the worker writes to current_file while loading models (once per
// batch, before file 1). Mirrors store.MODEL_LOADING in the backend.
export const MODEL_LOADING = '__loading_models__'

export const isPreloading = (job) =>
  job.status === 'running' && job.current_file === MODEL_LOADING && !job.done

export const isTerminal = (s) => ['completed', 'failed', 'interrupted'].includes(s)

export const shortId = (id) => (id || '').slice(0, 8)

export const fmtTime = (iso) => (iso ? new Date(iso).toLocaleString() : '—')

// status is job.status: while a job is only 'queued' (not yet started), no file has
// been picked up yet, so every row must read as pending — 'done' can lag behind a
// stale re-render of a job that hasn't actually started processing.
// 'done' counts every processed file including failures, so failedFiles (from the
// worker) is what distinguishes a failed row from a completed one mid-run.
export const buildQueueRows = (files, done, total, status, failedFiles = []) =>
  files.map((name, i) => ({
    name,
    state:
      status === 'queued'
        ? 'pending'
        : i < done
          ? failedFiles.includes(name)
            ? 'failed'
            : 'completed'
          : i === done && done < total
            ? 'analyzing'
            : 'pending',
  }))
