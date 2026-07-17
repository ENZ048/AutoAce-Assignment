import { useEffect, useState } from 'react'
import { buildQueueRows, isActive, isPreloading } from '../lib/status'

function Stat({ value, label }) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 text-center">
      <div className="font-display text-3xl font-bold text-accent">{value}</div>
      <div className="mt-1 text-xs uppercase tracking-wide">{label}</div>
    </div>
  )
}

function elapsed(startedAt) {
  if (!startedAt) return '0:00'
  const s = Math.max(0, Math.floor((Date.now() - new Date(startedAt)) / 1000))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

export default function LiveQueue({ job }) {
  // Elapsed is derived from started_at at render time; without a local 1s
  // tick it only re-renders when a poll lands (every 2s) and jumps by 2.
  const [, setTick] = useState(0)
  const running = isActive(job.status)
  useEffect(() => {
    if (!running) return undefined
    const t = setInterval(() => setTick((n) => n + 1), 1000)
    return () => clearInterval(t)
  }, [running])
  const preloading = isPreloading(job)
  // While models load, no file is being analyzed yet — every row reads pending.
  const rows = buildQueueRows(job.files, job.done, job.total, preloading ? 'queued' : job.status, job.failed_files)
  return (
    <section>
      {preloading && (
        <p className="mb-3 rounded-lg bg-blue-100 px-3 py-2 text-xs text-blue-700">
          Loading analysis models… happens once per batch, before the first file.
        </p>
      )}
      <div className="grid grid-cols-3 gap-3">
        <Stat value={job.done} label="Processed" />
        <Stat value={job.total - job.done} label="Remaining" />
        <Stat value={elapsed(job.started_at)} label="Elapsed" />
      </div>
      <div className="mt-3 h-2 overflow-hidden rounded-full bg-gray-200">
        <div className="h-full rounded-full bg-accent transition-all"
          style={{ width: `${job.total ? (job.done / job.total) * 100 : 0}%` }} />
      </div>
      <ul className="mt-4 max-h-96 space-y-1 overflow-y-auto rounded-xl border border-gray-200 bg-white p-3">
        {rows.map((r) => (
          <li key={r.name}
            className={`flex items-center justify-between rounded-lg px-3 py-1.5 font-mono text-xs ${
              r.state === 'analyzing' ? 'queue-analyzing text-blue-700'
              : r.state === 'completed' ? 'text-gray-400'
              : r.state === 'failed' ? 'text-red-600'
              : 'text-body'}`}>
            {/* strike only the filename — never the status marker */}
            <span className={r.state === 'completed' ? 'line-through decoration-green-700/40' : ''}>{r.name}</span>
            {r.state === 'completed'
              ? <span className="text-lg font-bold leading-none text-green-600" aria-label="completed">✓</span>
              : <span>{r.state === 'failed' ? '✗ failed' : r.state === 'analyzing' ? 'analyzing…' : 'queued'}</span>}
          </li>
        ))}
      </ul>
      {job.status === 'queued' && (
        <p className="mt-2 text-xs">Waiting for the current batch to finish — one batch runs at a time.</p>
      )}
    </section>
  )
}
