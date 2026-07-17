import { buildQueueRows } from '../lib/status'

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
  const rows = buildQueueRows(job.files, job.done, job.total, job.status, job.failed_files)
  return (
    <section>
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
              : r.state === 'completed' ? 'text-gray-400 line-through decoration-green-700/40'
              : r.state === 'failed' ? 'text-red-600'
              : 'text-body'}`}>
            <span>{r.name}</span>
            <span>{r.state === 'completed' ? '✓' : r.state === 'failed' ? '✗ failed' : r.state === 'analyzing' ? 'analyzing…' : 'queued'}</span>
          </li>
        ))}
      </ul>
      {job.status === 'queued' && (
        <p className="mt-2 text-xs">Waiting for the current batch to finish — one batch runs at a time.</p>
      )}
      <p className="mt-2 text-xs text-gray-400">
        Safe to close or reload this page — progress is saved on the server.
      </p>
    </section>
  )
}
