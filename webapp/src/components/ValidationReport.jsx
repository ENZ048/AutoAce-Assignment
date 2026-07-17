export default function ValidationReport({ job, onStart, onDiscard, disabled }) {
  const hasManifest = !job.warnings.some((w) => w.includes('no CSV manifest found'))
  return (
    <section className="rounded-xl border border-gray-200 bg-white p-6">
      <h2 className="text-lg font-semibold">Validation report</h2>
      <p className="mt-1 text-sm">
        <strong className="text-ink">{job.total}</strong> audio file{job.total === 1 ? '' : 's'} ready ·
        manifest {hasManifest ? 'found' : 'not found'}
      </p>
      {job.warnings.length > 0 && (
        <ul className="mt-3 space-y-1">
          {job.warnings.map((w) => (
            <li key={w} className="rounded-lg bg-amber-100 px-3 py-2 font-mono text-xs text-amber-700">
              {w}
            </li>
          ))}
        </ul>
      )}
      <p className="mt-3 text-xs text-gray-400">
        Nothing has been processed yet. Review the report, then start the analysis.
      </p>
      <div className="mt-4 flex gap-3">
        <button onClick={onStart} disabled={disabled}
          className="rounded-lg bg-navy px-4 py-2 text-sm font-medium text-white disabled:opacity-60">
          Start processing
        </button>
        <button onClick={onDiscard} disabled={disabled}
          className="rounded-lg border border-gray-200 px-4 py-2 text-sm font-medium text-ink disabled:opacity-60">
          Discard batch
        </button>
      </div>
    </section>
  )
}
