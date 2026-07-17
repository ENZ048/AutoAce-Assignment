import { useEffect, useState } from 'react'
import toast from 'react-hot-toast'
import { downloadArtifact, getErrors, getResults } from '../api'

const ENUM_CHIP = {
  neutral: 'bg-gray-100 text-gray-700', satisfied: 'bg-green-100 text-green-700',
  frustrated: 'bg-amber-100 text-amber-700', upset: 'bg-red-100 text-red-700',
  distressed: 'bg-red-100 text-red-700', low: 'bg-gray-100 text-gray-700',
  medium: 'bg-amber-100 text-amber-700', high: 'bg-red-100 text-red-700',
  none: 'bg-gray-100 text-gray-700', clear: 'bg-green-100 text-green-700',
  slightly_impaired: 'bg-amber-100 text-amber-700', severely_impaired: 'bg-red-100 text-red-700',
}

function Chip({ value }) {
  const cls = ENUM_CHIP[value] ?? 'bg-gray-100 text-gray-700'
  return (
    <span className={`inline-block whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ${cls}`}>
      {String(value).replaceAll('_', ' ')}
    </span>
  )
}

const Muted = () => <span className="text-xs text-gray-400">—</span>

const Bool = ({ value }) =>
  value ? <span className="text-sm font-bold leading-none text-green-600">✓</span> : <Muted />

function ConfidenceCell({ value }) {
  return (
    <div className="flex items-center justify-end gap-2">
      <span className="h-1 w-12 overflow-hidden rounded-full bg-gray-100">
        <span className="block h-full rounded-full bg-accent" style={{ width: `${Math.round(value * 100)}%` }} />
      </span>
      <span className="font-mono text-xs text-ink">{value.toFixed(2)}</span>
    </div>
  )
}

// One column per schema field (the flat 9-field contract), styled for reading:
// chips for enums, centered green checks for booleans, a meter for confidence.
const COLUMNS = [
  { label: 'file', cell: (r) => <span className="whitespace-nowrap font-mono text-xs text-ink">{r.name}</span> },
  { label: 'tone', cell: (r) => <Chip value={r.emotional_tone} /> },
  { label: 'intensity', cell: (r) => <Chip value={r.emotional_intensity} /> },
  { label: 'noise', center: true, cell: (r) => <Bool value={r.background_noise_present} /> },
  {
    label: 'noise type',
    cell: (r) =>
      r.background_noise_type
        ? <span className="whitespace-nowrap font-mono text-xs text-ink">{r.background_noise_type}</span>
        : <Muted />,
  },
  { label: 'severity', cell: (r) => <Chip value={r.background_noise_severity} /> },
  { label: 'quality', cell: (r) => <Chip value={r.audio_quality} /> },
  { label: 'overlap', center: true, cell: (r) => <Bool value={r.speaker_overlap_present} /> },
  { label: 'long silence', center: true, cell: (r) => <Bool value={r.long_silence_present} /> },
  { label: 'confidence', right: true, cell: (r) => <ConfidenceCell value={r.confidence} /> },
]

export default function ResultsSection({ job }) {
  const [rows, setRows] = useState([])
  const [errors, setErrors] = useState([])
  const [filter, setFilter] = useState('')

  useEffect(() => {
    // Keyed on finished_at too: same job id can complete more than once (re-run),
    // and this section only mounts for completed jobs — a fetch failure here is a
    // real problem, not an expected in-progress 409, so it gets a toast, not silence.
    let alive = true
    getResults(job.id).then((r) => { if (alive) setRows(r) })
      .catch((e) => { if (alive) toast.error(e.response?.data?.detail ?? 'Could not load results') })
    getErrors(job.id).then((e) => { if (alive) setErrors(e) })
      .catch((e) => { if (alive) toast.error(e.response?.data?.detail ?? 'Could not load errors') })
    return () => { alive = false }
  }, [job.id, job.finished_at])

  const shown = rows.filter((r) => r.name.toLowerCase().includes(filter.toLowerCase()))

  return (
    <section className="space-y-6">
      <div className="grid grid-cols-3 gap-3 text-center">
        {[[job.results_count, 'Succeeded', 'text-green-700'],
          [job.errors_count, 'Failed', job.errors_count ? 'text-red-700' : 'text-gray-400'],
          [job.warnings.length, 'Warnings', job.warnings.length ? 'text-amber-700' : 'text-gray-400'],
        ].map(([v, label, tone]) => (
          <div key={label} className="rounded-xl border border-gray-200 bg-white p-4">
            <div className={`font-display text-3xl font-bold ${tone}`}>{v ?? 0}</div>
            <div className="mt-1 text-xs uppercase tracking-wide">{label}</div>
          </div>
        ))}
      </div>

      {job.warnings.length > 0 && (
        <ul className="space-y-1">
          {job.warnings.map((w) => (
            <li key={w} className="rounded-lg bg-amber-100 px-3 py-2 font-mono text-xs text-amber-700">{w}</li>
          ))}
        </ul>
      )}

      {errors.length > 0 && (
        <div className="overflow-x-auto rounded-xl border border-red-200 bg-white">
          <table className="w-full text-left text-sm">
            <thead className="bg-red-100 text-xs uppercase tracking-wide text-red-700">
              <tr><th className="px-4 py-2">Failed file</th><th className="px-4 py-2">Reason</th></tr>
            </thead>
            <tbody>
              {errors.map((e) => (
                <tr key={e.name} className="border-t border-red-100">
                  <td className="px-4 py-2 font-mono text-xs">{e.name}</td>
                  <td className="px-4 py-2 font-mono text-xs">{e.error}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="flex items-center justify-between gap-3">
        <input value={filter} onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter by filename…"
          className="w-56 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm" />
        <div className="flex gap-2">
          {['results.csv', 'results.json', 'errors.csv'].map((a) => (
            <button key={a} onClick={() => downloadArtifact(job.id, a)}
              className="rounded-lg border border-gray-200 bg-white px-3 py-1.5 font-mono text-xs text-ink">
              ⬇ {a}
            </button>
          ))}
        </div>
      </div>

      <div className="max-h-[32rem] overflow-auto rounded-xl border border-gray-200 bg-white">
        <table className="w-full text-left text-sm">
          <thead className="sticky top-0 z-10 bg-white/95 backdrop-blur">
            <tr className="border-b border-gray-200">
              {COLUMNS.map((c) => (
                <th key={c.label}
                  className="whitespace-nowrap px-4 py-3 text-center align-middle text-[11px] font-semibold uppercase tracking-[0.06em] text-gray-600">
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.name} className="border-t border-gray-100 transition-colors first:border-t-0 hover:bg-wash/60">
                {COLUMNS.map((c) => (
                  <td key={c.label} className={`px-4 py-3 ${c.right ? 'text-right' : c.center ? 'text-center' : ''}`}>{c.cell(r)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
