import { useCallback, useEffect, useState } from 'react'
import toast from 'react-hot-toast'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { deleteJob, getJob, rerunJob, startJob } from '../api'
import LiveQueue from '../components/LiveQueue'
import ResultsSection from '../components/ResultsSection'
import StatusChip from '../components/StatusChip'
import ValidationReport from '../components/ValidationReport'
import { isActive } from '../lib/status'

export default function JobPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [job, setJob] = useState(null)
  const [missing, setMissing] = useState(false)

  const refresh = useCallback(() =>
    getJob(id).then(setJob).catch((e) => e.response?.status === 404 && setMissing(true)), [id])

  useEffect(() => {
    refresh()
    const t = setInterval(() => {
      setJob((j) => { if (!j || isActive(j.status)) refresh(); return j })
    }, 2000)
    return () => clearInterval(t)
  }, [refresh])

  if (missing) return <Shell><p className="text-sm">This batch no longer exists.</p></Shell>
  if (!job) return <Shell><p className="text-sm">Loading…</p></Shell>

  const act = (fn, okMsg) => () =>
    fn(job.id).then(() => { okMsg && toast.success(okMsg); refresh() })
      .catch((e) => toast.error(e.response?.data?.detail ?? 'Action failed'))

  return (
    <Shell>
      <header className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">
            <span className="text-accent">Batch</span> {job.original_name}
          </h1>
          <p className="mt-1 font-mono text-xs text-gray-400">{job.id}</p>
        </div>
        <StatusChip status={job.status} />
      </header>

      {job.status === 'awaiting_confirmation' && (
        <ValidationReport job={job}
          onStart={act(startJob, 'Batch queued')}
          onDiscard={() => deleteJob(job.id).then(() => navigate('/'))} />
      )}
      {(job.status === 'queued' || job.status === 'running') && <LiveQueue job={job} />}
      {job.status === 'completed' && <ResultsSection job={job} />}
      {(job.status === 'failed' || job.status === 'interrupted') && (
        <section className="rounded-xl border border-red-200 bg-white p-6">
          <h2 className="text-lg font-semibold text-red-700">
            {job.status === 'failed' ? 'Batch failed' : 'Batch interrupted'}
          </h2>
          <p className="mt-2 rounded-lg bg-red-100 px-3 py-2 font-mono text-xs text-red-700">
            {job.error ?? 'No details recorded.'}
          </p>
          <button onClick={act(rerunJob, 'Batch re-queued')}
            className="mt-4 rounded-lg bg-navy px-4 py-2 text-sm font-medium text-white">
            Re-run batch
          </button>
        </section>
      )}
    </Shell>
  )
}

function Shell({ children }) {
  return (
    <main className="mx-auto max-w-6xl px-4 py-8">
      <Link to="/" className="text-sm text-accent">← All batches</Link>
      <div className="mt-4">{children}</div>
    </main>
  )
}
