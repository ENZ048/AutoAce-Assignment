import { useCallback, useEffect, useRef, useState } from 'react'
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
  const [busy, setBusy] = useState(false)
  const [pollNonce, setPollNonce] = useState(0)
  // Mirrors the latest known job status outside React state so the interval tick
  // (a plain callback, not a state updater) can decide whether to poll without
  // causing a network side effect from inside setJob. stoppedRef latches true on a
  // confirmed 404 so a deleted/nonexistent id is never polled again.
  const statusRef = useRef(null)
  const stoppedRef = useRef(false)

  const refresh = useCallback(() =>
    getJob(id).then((j) => {
      statusRef.current = j.status
      setJob(j)
    }).catch((e) => {
      if (e.response?.status === 404) {
        stoppedRef.current = true
        setMissing(true)
      }
    }), [id])

  // React Router keeps this component mounted across a `:id` change (no remount),
  // so the previous job's data and 404 flag would otherwise leak into the new url —
  // a live job could render as "no longer exists", or briefly show stale content.
  useEffect(() => {
    setJob(null)
    setMissing(false)
  }, [id])

  useEffect(() => {
    statusRef.current = null
    stoppedRef.current = false
    refresh()
    const t = setInterval(() => {
      if (stoppedRef.current) return
      if (statusRef.current === null || isActive(statusRef.current)) refresh()
    }, 2000)
    return () => clearInterval(t)
  }, [refresh, pollNonce])

  if (missing) return <Shell><p className="text-sm">This batch no longer exists.</p></Shell>
  if (!job) return <Shell><p className="text-sm">Loading…</p></Shell>

  // Shared by Start / Re-run / Discard: guards re-entry so a fast double-click can't
  // fire a second concurrent call, and bumping pollNonce on success re-triggers the
  // polling effect above, which restarts the interval and fetches immediately.
  const act = (fn, okMsg, onSuccess) => async () => {
    if (busy) return
    setBusy(true)
    try {
      await fn(job.id)
      okMsg && toast.success(okMsg)
      onSuccess?.()
    } catch (e) {
      toast.error(e.response?.data?.detail ?? 'Action failed')
    } finally {
      setBusy(false)
    }
  }
  const restartPolling = () => setPollNonce((n) => n + 1)

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

      {job.status === 'validating' && <p className="text-sm">Validating upload…</p>}
      {job.status === 'awaiting_confirmation' && (
        <ValidationReport job={job} disabled={busy}
          onStart={act(startJob, 'Batch queued', restartPolling)}
          onDiscard={act(deleteJob, undefined, () => navigate('/'))} />
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
          <button onClick={act(rerunJob, 'Batch re-queued', restartPolling)} disabled={busy}
            className="mt-4 rounded-lg bg-navy px-4 py-2 text-sm font-medium text-white disabled:opacity-60">
            Re-run batch
          </button>
        </section>
      )}
    </Shell>
  )
}

function Shell({ children }) {
  return (
    <main className="mx-auto max-w-7xl px-6 py-8">
      <Link to="/" className="text-sm text-accent">← All batches</Link>
      <div className="mt-4">{children}</div>
    </main>
  )
}
