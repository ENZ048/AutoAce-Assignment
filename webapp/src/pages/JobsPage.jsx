import { useCallback, useEffect, useRef, useState } from 'react'
import toast from 'react-hot-toast'
import { useNavigate } from 'react-router-dom'
import { clearToken, deleteJob, listJobs } from '../api'
import StatusChip from '../components/StatusChip'
import UploadCard from '../components/UploadCard'
import { fmtTime, isTerminal, shortId } from '../lib/status'

export default function JobsPage() {
  const navigate = useNavigate()
  const [jobs, setJobs] = useState([])
  const aliveRef = useRef(true)

  const load = useCallback(() =>
    listJobs().then((j) => { if (aliveRef.current) setJobs(j) }).catch(() => {}), [])

  useEffect(() => {
    aliveRef.current = true
    load()
    const t = setInterval(load, 3000)
    return () => { aliveRef.current = false; clearInterval(t) }
  }, [load])

  const handleDelete = async (job) => {
    if (!window.confirm(`Delete batch "${job.original_name}"? This cannot be undone.`)) return
    try {
      await deleteJob(job.id)
      load()
    } catch (e) {
      toast.error(e.response?.data?.detail ?? 'Delete failed')
    }
  }

  return (
    <main className="mx-auto max-w-5xl px-4 py-8">
      <header className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-bold">
          <span className="text-accent">Batch</span> analysis
        </h1>
        <button onClick={() => { clearToken(); navigate('/login') }}
          className="text-sm underline">Sign out</button>
      </header>
      <UploadCard onCreated={(job) => navigate(`/jobs/${job.id}`)} />
      <section className="mt-8 overflow-x-auto rounded-xl border border-gray-200 bg-white">
        {jobs.length === 0 ? (
          <p className="p-6 text-center text-sm">
            No batches yet. Upload one above to see validation, progress and results here.
          </p>
        ) : (
          <table className="w-full text-left text-sm">
            <thead className="border-b border-gray-200 text-xs uppercase tracking-wide">
              <tr>
                <th className="px-4 py-3">Batch</th><th className="px-4 py-3">Uploaded</th>
                <th className="px-4 py-3">Status</th><th className="px-4 py-3">Progress</th>
                <th className="px-4 py-3">Files</th><th className="px-4 py-3">Actions</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id} className="border-b border-gray-100 last:border-0">
                  <td className="px-4 py-3">
                    <div className="font-medium text-ink">{j.original_name}</div>
                    <div className="font-mono text-xs text-gray-400">{shortId(j.id)}</div>
                  </td>
                  <td className="px-4 py-3">{fmtTime(j.created_at)}</td>
                  <td className="px-4 py-3"><StatusChip status={j.status} /></td>
                  <td className="px-4 py-3 font-mono">
                    {j.status === 'running' ? `${j.done}/${j.total}` : '—'}
                  </td>
                  <td className="px-4 py-3">{j.total}</td>
                  <td className="px-4 py-3">
                    <div className="flex gap-2">
                      <button onClick={() => navigate(`/jobs/${j.id}`)}
                        className="rounded-lg border border-accent/30 bg-accent/5 px-3 py-1 text-xs font-medium text-accent hover:bg-accent/10">
                        Open
                      </button>
                      {/* deletable whenever the API allows it: anything not queued/running
                          (a running batch must be left to finish or fail first) */}
                      {(isTerminal(j.status) || j.status === 'awaiting_confirmation') && (
                        <button onClick={() => handleDelete(j)}
                          className="rounded-lg border border-red-200 bg-red-50 px-3 py-1 text-xs font-medium text-red-700 hover:bg-red-100">
                          Delete
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </main>
  )
}
