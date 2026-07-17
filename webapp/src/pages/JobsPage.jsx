import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { clearToken, listJobs } from '../api'
import StatusChip from '../components/StatusChip'
import UploadCard from '../components/UploadCard'
import { fmtTime, shortId } from '../lib/status'

export default function JobsPage() {
  const navigate = useNavigate()
  const [jobs, setJobs] = useState([])

  useEffect(() => {
    let alive = true
    const load = () => listJobs().then((j) => alive && setJobs(j)).catch(() => {})
    load()
    const t = setInterval(load, 3000)
    return () => { alive = false; clearInterval(t) }
  }, [])

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
                <th className="px-4 py-3">Files</th><th className="px-4 py-3" />
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
                  <td className="px-4 py-3 text-right">
                    <button onClick={() => navigate(`/jobs/${j.id}`)}
                      className="font-medium text-accent">Open</button>
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
