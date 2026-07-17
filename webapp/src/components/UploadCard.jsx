import { useRef, useState } from 'react'
import toast from 'react-hot-toast'
import { createJob } from '../api'

export default function UploadCard({ onCreated }) {
  const zipRef = useRef()
  const folderRef = useRef()
  const [progress, setProgress] = useState(null) // null | 0..100

  const send = async (files) => {
    if (!files.length) return
    const form = new FormData()
    for (const f of files) form.append('files', f, f.name)
    setProgress(0)
    try {
      const job = await createJob(form, (e) =>
        setProgress(e.total ? Math.round((e.loaded / e.total) * 100) : 0))
      onCreated(job)
    } catch (err) {
      toast.error(err.response?.data?.detail ?? 'Upload failed')
    } finally {
      setProgress(null)
    }
  }

  return (
    <section
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => { e.preventDefault(); send([...e.dataTransfer.files]) }}
      className="rounded-xl border-2 border-dashed border-gray-200 bg-white p-6 text-center"
    >
      <h2 className="text-lg font-semibold">Upload a batch</h2>
      <p className="mx-auto mt-1 max-w-md text-sm">
        Drop a ZIP here, or choose a ZIP / folder. A batch is audio files plus one CSV manifest
        (<code className="font-mono">name,result_json</code>) at the folder root.
      </p>
      {progress === null ? (
        <div className="mt-4 flex justify-center gap-3">
          <button onClick={() => zipRef.current.click()}
            className="rounded-lg bg-navy px-4 py-2 text-sm font-medium text-white">
            Choose ZIP
          </button>
          <button onClick={() => folderRef.current.click()}
            className="rounded-lg border border-gray-200 px-4 py-2 text-sm font-medium text-ink">
            Choose folder
          </button>
          <input ref={zipRef} type="file" accept=".zip" hidden
            onChange={(e) => send([...e.target.files])} />
          <input ref={folderRef} type="file" webkitdirectory="" hidden
            onChange={(e) => send([...e.target.files])} />
        </div>
      ) : (
        <div className="mx-auto mt-4 max-w-md">
          <div className="h-2 overflow-hidden rounded-full bg-gray-200">
            <div className="h-full rounded-full bg-accent transition-all"
              style={{ width: `${progress}%` }} />
          </div>
          <p className="mt-1 text-xs">Uploading… {progress}%</p>
        </div>
      )}
    </section>
  )
}
