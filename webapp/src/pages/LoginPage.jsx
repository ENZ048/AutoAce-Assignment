import { useState } from 'react'
import toast from 'react-hot-toast'
import { useNavigate } from 'react-router-dom'
import { login, setToken } from '../api'

export default function LoginPage() {
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    setBusy(true)
    try {
      const { access_token } = await login(username, password)
      setToken(access_token)
      navigate('/')
    } catch (err) {
      toast.error(err.response?.status === 401
        ? 'Wrong username or password'
        : 'Could not reach the server')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="min-h-screen flex items-center justify-center px-4">
      <div className="w-full max-w-sm rounded-xl border border-gray-200 bg-white p-8 shadow-sm">
        <h1 className="text-2xl font-bold">
          <span className="text-accent">AutoAce</span> Evaluation
        </h1>
        <p className="mt-1 text-sm">Sign in to upload and review analysis batches.</p>
        <form onSubmit={submit} className="mt-6 space-y-4">
          <label className="block text-sm">
            Username
            <input value={username} onChange={(e) => setUsername(e.target.value)} required
              autoComplete="username"
              className="mt-1 w-full rounded-lg border border-gray-200 px-3 py-2 text-ink" />
          </label>
          <label className="block text-sm">
            Password
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)}
              required autoComplete="current-password"
              className="mt-1 w-full rounded-lg border border-gray-200 px-3 py-2 text-ink" />
          </label>
          <button type="submit" disabled={busy}
            className="w-full rounded-lg bg-navy px-4 py-2.5 font-medium text-white disabled:opacity-60">
            {busy ? 'Signing in…' : 'Sign in'}
          </button>
        </form>
      </div>
    </main>
  )
}
