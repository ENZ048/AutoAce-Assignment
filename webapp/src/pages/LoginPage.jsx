import { useState } from 'react'
import toast from 'react-hot-toast'
import { useNavigate } from 'react-router-dom'
import { login, setToken } from '../api'

export default function LoginPage() {
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
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
            <div className="relative mt-1">
              <input type={showPassword ? 'text' : 'password'} value={password}
                onChange={(e) => setPassword(e.target.value)}
                required autoComplete="current-password"
                className="w-full rounded-lg border border-gray-200 py-2 pl-3 pr-10 text-ink" />
              <button type="button" onClick={() => setShowPassword((v) => !v)}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
                title={showPassword ? 'Hide password' : 'Show password'}
                className="absolute inset-y-0 right-0 flex w-10 items-center justify-center text-gray-400 hover:text-ink">
                {showPassword ? (
                  /* eye-off */
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
                    <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
                    <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24" />
                    <line x1="1" y1="1" x2="23" y2="23" />
                  </svg>
                ) : (
                  /* eye */
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                    strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                    <circle cx="12" cy="12" r="3" />
                  </svg>
                )}
              </button>
            </div>
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
