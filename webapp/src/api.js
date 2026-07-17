import axios from 'axios'

const TOKEN_KEY = 'autoace_dashboard_token'
export const getToken = () => localStorage.getItem(TOKEN_KEY)
export const setToken = (t) => localStorage.setItem(TOKEN_KEY, t)
export const clearToken = () => localStorage.removeItem(TOKEN_KEY)

const client = axios.create({ baseURL: '' })

client.interceptors.request.use((config) => {
  const t = getToken()
  if (t) config.headers.Authorization = `Bearer ${t}`
  return config
})

client.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401 && !err.config?.url?.includes('/auth/login')) {
      clearToken()
      window.location.assign('/login')
    }
    return Promise.reject(err)
  },
)

export const login = (username, password) =>
  client.post('/api/auth/login', { username, password }).then((r) => r.data)
export const listJobs = () => client.get('/api/jobs').then((r) => r.data)
export const getJob = (id) => client.get(`/api/jobs/${id}`).then((r) => r.data)
export const createJob = (formData, onUploadProgress) =>
  client.post('/api/jobs', formData, { onUploadProgress }).then((r) => r.data)
export const startJob = (id) => client.post(`/api/jobs/${id}/start`).then((r) => r.data)
export const rerunJob = (id) => client.post(`/api/jobs/${id}/rerun`).then((r) => r.data)
export const deleteJob = (id) => client.delete(`/api/jobs/${id}`).then((r) => r.data)
export const getResults = (id) => client.get(`/api/jobs/${id}/results`).then((r) => r.data)
export const getErrors = (id) => client.get(`/api/jobs/${id}/errors`).then((r) => r.data)

export const downloadArtifact = async (id, name) => {
  const r = await client.get(`/api/jobs/${id}/download/${name}`, { responseType: 'blob' })
  const url = URL.createObjectURL(r.data)
  const a = document.createElement('a')
  a.href = url
  a.download = name
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
