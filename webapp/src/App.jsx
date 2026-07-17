import { Toaster } from 'react-hot-toast'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { getToken } from './api'
import JobPage from './pages/JobPage'
import JobsPage from './pages/JobsPage'
import LoginPage from './pages/LoginPage'

function RequireAuth({ children }) {
  return getToken() ? children : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <BrowserRouter>
      <Toaster position="top-right" />
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/" element={<RequireAuth><JobsPage /></RequireAuth>} />
        <Route path="/jobs/:id" element={<RequireAuth><JobPage /></RequireAuth>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
