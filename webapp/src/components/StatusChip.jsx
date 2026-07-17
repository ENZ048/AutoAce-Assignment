import { STATUS_META } from '../lib/status'

export default function StatusChip({ status }) {
  const meta = STATUS_META[status] ?? { label: status, chip: 'bg-gray-100 text-gray-700' }
  return (
    <span className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${meta.chip}`}>
      {meta.label}
    </span>
  )
}
