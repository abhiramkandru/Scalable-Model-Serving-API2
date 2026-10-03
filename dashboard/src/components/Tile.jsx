const TONES = {
  default: 'text-slate-100',
  good: 'text-emerald-400',
  warn: 'text-amber-400',
  bad: 'text-rose-400',
}

export default function Tile({ label, value, sub, tone = 'default' }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
      <div className={`mt-1 text-2xl font-semibold tabular-nums ${TONES[tone]}`}>{value}</div>
      {sub ? <div className="mt-0.5 truncate text-xs text-slate-500">{sub}</div> : null}
    </div>
  )
}
