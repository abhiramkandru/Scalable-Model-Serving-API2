import { fmtMs, fmtNum } from '../format'

// Burn rate 1 = consuming error budget exactly as fast as the SLO allows.
// 6x and 14.4x match the ticket / page thresholds in deploy/prometheus/rules.yml.
function tone(burn) {
  if (burn >= 14.4) return { text: 'text-rose-400', bar: 'bg-rose-500', label: 'paging' }
  if (burn >= 6) return { text: 'text-amber-400', bar: 'bg-amber-500', label: 'ticket' }
  if (burn > 1) return { text: 'text-yellow-300', bar: 'bg-yellow-400', label: 'above budget' }
  return { text: 'text-emerald-400', bar: 'bg-emerald-500', label: 'within budget' }
}

export default function SloPanel({ slo }) {
  if (!slo) return null
  const t = tone(slo.burn_rate)
  const pct = Math.min(100, (slo.burn_rate / 14.4) * 100)
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
      <h2 className="mb-2 text-sm font-medium text-slate-300">Service level objective</h2>
      <p className="text-xs text-slate-500">
        {(slo.target * 100).toFixed(2)}% of requests succeed within {fmtMs(slo.latency_ms)} · last {Math.round(slo.window_s / 60)} min
      </p>
      <div className="mt-3 flex items-end justify-between">
        <div>
          <div className="text-xs uppercase tracking-wide text-slate-400">Good requests</div>
          <div className="text-2xl font-semibold tabular-nums">{(slo.good_ratio * 100).toFixed(2)}%</div>
        </div>
        <div className="text-right">
          <div className="text-xs uppercase tracking-wide text-slate-400">Burn rate</div>
          <div className={`text-2xl font-semibold tabular-nums ${t.text}`}>{slo.burn_rate.toFixed(1)}×</div>
        </div>
      </div>
      <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-800" aria-hidden="true">
        <div className={`h-full ${t.bar}`} style={{ width: `${pct}%` }} />
      </div>
      <div className="mt-1 flex justify-between text-xs text-slate-500">
        <span>{t.label}</span>
        <span>{fmtNum(slo.requests)} requests in window</span>
      </div>
    </section>
  )
}
