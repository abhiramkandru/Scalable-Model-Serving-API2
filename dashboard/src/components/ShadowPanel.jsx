import { useEffect, useState } from 'react'
import { api } from '../api'
import { usePoll } from '../hooks'
import { fmtMs, fmtPct } from '../format'
import { NumberField, Toggle } from './ConfigPanel'

const FIELDS = [
  { key: 'max_batch_size', label: 'Candidate: max batch size', min: 1, max: 256, step: 1, hint: 'applied to the shadow backend only' },
  { key: 'max_wait_ms', label: 'Candidate: batching window (ms)', min: 0, max: 50, step: 0.5, hint: 'applied to the shadow backend only' },
]

export default function ShadowPanel() {
  const [refresh, setRefresh] = useState(0)
  const { data: shadow, error } = usePoll('/api/shadow', 2000, true, refresh)
  const [fraction, setFraction] = useState(10)
  const [cand, setCand] = useState(null)
  const [msg, setMsg] = useState(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api('/api/config')
      .then(({ config }) =>
        setCand({
          max_batch_size: config.max_batch_size,
          max_wait_ms: config.max_wait_ms,
          enable_prefix_cache: config.enable_prefix_cache,
          enable_batching: config.enable_batching,
        }),
      )
      .catch(() => {})
  }, [])

  // When a shadow is already running (e.g. page reload), show its parameters.
  useEffect(() => {
    if (shadow?.active) {
      setFraction(Math.round(shadow.fraction * 100))
      setCand((c) => ({ ...(c || {}), ...shadow.overrides }))
    }
    // only re-sync when the shadow flips between active/inactive
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shadow?.active])

  const act = async (fn, okText) => {
    setBusy(true)
    setMsg(null)
    try {
      await fn()
      setMsg({ ok: true, text: okText })
      setRefresh((r) => r + 1)
    } catch (e) {
      setMsg({ ok: false, text: e.status === 403 ? 'An admin API key is required.' : e.message })
    } finally {
      setBusy(false)
    }
  }

  const start = () =>
    act(() => api('/api/shadow', { method: 'PUT', body: { fraction: fraction / 100, overrides: cand } }),
      shadow?.active ? 'Shadow updated.' : 'Shadow started: mirroring traffic.')
  const stop = () => act(() => api('/api/shadow', { method: 'DELETE' }), 'Shadow stopped.')
  const promote = () => {
    if (!window.confirm('Apply the candidate settings to the primary backend?')) return
    return act(() => api('/api/shadow/promote', { method: 'POST' }), 'Promoted: primary now uses the candidate settings.')
  }

  const hasData = shadow && (shadow.active || shadow.samples > 0)
  const delta = shadow?.p95_delta_pct
  const deltaTone = delta == null ? 'text-slate-400' : delta <= 0 ? 'text-emerald-400' : delta < 15 ? 'text-amber-400' : 'text-rose-400'

  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
      <h2 className="mb-1 text-sm font-medium text-slate-300">Shadow traffic rollout</h2>
      <p className="mb-2 text-xs text-slate-500">
        Mirror a share of live requests to a candidate configuration. Clients only ever see the primary&apos;s response.
      </p>

      {cand ? (
        <div className="divide-y divide-slate-800">
          <div className="pb-2">
            <NumberField
              field={{ key: 'shadow_pct', label: 'Traffic mirrored (%)', min: 0, max: 100, step: 1, hint: 'share of requests copied to the shadow' }}
              value={fraction}
              onChange={setFraction}
            />
          </div>
          <div className="pt-2">
            {FIELDS.map((f) => (
              <NumberField key={f.key} field={f} value={cand[f.key]} onChange={(v) => setCand((c) => ({ ...c, [f.key]: v }))} />
            ))}
            <Toggle
              checked={!!cand.enable_prefix_cache}
              onChange={(v) => setCand((c) => ({ ...c, enable_prefix_cache: v }))}
              label="Candidate: prefix cache"
            />
            <Toggle
              checked={!!cand.enable_batching}
              onChange={(v) => setCand((c) => ({ ...c, enable_batching: v }))}
              label="Candidate: continuous batching"
            />
          </div>
        </div>
      ) : (
        <p className="text-sm text-slate-500">{error ? error.message : 'Loading…'}</p>
      )}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button className="rounded-md bg-sky-500 px-3 py-1.5 text-sm font-medium text-slate-950 disabled:opacity-40" disabled={busy || !cand} onClick={start}>
          {shadow?.active ? 'Update shadow' : 'Start shadow'}
        </button>
        <button className="rounded-md border border-slate-700 px-3 py-1.5 text-sm text-slate-300 disabled:opacity-40" disabled={busy || !shadow?.active} onClick={stop}>
          Stop
        </button>
        <button className="rounded-md border border-emerald-700 px-3 py-1.5 text-sm text-emerald-300 disabled:opacity-40" disabled={busy || !shadow?.active} onClick={promote}>
          Promote to primary
        </button>
        {msg ? <span className={`text-xs ${msg.ok ? 'text-emerald-400' : 'text-rose-400'}`}>{msg.text}</span> : null}
      </div>

      {hasData ? (
        <table className="mt-4 w-full text-sm tabular-nums">
          <thead className="text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="py-1 text-left font-medium"> </th>
              <th className="py-1 text-right font-medium">p50</th>
              <th className="py-1 text-right font-medium">p95</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="py-1 text-slate-400">Primary (same requests)</td>
              <td className="text-right">{fmtMs(shadow.primary.p50_ms)}</td>
              <td className="text-right">{fmtMs(shadow.primary.p95_ms)}</td>
            </tr>
            <tr>
              <td className="py-1 text-slate-400">Shadow candidate</td>
              <td className="text-right">{fmtMs(shadow.shadow.p50_ms)}</td>
              <td className="text-right">{fmtMs(shadow.shadow.p95_ms)}</td>
            </tr>
          </tbody>
          <tfoot className="text-xs text-slate-500">
            <tr>
              <td colSpan={3} className="pt-2">
                {shadow.samples} compared · p95 delta{' '}
                <span className={deltaTone}>{delta == null ? '–' : `${delta > 0 ? '+' : ''}${delta.toFixed(0)}%`}</span> · output match{' '}
                {shadow.output_match_rate == null ? '–' : fmtPct(shadow.output_match_rate)} · {shadow.errors} errors
              </td>
            </tr>
          </tfoot>
        </table>
      ) : null}
    </section>
  )
}
