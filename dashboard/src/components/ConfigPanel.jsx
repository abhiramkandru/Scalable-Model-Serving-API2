import { useEffect, useState } from 'react'
import { api } from '../api'

const NUMERIC = [
  { key: 'max_batch_size', label: 'Max batch size', min: 1, max: 256, step: 1, hint: 'sequences decoded together per step' },
  { key: 'max_wait_ms', label: 'Batching window (ms)', min: 0, max: 50, step: 0.5, hint: 'wait for a batch to form when idle' },
  { key: 'max_batch_tokens', label: 'Prefill token budget', min: 256, max: 65536, step: 256, hint: 'uncached prompt tokens admitted per step' },
  { key: 'max_queue', label: 'Max queue', min: 10, max: 5000, step: 10, hint: 'beyond this, requests are shed (503)' },
  { key: 'request_timeout_s', label: 'Request timeout (s)', min: 1, max: 120, step: 1, hint: 'per-request deadline' },
]
const TOGGLES = [
  { key: 'enable_batching', label: 'Continuous batching', hint: 'off = one sequence at a time' },
  { key: 'enable_prefix_cache', label: 'Prefix / KV-cache reuse', hint: 'reuse cached prompt blocks' },
]

export function Toggle({ checked, onChange, label, hint }) {
  return (
    <label className="flex cursor-pointer items-start justify-between gap-3 py-1.5">
      <span>
        <span className="block text-sm text-slate-200">{label}</span>
        {hint ? <span className="block text-xs text-slate-500">{hint}</span> : null}
      </span>
      <input type="checkbox" className="mt-1 h-4 w-4" checked={checked} onChange={(e) => onChange(e.target.checked)} />
    </label>
  )
}

export function NumberField({ field, value, onChange }) {
  return (
    <div className="py-1.5">
      <div className="flex items-baseline justify-between">
        <label className="text-sm text-slate-200" htmlFor={`f-${field.key}`}>
          {field.label}
        </label>
        <input
          id={`f-${field.key}`}
          type="number"
          className="w-24 rounded-md border border-slate-700 bg-slate-950 px-2 py-0.5 text-right text-sm tabular-nums"
          min={field.min}
          max={field.max}
          step={field.step}
          value={value}
          onChange={(e) => onChange(e.target.value === '' ? '' : Number(e.target.value))}
        />
      </div>
      <input
        type="range"
        className="mt-1 w-full"
        min={field.min}
        max={field.max}
        step={field.step}
        value={value === '' ? field.min : value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="text-xs text-slate-500">{field.hint}</div>
    </div>
  )
}

export default function ConfigPanel({ onChanged }) {
  const [server, setServer] = useState(null)
  const [draft, setDraft] = useState(null)
  const [msg, setMsg] = useState(null)
  const [busy, setBusy] = useState(false)

  const load = async () => {
    try {
      const { config } = await api('/api/config')
      setServer(config)
      setDraft(config)
    } catch (e) {
      setMsg({ ok: false, text: e.message })
    }
  }
  useEffect(() => {
    load()
  }, [])

  if (!draft) {
    return <Shell>{msg ? <p className="text-sm text-rose-400">{msg.text}</p> : <p className="text-sm text-slate-500">Loading…</p>}</Shell>
  }

  const changes = Object.fromEntries(
    Object.keys(draft)
      .filter((k) => k !== 'shadow_fraction' && draft[k] !== server[k] && draft[k] !== '')
      .map((k) => [k, draft[k]]),
  )
  const dirty = Object.keys(changes).length > 0
  const set = (k) => (v) => setDraft((d) => ({ ...d, [k]: v }))

  const apply = async () => {
    setBusy(true)
    setMsg(null)
    try {
      const { config } = await api('/api/config', { method: 'PUT', body: changes })
      setServer(config)
      setDraft(config)
      setMsg({ ok: true, text: 'Applied. New settings take effect on the next scheduler step.' })
      onChanged?.()
    } catch (e) {
      setMsg({ ok: false, text: e.status === 403 ? 'An admin API key is required to change settings.' : e.message })
    } finally {
      setBusy(false)
    }
  }

  return (
    <Shell>
      <div className="divide-y divide-slate-800">
        <div className="pb-2">
          {TOGGLES.map((t) => (
            <Toggle key={t.key} checked={draft[t.key]} onChange={set(t.key)} label={t.label} hint={t.hint} />
          ))}
        </div>
        <div className="pt-2">
          {NUMERIC.map((f) => (
            <NumberField key={f.key} field={f} value={draft[f.key]} onChange={set(f.key)} />
          ))}
        </div>
      </div>
      <div className="mt-3 flex items-center gap-2">
        <button
          className="rounded-md bg-sky-500 px-3 py-1.5 text-sm font-medium text-slate-950 disabled:cursor-not-allowed disabled:opacity-40"
          disabled={!dirty || busy}
          onClick={apply}
        >
          {busy ? 'Applying…' : 'Apply'}
        </button>
        <button
          className="rounded-md border border-slate-700 px-3 py-1.5 text-sm text-slate-300 disabled:opacity-40"
          disabled={!dirty || busy}
          onClick={() => setDraft(server)}
        >
          Reset
        </button>
        {msg ? <span className={`text-xs ${msg.ok ? 'text-emerald-400' : 'text-rose-400'}`}>{msg.text}</span> : null}
      </div>
    </Shell>
  )
}

function Shell({ children }) {
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
      <h2 className="mb-2 text-sm font-medium text-slate-300">Runtime configuration</h2>
      {children}
    </section>
  )
}
