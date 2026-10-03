import { useEffect, useMemo, useState } from 'react'
import { api, getKey, setKey } from './api'
import { useStats } from './hooks'
import { fmtMs, fmtNum, fmtPct } from './format'
import Tile from './components/Tile'
import LineChart from './components/LineChart'
import ConfigPanel from './components/ConfigPanel'
import ShadowPanel from './components/ShadowPanel'
import ClusterTable from './components/ClusterTable'
import SloPanel from './components/SloPanel'

const C = { sky: '#38bdf8', emerald: '#34d399', amber: '#fbbf24', rose: '#fb7185', violet: '#a78bfa' }

export default function App() {
  const [info, setInfo] = useState(null)
  const [infoError, setInfoError] = useState(null)
  const [keyDraft, setKeyDraft] = useState(getKey())
  const [keyVersion, setKeyVersion] = useState(0)

  useEffect(() => {
    api('/api/info')
      .then(setInfo)
      .catch((e) => setInfoError(e))
  }, [])

  const { latest, history, error } = useStats(info != null)
  const unauthorized = error?.status === 401

  const data = useMemo(
    () => history.map((s) => ({ ...s, kv_pct: s.kv_usage * 100, prefix_pct: s.prefix_hit_rate * 100 })),
    [history],
  )

  const applyKey = () => {
    setKey(keyDraft.trim())
    setKeyVersion((v) => v + 1) // remount panels so they refetch with the new key
  }

  if (infoError) {
    return (
      <Page>
        <div className="rounded-xl border border-rose-900 bg-rose-950/40 p-4 text-sm text-rose-300">
          Cannot reach the serving API: {infoError.message}. In development, start it on port 8000 (see README).
        </div>
      </Page>
    )
  }

  const err = latest?.error_rate ?? 0
  const gpu = latest?.gpu_util ?? 0
  const queue = latest?.queue_depth ?? 0

  return (
    <Page
      header={
        <header className="mb-6 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold text-slate-100">Model serving</h1>
            <p className="text-xs text-slate-500">
              {info ? `${info.node} · ${info.backend} backend · v${info.version}` : 'connecting…'}
              {error && !unauthorized ? <span className="ml-2 text-rose-400">connection problem: {error.message}</span> : null}
            </p>
          </div>
          {info?.auth_required ? (
            <form
              className="flex items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault()
                applyKey()
              }}
            >
              <input
                type="password"
                autoComplete="off"
                placeholder="API key"
                value={keyDraft}
                onChange={(e) => setKeyDraft(e.target.value)}
                className="w-56 rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-sm"
              />
              <button className="rounded-md border border-slate-700 px-3 py-1 text-sm text-slate-200">Use key</button>
            </form>
          ) : null}
        </header>
      }
    >
      {unauthorized ? (
        <div className="mb-6 rounded-xl border border-amber-900 bg-amber-950/40 p-4 text-sm text-amber-200">
          This server requires an API key. Enter one above (it is kept in this tab&apos;s session storage only).
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Tile label="Requests / s" value={fmtNum(latest?.rps, 1)} sub={`${fmtNum(latest?.tokens_per_s)} tokens/s`} />
        <Tile
          label="p95 latency"
          value={fmtMs(latest?.p95_ms)}
          sub={`p50 ${fmtMs(latest?.p50_ms)} · p99 ${fmtMs(latest?.p99_ms)}`}
          tone={latest && latest.p95_ms > (latest.slo?.latency_ms ?? 500) ? 'warn' : 'default'}
        />
        <Tile
          label="GPU utilisation"
          value={`${Math.round(gpu)}%`}
          sub={latest?.gpu_source === 'simulated' ? 'simulated (no GPU / NVML)' : latest?.gpu_source}
          tone={gpu > 95 ? 'warn' : 'default'}
        />
        <Tile
          label="Queue depth"
          value={fmtNum(queue)}
          sub={`${fmtNum(latest?.running)} running · batch ≈ ${fmtNum(latest?.batch_size, 1)}`}
          tone={queue > 50 ? 'warn' : 'default'}
        />
        <Tile label="Error rate" value={fmtPct(err, 2)} sub="errors, timeouts, shed" tone={err > 0.01 ? 'bad' : err > 0 ? 'warn' : 'good'} />
        <Tile label="KV cache" value={fmtPct(latest?.kv_usage)} sub="blocks in use" tone={(latest?.kv_usage ?? 0) > 0.9 ? 'warn' : 'default'} />
        <Tile label="Prefix cache hits" value={fmtPct(latest?.prefix_hit_rate)} sub="prompt tokens reused (10 s)" />
        <Tile
          label="Shadow traffic"
          value={fmtPct(latest?.shadow_fraction)}
          sub={latest?.shadow_fraction ? 'mirroring to candidate' : 'off'}
        />
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-2">
        <LineChart
          title="Latency"
          data={data}
          series={[
            { key: 'p50_ms', label: 'p50', color: C.emerald },
            { key: 'p95_ms', label: 'p95', color: C.sky },
            { key: 'p99_ms', label: 'p99', color: C.violet },
          ]}
          format={fmtMs}
        />
        <LineChart
          title="Throughput"
          data={data}
          series={[{ key: 'rps', label: 'req/s', color: C.sky }]}
          format={(v) => (v < 10 && v > 0 ? v.toFixed(1) : v.toFixed(0))}
        />
        <LineChart
          title="GPU & cache"
          data={data}
          yMax={100}
          series={[
            { key: 'gpu_util', label: 'GPU %', color: C.amber },
            { key: 'kv_pct', label: 'KV %', color: C.violet },
            { key: 'prefix_pct', label: 'prefix hit %', color: C.emerald },
          ]}
          format={(v) => `${Math.round(v)}%`}
        />
        <LineChart
          title="Queue & batching"
          data={data}
          series={[
            { key: 'queue_depth', label: 'queued', color: C.rose },
            { key: 'running', label: 'running', color: C.sky },
            { key: 'batch_size', label: 'avg batch', color: C.emerald },
          ]}
          format={(v) => v.toFixed(0)}
        />
      </div>

      <div key={keyVersion} className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-2">
        <ConfigPanel />
        <ShadowPanel />
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-3">
        <SloPanel slo={latest?.slo} />
        <div className="lg:col-span-2">
          <ClusterTable key={keyVersion} />
        </div>
      </div>
    </Page>
  )
}

function Page({ header, children }) {
  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      {header}
      {children}
    </div>
  )
}
