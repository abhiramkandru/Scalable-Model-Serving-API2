import { usePoll } from '../hooks'
import { fmtMs, fmtNum, fmtPct } from '../format'

export default function ClusterTable() {
  const { data, error } = usePoll('/api/cluster', 3000)
  const nodes = data?.nodes ?? []
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="text-sm font-medium text-slate-300">Inference nodes</h2>
        <span className="text-xs text-slate-500">
          {data ? (data.source === 'prometheus' ? 'all replicas via Prometheus' : 'this node only (set PROMETHEUS_URL for all replicas)') : ''}
        </span>
      </div>
      {error ? <p className="text-sm text-rose-400">{error.message}</p> : null}
      <div className="overflow-x-auto">
        <table className="w-full min-w-[34rem] text-sm tabular-nums">
          <thead className="text-xs uppercase tracking-wide text-slate-500">
            <tr>
              <th className="py-1 text-left font-medium">Node</th>
              <th className="py-1 text-right font-medium">GPU</th>
              <th className="py-1 text-right font-medium">Queue</th>
              <th className="py-1 text-right font-medium">Running</th>
              <th className="py-1 text-right font-medium">KV cache</th>
              <th className="py-1 text-right font-medium">Req/s</th>
              <th className="py-1 text-right font-medium">p95</th>
            </tr>
          </thead>
          <tbody>
            {nodes.map((n) => (
              <tr key={n.node} className="border-t border-slate-800">
                <td className="max-w-[14rem] truncate py-1.5 text-slate-300">{n.node}</td>
                <td className="text-right">{n.gpu_util == null ? '–' : `${Math.round(n.gpu_util)}%`}</td>
                <td className="text-right">{fmtNum(n.queue)}</td>
                <td className="text-right">{fmtNum(n.running)}</td>
                <td className="text-right">{fmtPct(n.kv)}</td>
                <td className="text-right">{fmtNum(n.rps, 1)}</td>
                <td className="text-right">{fmtMs(n.p95_ms)}</td>
              </tr>
            ))}
            {!nodes.length && !error ? (
              <tr>
                <td colSpan={7} className="py-3 text-center text-slate-500">
                  No data yet
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </section>
  )
}
