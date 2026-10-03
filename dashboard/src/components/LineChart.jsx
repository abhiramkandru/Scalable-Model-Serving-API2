import { useMemo, useState } from 'react'

const W = 600
const H = 170
const PAD = { l: 48, r: 10, t: 10, b: 20 }

function niceCeil(v) {
  const p = Math.pow(10, Math.floor(Math.log10(v)))
  const f = v / p
  const nf = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10
  return nf * p
}

/**
 * Small dependency-free SVG line chart.
 * series: [{ key, label, color }]; data: array of snapshot objects (one per second).
 */
export default function LineChart({ title, data, series, format = (v) => v.toFixed(1), yMax }) {
  const [hover, setHover] = useState(null)
  const n = data.length
  const innerW = W - PAD.l - PAD.r
  const innerH = H - PAD.t - PAD.b

  const { max, paths } = useMemo(() => {
    let m = yMax ?? 0
    if (yMax == null) {
      for (const d of data) for (const s of series) m = Math.max(m, d[s.key] ?? 0)
    }
    m = niceCeil(m > 0 ? m : 1)
    const x = (i) => PAD.l + (n <= 1 ? 0 : (i / (n - 1)) * innerW)
    const y = (v) => PAD.t + (1 - Math.min(v, m) / m) * innerH
    const ps = series.map((s) =>
      data.map((d, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(d[s.key] ?? 0).toFixed(1)}`).join(' '),
    )
    return { max: m, paths: ps }
  }, [data, series, yMax, n, innerW, innerH])

  const onMove = (e) => {
    if (n < 2) return
    const rect = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * W
    const idx = Math.round(((px - PAD.l) / innerW) * (n - 1))
    setHover(Math.max(0, Math.min(n - 1, idx)))
  }

  const shown = n ? data[hover ?? n - 1] : null
  const hoverX = hover == null || n < 2 ? null : PAD.l + (hover / (n - 1)) * innerW

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h3 className="text-sm font-medium text-slate-300">{title}</h3>
        <div className="flex flex-wrap gap-x-4 text-xs tabular-nums">
          {series.map((s) => (
            <span key={s.key} className="flex items-center gap-1.5 text-slate-400">
              <span className="inline-block h-2 w-2 rounded-full" style={{ background: s.color }} />
              {s.label}
              <span className="text-slate-200">{shown ? format(shown[s.key] ?? 0) : '–'}</span>
            </span>
          ))}
        </div>
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label={title}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const yy = PAD.t + (1 - f) * innerH
          return (
            <g key={f}>
              <line x1={PAD.l} x2={W - PAD.r} y1={yy} y2={yy} stroke="#1e293b" strokeWidth="1" />
              <text x={PAD.l - 6} y={yy + 3} textAnchor="end" fontSize="10" fill="#64748b">
                {format(max * f)}
              </text>
            </g>
          )
        })}
        <text x={PAD.l} y={H - 4} fontSize="10" fill="#64748b">
          {n > 1 ? `-${n - 1}s` : ''}
        </text>
        <text x={W - PAD.r} y={H - 4} textAnchor="end" fontSize="10" fill="#64748b">
          now
        </text>
        {paths.map((d, i) => (
          <path key={series[i].key} d={d} fill="none" stroke={series[i].color} strokeWidth="1.75" strokeLinejoin="round" />
        ))}
        {hoverX != null ? <line x1={hoverX} x2={hoverX} y1={PAD.t} y2={H - PAD.b} stroke="#475569" strokeDasharray="3 3" /> : null}
      </svg>
    </div>
  )
}
