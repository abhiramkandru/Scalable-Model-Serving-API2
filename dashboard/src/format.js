export const fmtMs = (v) => (v == null ? '–' : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`)
export const fmtPct = (ratio, digits = 0) => (ratio == null ? '–' : `${(ratio * 100).toFixed(digits)}%`)
export const fmtNum = (v, digits = 0) => (v == null ? '–' : Number(v).toLocaleString(undefined, { maximumFractionDigits: digits }))
