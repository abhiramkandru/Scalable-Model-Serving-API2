import { useEffect, useState } from 'react'
import { api } from './api'

/** Poll a GET endpoint. `refreshKey` forces an immediate refetch when it changes. */
export function usePoll(path, intervalMs, enabled = true, refreshKey = 0) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => {
    if (!enabled) return undefined
    let alive = true
    let timer
    const tick = async () => {
      try {
        const d = await api(path)
        if (alive) {
          setData(d)
          setError(null)
        }
      } catch (e) {
        if (alive) setError(e)
      }
      if (alive) timer = setTimeout(tick, intervalMs)
    }
    tick()
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [path, intervalMs, enabled, refreshKey])
  return { data, error }
}

/** Seeds from /api/history, then appends /api/stats once per second (keeps the last 5 minutes). */
export function useStats(enabled) {
  const [history, setHistory] = useState([])
  const [latest, setLatest] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => {
    if (!enabled) return undefined
    let alive = true
    let timer
    api('/api/history')
      .then((h) => {
        if (!alive) return
        const lastTs = h.length ? h[h.length - 1].ts : 0
        setHistory((prev) => [...h, ...prev.filter((p) => p.ts > lastTs)].slice(-300))
      })
      .catch((e) => alive && setError(e))
    const tick = async () => {
      try {
        const s = await api('/api/stats')
        if (alive) {
          setLatest(s)
          setError(null)
          setHistory((h) => (h.length && h[h.length - 1].ts === s.ts ? h : [...h, s].slice(-300)))
        }
      } catch (e) {
        if (alive) setError(e)
      }
      if (alive) timer = setTimeout(tick, 1000)
    }
    tick()
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [enabled])
  return { latest, history, error }
}
