const KEY = 'serving.apiKey'

export const getKey = () => sessionStorage.getItem(KEY) || ''
export const setKey = (k) => (k ? sessionStorage.setItem(KEY, k) : sessionStorage.removeItem(KEY))

function describe(detail, fallback) {
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    // FastAPI validation errors: [{loc: ['body', 'max_batch_size'], msg: '...'}]
    return detail.map((e) => `${(e.loc || []).slice(1).join('.')}: ${e.msg}`).join('; ')
  }
  return fallback
}

export async function api(path, { method = 'GET', body } = {}) {
  const headers = {}
  const key = getKey()
  if (key) headers['Authorization'] = `Bearer ${key}`
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  const res = await fetch(path, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const j = await res.json()
      detail = describe(j.detail, detail)
    } catch {
      /* non-JSON error body */
    }
    const err = new Error(detail || `HTTP ${res.status}`)
    err.status = res.status
    throw err
  }
  return res.json()
}
