const API_BASE = import.meta.env.VITE_API_BASE_URL ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  if (!response.ok) {
    let detail: string | undefined
    const text = await response.text()
    if (text) {
      try {
        const parsed = JSON.parse(text) as { detail?: unknown; error?: unknown; message?: unknown }
        const candidate = parsed.detail ?? parsed.error ?? parsed.message
        detail = candidate === undefined ? text : String(candidate)
      } catch {
        detail = text
      }
    }
    throw new Error(detail || `Request failed: ${response.status}`)
  }
  return (await response.json()) as T
}

export const api = {
  configOptions: () =>
    request<{
      presets: string[]
      defaults: Record<string, unknown>
      pipeline_modes: string[]
      dataset_types: string[]
      dataset_sources: string[]
      dataset_names: string[]
      vector_db_types: string[]
      retrieval_modes: string[]
      hybrid_fusion_methods: string[]
      reranker_modes: string[]
      service_runtime: { startup_mode: string[]; offload_policy: string[] }
      tts_providers: string[]
      models: Record<string, Array<{ type: string; name: string }>>
    }>('/api/config/options'),
  createConfig: (payload: { preset_name?: string; config_patch?: Record<string, unknown>; auto_devices?: boolean }) =>
    request<{ config: Record<string, unknown>; flat: Record<string, unknown> }>('/api/config/create', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  validateConfig: (config: Record<string, unknown>) =>
    request<{ config: Record<string, unknown> }>('/api/config/validate', {
      method: 'POST',
      body: JSON.stringify({ config, auto_devices: true }),
    }),
  graphPreview: (config: Record<string, unknown>) =>
    request<{ levels: string[][]; nodes: Array<{ id: string; stage: string; depends_on: string[] }> }>('/api/graph/preview', {
      method: 'POST',
      body: JSON.stringify({ config, auto_devices: true }),
    }),
  submitEvaluation: (config: Record<string, unknown>) =>
    request<{ job_id: string }>('/api/jobs/evaluation', {
      method: 'POST',
      body: JSON.stringify({ config, auto_devices: true }),
    }),
  listJobs: () => request<{ jobs: Array<Record<string, unknown>> }>('/api/jobs'),
  getJob: (jobId: string) => request<Record<string, unknown>>(`/api/jobs/${jobId}`),
  cancelJob: (jobId: string) => request<Record<string, unknown>>(`/api/jobs/${jobId}/cancel`, { method: 'POST' }),
  getJobResult: (jobId: string) => request<{ result: Record<string, unknown> }>(`/api/jobs/${jobId}/result`),
  getJobMetadata: (jobId: string) => request<Record<string, unknown>>(`/api/jobs/${jobId}/metadata`),
  getJobArtifacts: (jobId: string) => request<Record<string, unknown>>(`/api/jobs/${jobId}/artifacts`),
  leaderboard: (metric: string) => request<{ rows: Array<Record<string, unknown>> }>(`/api/leaderboard?metric=${encodeURIComponent(metric)}`),
  runs: () => request<{ runs: Array<Record<string, unknown>> }>('/api/leaderboard/runs'),
  runDetails: (runId: number) => request<Record<string, unknown>>(`/api/leaderboard/runs/${runId}`),
  liveQuery: (config: Record<string, unknown>, queryText: string, k: number) =>
    request<{ results: Array<Record<string, unknown>>; query_text: string }>('/api/live/query', {
      method: 'POST',
      body: JSON.stringify({ config, query_text: queryText, k, auto_devices: true }),
    }),
}
