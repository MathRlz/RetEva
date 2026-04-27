import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'

type JsonRecord = Record<string, unknown>

type ConfigOptions = {
  presets: string[]
  defaults: JsonRecord
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
}

type Props = {
  configText: string
  setConfigText: (value: string) => void
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function labelFromPath(path: string): string {
  const key = path.split('.').at(-1) ?? path
  return key
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (m) => m.toUpperCase())
}

function getAtPath(root: unknown, path: string): unknown {
  return path.split('.').reduce<unknown>((acc, key) => (isRecord(acc) ? acc[key] : undefined), root)
}

function setAtPath(root: JsonRecord, path: string, value: unknown): JsonRecord {
  const keys = path.split('.')
  const copy = structuredClone(root) as JsonRecord
  let cursor: JsonRecord = copy
  for (const key of keys.slice(0, -1)) {
    const next = cursor[key]
    if (!isRecord(next)) {
      cursor[key] = {}
    }
    cursor = cursor[key] as JsonRecord
  }
  cursor[keys.at(-1) ?? path] = value
  return copy
}

function parseByCurrentType(raw: string, current: unknown): unknown {
  if (Array.isArray(current)) {
    try {
      const parsed = JSON.parse(raw)
      return Array.isArray(parsed) ? parsed : current
    } catch {
      return current
    }
  }
  if (isRecord(current)) {
    try {
      const parsed = JSON.parse(raw)
      return isRecord(parsed) ? parsed : current
    } catch {
      return current
    }
  }
  if (typeof current === 'number') {
    const numeric = Number(raw)
    return Number.isFinite(numeric) ? numeric : current
  }
  if (typeof current === 'boolean') {
    return raw === 'true'
  }
  return raw
}

export function ConfigPage({ configText, setConfigText }: Props) {
  const [options, setOptions] = useState<ConfigOptions | null>(null)
  const [selectedPreset, setSelectedPreset] = useState('')
  const [formConfig, setFormConfig] = useState<JsonRecord | null>(null)
  const [graph, setGraph] = useState<string>('')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [allowJsonEdit, setAllowJsonEdit] = useState(false)

  useEffect(() => {
    void (async () => {
      const loaded = await api.configOptions()
      setOptions(loaded)
      if ((loaded.presets || []).length > 0) {
        setSelectedPreset(loaded.presets[0])
      }
      if (isRecord(loaded.defaults)) {
        setFormConfig(loaded.defaults)
      }
    })()
  }, [])

  useEffect(() => {
    if (!formConfig) {
      return
    }
    setConfigText(JSON.stringify(formConfig, null, 2))
  }, [formConfig, setConfigText])

  const modelTypeOptions = useMemo(() => {
    if (!options) {
      return { asr: [], text: [], audio: [] } as Record<string, string[]>
    }
    return {
      asr: Array.from(new Set((options.models.asr || []).map((entry) => entry.type))).sort(),
      text: Array.from(new Set((options.models.text_embedding || []).map((entry) => entry.type))).sort(),
      audio: Array.from(new Set((options.models.audio_embedding || []).map((entry) => entry.type))).sort(),
    }
  }, [options])

  const modelNameOptions = useMemo(() => {
    if (!options || !formConfig) {
      return { asr: [], text: [], audio: [] } as Record<string, string[]>
    }
    const asrType = String(getAtPath(formConfig, 'runtime.model.asr_model_type') ?? '')
    const textType = String(getAtPath(formConfig, 'runtime.model.text_emb_model_type') ?? '')
    const audioType = String(getAtPath(formConfig, 'runtime.model.audio_emb_model_type') ?? '')
    return {
      asr: (options.models.asr || []).filter((entry) => entry.type === asrType).map((entry) => entry.name),
      text: (options.models.text_embedding || []).filter((entry) => entry.type === textType).map((entry) => entry.name),
      audio: (options.models.audio_embedding || []).filter((entry) => entry.type === audioType).map((entry) => entry.name),
    }
  }, [options, formConfig])

  function selectOptionsForPath(path: string): string[] | null {
    if (!options) {
      return null
    }
    const staticMap: Record<string, string[]> = {
      'runtime.model.pipeline_mode': options.pipeline_modes || [],
      'runtime.data.dataset_type': options.dataset_types || [],
      'runtime.data.dataset_source': options.dataset_sources || [],
      'runtime.data.dataset_name': options.dataset_names || [],
      'runtime.vector_db.type': options.vector_db_types || [],
      'runtime.vector_db.retrieval_mode': options.retrieval_modes || [],
      'runtime.vector_db.reranker_mode': options.reranker_modes || [],
      'runtime.vector_db.hybrid_fusion_method': options.hybrid_fusion_methods || [],
      'experiment.service_runtime.startup_mode': options.service_runtime.startup_mode || [],
      'experiment.service_runtime.offload_policy': options.service_runtime.offload_policy || [],
      'experiment.audio_synthesis.provider': options.tts_providers || [],
      'runtime.model.asr_model_type': modelTypeOptions.asr,
      'runtime.model.text_emb_model_type': modelTypeOptions.text,
      'runtime.model.audio_emb_model_type': modelTypeOptions.audio,
      'runtime.model.asr_model_name': modelNameOptions.asr,
      'runtime.model.text_emb_model_name': modelNameOptions.text,
      'runtime.model.audio_emb_model_name': modelNameOptions.audio,
    }
    return staticMap[path] ?? null
  }

  function updatePath(path: string, value: unknown) {
    if (!formConfig) {
      return
    }
    setFormConfig(setAtPath(formConfig, path, value))
  }

  async function loadPreset() {
    if (!selectedPreset) {
      return
    }
    const response = await api.createConfig({ preset_name: selectedPreset, auto_devices: true })
    setFormConfig(response.config)
    setMessage(`Loaded preset: ${selectedPreset}`)
    setError('')
  }

  async function validate() {
    if (!formConfig) {
      return
    }
    try {
      await api.validateConfig(formConfig)
      setMessage('Config valid')
      setError('')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  async function previewGraph() {
    if (!formConfig) {
      return
    }
    try {
      const response = await api.graphPreview(formConfig)
      setGraph(JSON.stringify(response, null, 2))
      setError('')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  function renderField(path: string, value: unknown) {
    const optionsForPath = selectOptionsForPath(path)
    if (optionsForPath && optionsForPath.length > 0) {
      return (
        <select value={String(value ?? '')} onChange={(event) => updatePath(path, event.target.value)}>
          <option value="">(none)</option>
          {optionsForPath.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      )
    }
    if (typeof value === 'boolean') {
      return (
        <input
          type="checkbox"
          checked={value}
          onChange={(event) => updatePath(path, event.target.checked)}
        />
      )
    }
    if (typeof value === 'number') {
      return (
        <input
          type="number"
          value={String(value)}
          onChange={(event) => updatePath(path, parseByCurrentType(event.target.value, value))}
        />
      )
    }
    if (Array.isArray(value) || isRecord(value)) {
      return (
        <textarea
          rows={3}
          value={JSON.stringify(value)}
          onChange={(event) => updatePath(path, parseByCurrentType(event.target.value, value))}
        />
      )
    }
    return (
      <input
        type="text"
        value={value === null || value === undefined ? '' : String(value)}
        onChange={(event) => updatePath(path, parseByCurrentType(event.target.value, value))}
      />
    )
  }

  function renderObject(prefix: string, value: JsonRecord) {
    return (
      <div className="form-grid">
        {Object.entries(value).map(([key, itemValue]) => {
          const path = prefix ? `${prefix}.${key}` : key
          if (isRecord(itemValue)) {
            return (
              <details key={path} open>
                <summary>{labelFromPath(path)}</summary>
                {renderObject(path, itemValue)}
              </details>
            )
          }
          return (
            <label key={path} className="field-row">
              <span>{labelFromPath(path)}</span>
              {renderField(path, itemValue)}
            </label>
          )
        })}
      </div>
    )
  }

  return (
    <section className="panel">
      <h2>Config Builder</h2>
      <div className="row">
        <select value={selectedPreset} onChange={(event) => setSelectedPreset(event.target.value)}>
          {(options?.presets || []).map((preset) => (
            <option value={preset} key={preset}>
              {preset}
            </option>
          ))}
        </select>
        <button onClick={() => void loadPreset()}>Load preset</button>
        <button onClick={() => void validate()}>Validate</button>
        <button onClick={() => void previewGraph()}>Preview DAG</button>
      </div>
      {message ? <p className="message">{message}</p> : null}
      {error ? <p className="error">{error}</p> : null}

      {formConfig ? renderObject('', formConfig) : <p>Loading form options...</p>}

      <details>
        <summary>Generated JSON (advanced fallback)</summary>
        <label className="field-row">
          <span>Allow manual JSON edits</span>
          <input type="checkbox" checked={allowJsonEdit} onChange={(event) => setAllowJsonEdit(event.target.checked)} />
        </label>
        <textarea
          value={configText}
          onChange={(event) => {
            setConfigText(event.target.value)
            if (allowJsonEdit) {
              try {
                const parsed = JSON.parse(event.target.value) as JsonRecord
                setFormConfig(parsed)
              } catch {
                // keep text as-is until valid JSON
              }
            }
          }}
          rows={18}
          readOnly={!allowJsonEdit}
        />
      </details>

      <h3>Graph Preview</h3>
      <pre>{graph || 'No graph preview yet.'}</pre>
    </section>
  )
}
