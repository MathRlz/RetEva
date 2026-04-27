import { useState } from 'react'
import { api } from '../api'

type Props = {
  configText: string
}

export function LivePage({ configText }: Props) {
  const [queryText, setQueryText] = useState('')
  const [k, setK] = useState(5)
  const [resultText, setResultText] = useState('')
  const [error, setError] = useState('')

  async function runLiveQuery() {
    try {
      const config = JSON.parse(configText) as Record<string, unknown>
      const response = await api.liveQuery(config, queryText, k)
      setResultText(JSON.stringify(response, null, 2))
      setError('')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <section className="panel">
      <h2>Live Test</h2>
      <div className="row">
        <label>
          Top K:
          <input type="number" min={1} value={k} onChange={(event) => setK(Number(event.target.value))} />
        </label>
      </div>
      <textarea
        rows={6}
        value={queryText}
        onChange={(event) => setQueryText(event.target.value)}
        placeholder="Ask question for retrieval..."
      />
      <div className="row">
        <button onClick={() => void runLiveQuery()}>Run live query</button>
      </div>
      {error ? <p className="error">{error}</p> : null}
      <pre>{resultText || 'No live query yet.'}</pre>
    </section>
  )
}

