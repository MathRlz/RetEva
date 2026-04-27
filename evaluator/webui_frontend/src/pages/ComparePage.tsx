import { useEffect, useState } from 'react'
import { api } from '../api'

type Props = {
  selectedRunIds: number[]
}

export function ComparePage({ selectedRunIds }: Props) {
  const [runs, setRuns] = useState<Array<Record<string, unknown>>>([])
  const [error, setError] = useState('')

  useEffect(() => {
    if (selectedRunIds.length === 0) {
      const timer = setTimeout(() => {
        setRuns([])
      }, 0)
      return () => clearTimeout(timer)
    }
    void (async () => {
      try {
        const loaded = await Promise.all(selectedRunIds.map((id) => api.runDetails(id)))
        setRuns(loaded)
        setError('')
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      }
    })()
  }, [selectedRunIds])

  const metricKeys = Array.from(
    new Set(
      runs.flatMap((run) =>
        Object.keys((run.metrics as Record<string, unknown> | undefined) ?? {})
      )
    )
  )

  return (
    <section className="panel">
      <h2>Compare Runs</h2>
      {selectedRunIds.length === 0 ? <p>Select runs in Results page.</p> : null}
      {error ? <p className="error">{error}</p> : null}
      {runs.length > 0 ? (
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              {runs.map((run) => (
                <th key={String(run.run_id)}>Run {String(run.run_id)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metricKeys.map((metric) => (
              <tr key={metric}>
                <td>{metric}</td>
                {runs.map((run) => {
                  const metrics = (run.metrics as Record<string, unknown> | undefined) ?? {}
                  return <td key={`${String(run.run_id)}-${metric}`}>{String(metrics[metric] ?? '')}</td>
                })}
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </section>
  )
}
