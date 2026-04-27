import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'

type Props = {
  selectedRunIds: number[]
  setSelectedRunIds: (ids: number[]) => void
}

export function ResultsPage({ selectedRunIds, setSelectedRunIds }: Props) {
  const [rows, setRows] = useState<Array<Record<string, unknown>>>([])
  const [runs, setRuns] = useState<Array<Record<string, unknown>>>([])
  const [metric, setMetric] = useState('MRR')

  const refresh = useCallback(async () => {
    const [leaderboard, runList] = await Promise.all([api.leaderboard(metric), api.runs()])
    setRows(leaderboard.rows || [])
    setRuns(runList.runs || [])
  }, [metric])

  useEffect(() => {
    const timer = setTimeout(() => {
      void refresh()
    }, 0)
    return () => clearTimeout(timer)
  }, [refresh])

  function toggleRun(runId: number) {
    if (selectedRunIds.includes(runId)) {
      setSelectedRunIds(selectedRunIds.filter((id) => id !== runId))
      return
    }
    setSelectedRunIds([...selectedRunIds, runId])
  }

  return (
    <section className="panel">
      <h2>Results Browser</h2>
      <div className="row">
        <label>
          Metric:
          <input value={metric} onChange={(event) => setMetric(event.target.value)} />
        </label>
        <button onClick={() => void refresh()}>Refresh</button>
      </div>
      <h3>Leaderboard</h3>
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Experiment</th>
            <th>Dataset</th>
            <th>Mode</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={String(row.run_id)}>
              <td>{String(row.run_id)}</td>
              <td>{String(row.experiment_name ?? '')}</td>
              <td>{String(row.dataset_name ?? '')}</td>
              <td>{String(row.pipeline_mode ?? '')}</td>
              <td>{String(row.metric_value ?? '')}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <h3>Runs (select for compare)</h3>
      <table>
        <thead>
          <tr>
            <th>Select</th>
            <th>Run</th>
            <th>Experiment</th>
            <th>Dataset</th>
            <th>Mode</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => {
            const id = Number(run.run_id ?? 0)
            const checked = selectedRunIds.includes(id)
            return (
              <tr key={id}>
                <td>
                  <input type="checkbox" checked={checked} onChange={() => toggleRun(id)} />
                </td>
                <td>{id}</td>
                <td>{String(run.experiment_name ?? '')}</td>
                <td>{String(run.dataset_name ?? '')}</td>
                <td>{String(run.pipeline_mode ?? '')}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </section>
  )
}
