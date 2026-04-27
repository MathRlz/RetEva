import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'

type Props = {
  configText: string
}

export function RunPage({ configText }: Props) {
  const [jobs, setJobs] = useState<Array<Record<string, unknown>>>([])
  const [selectedJobId, setSelectedJobId] = useState('')
  const [details, setDetails] = useState('')
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  const refreshJobs = useCallback(async () => {
    try {
      const response = await api.listJobs()
      const refreshed = response.jobs || []
      setJobs(refreshed)
      if (selectedJobId) {
        const selected = refreshed.find((job) => String(job.job_id ?? '') === selectedJobId)
        if (selected && String(selected.status ?? '') === 'failed') {
          const failureText = String(selected.error ?? 'Job failed with unknown error')
          setError(failureText)
          setDetails(JSON.stringify(selected, null, 2))
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [selectedJobId])

  useEffect(() => {
    const bootstrapTimer = setTimeout(() => {
      void refreshJobs()
    }, 0)
    const intervalId = setInterval(() => {
      void refreshJobs()
    }, 3000)
    return () => {
      clearTimeout(bootstrapTimer)
      clearInterval(intervalId)
    }
  }, [refreshJobs])

  async function submitEvaluation() {
    try {
      const config = JSON.parse(configText) as Record<string, unknown>
      const response = await api.submitEvaluation(config)
      setSelectedJobId(response.job_id)
      setMessage(`Submitted job ${response.job_id}`)
      setError('')
      await refreshJobs()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  async function loadJobDetails(kind: 'status' | 'result' | 'metadata' | 'artifacts') {
    if (!selectedJobId) {
      return
    }
    try {
      let payload: unknown
      if (kind === 'status') payload = await api.getJob(selectedJobId)
      if (kind === 'result') payload = await api.getJobResult(selectedJobId)
      if (kind === 'metadata') payload = await api.getJobMetadata(selectedJobId)
      if (kind === 'artifacts') payload = await api.getJobArtifacts(selectedJobId)
      setDetails(JSON.stringify(payload, null, 2))
      setError('')
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : String(err)
      setError(errorMessage)
    }
  }

  async function cancelJob() {
    if (!selectedJobId) {
      return
    }
    try {
      await api.cancelJob(selectedJobId)
      setMessage(`Cancel requested for ${selectedJobId}`)
      setError('')
      await refreshJobs()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <section className="panel">
      <h2>Run Control</h2>
      <div className="row">
        <button onClick={() => void submitEvaluation()}>Run experiment</button>
        <button onClick={() => void refreshJobs()}>Refresh jobs</button>
        <button onClick={() => void cancelJob()}>Cancel selected</button>
      </div>
      {message ? <p className="message">{message}</p> : null}
      {error ? <p className="error">{error}</p> : null}
      <h3>Jobs</h3>
      <table>
        <thead>
          <tr>
            <th>Job ID</th>
            <th>Type</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => {
            const id = String(job.job_id ?? '')
            return (
              <tr key={id} onClick={() => setSelectedJobId(id)} className={selectedJobId === id ? 'selected' : ''}>
                <td>{id}</td>
                <td>{String(job.job_type ?? '')}</td>
                <td>{String(job.status ?? '')}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p>Selected: {selectedJobId || 'none'}</p>
      <div className="row">
        <button onClick={() => void loadJobDetails('status')}>Status</button>
        <button onClick={() => void loadJobDetails('result')}>Result</button>
        <button onClick={() => void loadJobDetails('metadata')}>Metadata</button>
        <button onClick={() => void loadJobDetails('artifacts')}>Artifacts</button>
        <button onClick={() => void loadJobDetails('status')}>Failure details</button>
      </div>
      <pre>{details || 'No details loaded.'}</pre>
    </section>
  )
}
