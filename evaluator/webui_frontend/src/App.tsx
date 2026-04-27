import { NavLink, Route, Routes } from 'react-router-dom'
import { useState } from 'react'
import { ConfigPage } from './pages/ConfigPage'
import { RunPage } from './pages/RunPage'
import { ResultsPage } from './pages/ResultsPage'
import { ComparePage } from './pages/ComparePage'
import { LivePage } from './pages/LivePage'

const starterConfig = JSON.stringify(
  {
    experiment_name: 'webui_experiment',
    output_dir: 'evaluation_results/webui',
    model: { pipeline_mode: 'asr_text_retrieval' },
  },
  null,
  2
)

function App() {
  const [configText, setConfigText] = useState(starterConfig)
  const [selectedRunIds, setSelectedRunIds] = useState<number[]>([])

  return (
    <div className="app">
      <header>
        <h1>Evaluator WebUI</h1>
        <nav>
          <NavLink to="/">Config</NavLink>
          <NavLink to="/run">Run</NavLink>
          <NavLink to="/results">Results</NavLink>
          <NavLink to="/compare">Compare</NavLink>
          <NavLink to="/live">Live</NavLink>
        </nav>
      </header>
      <main>
        <Routes>
          <Route path="/" element={<ConfigPage configText={configText} setConfigText={setConfigText} />} />
          <Route path="/run" element={<RunPage configText={configText} />} />
          <Route
            path="/results"
            element={<ResultsPage selectedRunIds={selectedRunIds} setSelectedRunIds={setSelectedRunIds} />}
          />
          <Route path="/compare" element={<ComparePage selectedRunIds={selectedRunIds} />} />
          <Route path="/live" element={<LivePage configText={configText} />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
