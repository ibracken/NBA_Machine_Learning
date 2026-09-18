import { useCallback, useEffect, useMemo, useState } from 'react';
import { fetchLineup } from './api';
import { formatCurrency, formatDateTime, formatNumber } from './format';
import type { LineupPlayer, LineupResponse } from './types';

const SALARY_CAP = 50000;

type LoadState = 'idle' | 'loading' | 'ready' | 'error';

export function App() {
  const [lineup, setLineup] = useState<LineupResponse | null>(null);
  const [status, setStatus] = useState<LoadState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const loadLineup = useCallback(async () => {
    setStatus('loading');
    setError(null);

    try {
      const data = await fetchLineup();
      setLineup(data);
      setLastUpdated(new Date());
      setStatus('ready');
    } catch (loadError) {
      setStatus('error');
      setError(loadError instanceof Error ? loadError.message : 'Unable to load lineup');
    }
  }, []);

  useEffect(() => {
    void loadLineup();
  }, [loadLineup]);

  const diagnostics = useMemo(() => {
    if (!lineup?.players.length) {
      return null;
    }

    const players = lineup.players;
    const avgValue = lineup.total_predicted_fp / (lineup.total_salary / 1000);
    const salaryUsedPct = Math.min(100, Math.round((lineup.total_salary / SALARY_CAP) * 100));
    const slateDate = players.find((player) => player.GAME_DATE)?.GAME_DATE;

    return { avgValue, salaryUsedPct, slateDate };
  }, [lineup]);

  return (
    <main className="app-shell">
      <header className="page-header">
        <div>
          <h1>Lineup Lab</h1>
          <p>NBA DraftKings optimizer</p>
        </div>
        <div className="header-actions">
          <StatusPill status={status} />
          <button type="button" onClick={loadLineup} disabled={status === 'loading'}>
            {status === 'loading' ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
      </header>

      <section className="summary-grid" aria-label="Lineup summary">
        <Metric label="Projected FP" value={lineup ? formatNumber.format(lineup.total_predicted_fp) : '-'} />
        <Metric label="Salary Used" value={lineup ? formatCurrency.format(lineup.total_salary) : '-'} />
        <Metric label="Remaining" value={lineup ? formatCurrency.format(lineup.remaining_salary) : '-'} />
        <Metric label="Avg Value" value={diagnostics ? `${formatNumber.format(diagnostics.avgValue)}x` : '-'} />
      </section>

      <section className="lineup-card" aria-label="Optimal lineup">
        <div className="card-header">
          <div>
            <h2>Optimal Lineup</h2>
            <p>{diagnostics?.slateDate ?? 'Slate date unavailable'}</p>
          </div>
          {lastUpdated ? <span>Updated {formatDateTime(lastUpdated)}</span> : null}
        </div>

        {diagnostics ? (
          <div className="salary-bar" aria-label={`Salary used ${diagnostics.salaryUsedPct}%`}>
            <span style={{ width: `${diagnostics.salaryUsedPct}%` }} />
          </div>
        ) : null}

        {status === 'loading' ? <LoadingState /> : null}
        {status === 'error' ? <ErrorState message={error ?? 'Unable to load lineup'} /> : null}
        {status === 'ready' ? (lineup ? <RosterTable players={lineup.players} /> : <EmptyState />) : null}
      </section>
    </main>
  );
}

function RosterTable({ players }: { players: LineupPlayer[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Slot</th>
            <th>Player</th>
            <th>Pos</th>
            <th>Salary</th>
            <th>Proj FP</th>
            <th>Value</th>
            <th>Min</th>
            <th>Avg FP</th>
          </tr>
        </thead>
        <tbody>
          {players.map((player) => (
            <tr key={`${player.SLOT}-${player.PLAYER}`}>
              <td><span className="slot">{player.SLOT}</span></td>
              <td className="player-name">{player.PLAYER}</td>
              <td>{player.POSITION}</td>
              <td>{formatCurrency.format(player.SALARY)}</td>
              <td className="strong">{formatNumber.format(player.PREDICTED_FP)}</td>
              <td>{formatNumber.format(valueScore(player))}x</td>
              <td>{formatNumber.format(player.PROJECTED_MIN ?? 0)}</td>
              <td>{formatNumber.format(player.SEASON_AVG_FP ?? 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatusPill({ status }: { status: LoadState }) {
  const text = status === 'loading' ? 'Loading' : status === 'error' ? 'API Error' : status === 'ready' ? 'Loaded' : 'Idle';
  return <span className={`status-pill ${status}`}>{text}</span>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function LoadingState() {
  return <div className="state-card">Loading optimal lineup...</div>;
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="state-card error">
      <strong>Could not load lineup.</strong>
      <p>{message}</p>
      <p>A 500 response means the request reached the API, but the Lambda/API Gateway side failed.</p>
    </div>
  );
}

function EmptyState() {
  return <div className="state-card">No lineup returned.</div>;
}

function valueScore(player: LineupPlayer) {
  return player.SALARY > 0 ? player.PREDICTED_FP / (player.SALARY / 1000) : 0;
}
