import type { LineupResponse } from './types';

const DEFAULT_API_ENDPOINT = 'https://hhw9yz6ar2.execute-api.us-east-1.amazonaws.com/lineup';
const API_ENDPOINT = import.meta.env.VITE_LINEUP_API_ENDPOINT ?? DEFAULT_API_ENDPOINT;
const USE_MOCK_DATA = import.meta.env.VITE_USE_MOCK_DATA === 'true';

const MOCK_DATA: LineupResponse = {
  success: true,
  lineup_size: 8,
  total_salary: 48500,
  remaining_salary: 1500,
  total_predicted_fp: 312.5,
  players: [
    { SLOT: 'PG', PLAYER: 'Luka Doncic', POSITION: 'PG/SG', SALARY: 11200, PREDICTED_FP: 58.3, PROJECTED_MIN: 36, PREV_FP: 62.1, PREV_MIN: 36, SEASON_AVG_FP: 54.2, SEASON_AVG_MIN: 35, GAME_DATE: '2025-11-17' },
    { SLOT: 'SG', PLAYER: 'Devin Booker', POSITION: 'SG', SALARY: 8900, PREDICTED_FP: 42.1, PROJECTED_MIN: 34, PREV_FP: 38.5, PREV_MIN: 34, SEASON_AVG_FP: 40.8, SEASON_AVG_MIN: 33, GAME_DATE: '2025-11-17' },
    { SLOT: 'SF', PLAYER: 'Jayson Tatum', POSITION: 'SF/PF', SALARY: 9800, PREDICTED_FP: 48.7, PROJECTED_MIN: 37, PREV_FP: 51.2, PREV_MIN: 37, SEASON_AVG_FP: 46.5, SEASON_AVG_MIN: 36, GAME_DATE: '2025-11-17' },
    { SLOT: 'PF', PLAYER: 'Giannis Antetokounmpo', POSITION: 'PF/C', SALARY: 11500, PREDICTED_FP: 62.4, PROJECTED_MIN: 35, PREV_FP: 58.9, PREV_MIN: 33, SEASON_AVG_FP: 60.1, SEASON_AVG_MIN: 34, GAME_DATE: '2025-11-17' },
    { SLOT: 'C', PLAYER: 'Joel Embiid', POSITION: 'C', SALARY: 10800, PREDICTED_FP: 56.8, PROJECTED_MIN: 34, PREV_FP: 53.4, PREV_MIN: 32, SEASON_AVG_FP: 55.7, SEASON_AVG_MIN: 33, GAME_DATE: '2025-11-17' },
    { SLOT: 'G', PLAYER: 'Damian Lillard', POSITION: 'PG', SALARY: 8700, PREDICTED_FP: 41.2, PROJECTED_MIN: 35, PREV_FP: 44.6, PREV_MIN: 35, SEASON_AVG_FP: 42.3, SEASON_AVG_MIN: 34, GAME_DATE: '2025-11-17' },
    { SLOT: 'F', PLAYER: 'Kevin Durant', POSITION: 'SF/PF', SALARY: 9600, PREDICTED_FP: 52.3, PROJECTED_MIN: 36, PREV_FP: 49.8, PREV_MIN: 36, SEASON_AVG_FP: 51.5, SEASON_AVG_MIN: 35, GAME_DATE: '2025-11-17' },
    { SLOT: 'UTIL', PLAYER: 'Anthony Edwards', POSITION: 'SG/SF', SALARY: 8000, PREDICTED_FP: 38.7, PROJECTED_MIN: 35, PREV_FP: 41.2, PREV_MIN: 37, SEASON_AVG_FP: 39.4, SEASON_AVG_MIN: 36, GAME_DATE: '2025-11-17' },
  ],
};

export async function fetchLineup(): Promise<LineupResponse> {
  if (USE_MOCK_DATA) {
    await new Promise((resolve) => window.setTimeout(resolve, 500));
    return MOCK_DATA;
  }

  const response = await fetch(API_ENDPOINT);
  if (!response.ok) {
    const details = await response.text();
    const suffix = details ? `: ${details.slice(0, 180)}` : '';
    throw new Error(`Lineup API returned ${response.status}${suffix}`);
  }

  const result: unknown = await response.json();
  const body = unwrapLambdaBody(result);

  if (!isLineupResponse(body)) {
    throw new Error('Lineup API returned an unexpected response shape');
  }

  if (body.success === false) {
    throw new Error(body.error ?? 'No lineup data available');
  }

  return body;
}

function unwrapLambdaBody(result: unknown): unknown {
  if (typeof result !== 'object' || result === null || !('body' in result)) {
    return result;
  }

  const body = (result as { body: unknown }).body;
  return typeof body === 'string' ? JSON.parse(body) : body;
}

function isLineupResponse(value: unknown): value is LineupResponse {
  if (typeof value !== 'object' || value === null) {
    return false;
  }

  const lineup = value as Partial<LineupResponse>;
  return Array.isArray(lineup.players)
    && typeof lineup.total_salary === 'number'
    && typeof lineup.remaining_salary === 'number'
    && typeof lineup.total_predicted_fp === 'number';
}
