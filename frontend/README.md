# Lineup Lab (frontend)

React + Vite viewer for the DraftKings lineup served by the `lineup-optimizer` Lambda through API Gateway.

## Run

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check + production bundle in dist/
```

## Configuration

Set in `frontend/.env.local` (not committed) or the shell:

| Variable | Default | Purpose |
|---|---|---|
| `VITE_LINEUP_API_ENDPOINT` | `https://hhw9yz6ar2.execute-api.us-east-1.amazonaws.com/lineup` | API Gateway route → `lineup-optimizer` |
| `VITE_USE_MOCK_DATA` | unset | `true` renders the built-in sample lineup without calling the API |

## What it shows

`lineup-optimizer` (API mode) reads the first non-empty file in this order and returns the latest date in it:

1. `model_comparison/complex_position_overlap/fp_current/daily_lineups.parquet`
2. `model_comparison/complex_position_overlap/daily_lineups.parquet`
3. `data/daily_lineups/current.parquet`

The API response is normalized to `PREDICTED_FP`/`GAME_DATE` and the UI computes value (FP per $1k) and salary usage against the $50,000 cap. `src/api.ts` unwraps either a raw JSON body or a Lambda-proxy `{statusCode, body}` envelope.

## Response shape

```json
{
  "success": true,
  "lineup_size": 8,
  "total_salary": 48500,
  "remaining_salary": 1500,
  "total_predicted_fp": 312.5,
  "players": [
    { "SLOT": "PG", "PLAYER": "…", "POSITION": "PG/SG", "SALARY": 9000,
      "PREDICTED_FP": 45.5, "PROJECTED_MIN": 34.0, "GAME_DATE": "2026-01-17" }
  ]
}
```

## Layout

```
frontend/
├── index.html          # Vite entry
├── src/
│   ├── main.tsx        # React root
│   ├── App.tsx         # page, summary metrics, roster table
│   ├── api.ts          # fetch + response validation, mock data
│   ├── format.ts       # currency/number/date formatters
│   ├── types.ts        # LineupPlayer / LineupResponse
│   └── styles.css
├── package.json
├── tsconfig.json
└── vite.config.ts
```
