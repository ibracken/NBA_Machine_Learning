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

`lineup-optimizer` (API mode) returns the lineup the nightly email uses: the learned-minutes lineup, or DFF's while
in-house projections are paused. It reads both files (`API_LINEUP_SOURCES` in the Lambda) and returns the most recent slate,
preferring in-house on a tie. The response says which one (`source`) and for which slate (`slate_date`). The page warns
when that isn't today; outside game days it shows the last slate played. Actual FP appears once games are graded.

The Lambda runs at 1769 MB (one full vCPU). At 512 MB a cold start exceeded Lambda's 10 s init limit and took about
19 s end to end, close to API Gateway's 30 s limit; at 1769 MB it takes about 2 s. deploy.py doesn't set memory.

The API response is normalized to `PREDICTED_FP`/`GAME_DATE` and the UI computes value (FP per $1k) and salary usage against the $50,000 cap. `src/api.ts` unwraps either a raw JSON body or a Lambda-proxy `{statusCode, body}` envelope.

## Response shape

```json
{
  "success": true,
  "source": "Learned minutes",
  "slate_date": "2026-01-17",
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
