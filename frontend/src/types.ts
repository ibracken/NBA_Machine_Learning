export type LineupPlayer = {
  SLOT: string;
  PLAYER: string;
  POSITION: string;
  SALARY: number;
  PREDICTED_FP: number;
  PROJECTED_MIN?: number;
  PREV_FP?: number;
  PREV_MIN?: number;
  SEASON_AVG_FP?: number;
  SEASON_AVG_MIN?: number;
  GAME_DATE?: string;
};

export type LineupResponse = {
  success?: boolean;
  error?: string;
  lineup_size?: number;
  total_salary: number;
  remaining_salary: number;
  total_predicted_fp: number;
  players: LineupPlayer[];
};
