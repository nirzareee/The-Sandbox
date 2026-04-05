import { getPlayerName } from '../utils';

export default function GameOver({ finalScores, players }) {
  const scores = finalScores?.final_scores || {};
  const sorted = Object.entries(scores).sort(([, a], [, b]) => b - a);

  return (
    <div className="game-over-screen">
      <div className="game-over-container">
        <h1 className="game-over-title">Game Over</h1>
        <p>Thanks for playing The Sandbox!</p>

        {sorted.length > 0 && (
          <table className="leaderboard-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Player</th>
                <th>Score</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map(([pid, score], i) => {
                const rankClass = i === 0 ? 'gold' : i === 1 ? 'silver' : i === 2 ? 'bronze' : '';
                return (
                  <tr key={pid} className={rankClass || undefined}>
                    <td>{i + 1}</td>
                    <td>{getPlayerName(players, pid)}</td>
                    <td>{score}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}

        <button className="btn-play-again" onClick={() => window.location.reload()}>
          Play Again
        </button>
      </div>
    </div>
  );
}
