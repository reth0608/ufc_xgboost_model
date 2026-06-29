import { Loader2, RotateCcw, Swords } from "lucide-react";
import { useMemo, useState } from "react";
import { getFighterByRef, getFighterStats, predictFight, predictFightByRef } from "./api";
import FighterSearch from "./components/FighterSearch";
import ProbabilityMeter from "./components/ProbabilityMeter";
import ShapExplainer from "./components/ShapExplainer";
import StatBar from "./components/StatBar";

function statValue(payload, key) {
  const value = payload?.stats?.[key];
  return value === null || value === undefined ? Number.NaN : Number(value);
}

export default function App() {
  const [fighterA, setFighterA] = useState(null);
  const [fighterB, setFighterB] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [statsA, setStatsA] = useState(null);
  const [statsB, setStatsB] = useState(null);
  const [refA, setRefA] = useState("");
  const [refB, setRefB] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const refsReady = refA.trim() && refB.trim() && refA.trim() !== refB.trim();
  const fightersReady = fighterA && fighterB && fighterA.fighter_id !== fighterB.fighter_id;
  const canPredict = (refsReady || fightersReady) && !loading;
  const statRows = useMemo(
    () => [
      ["ELO", statValue(statsA, "pre_fight_elo"), statValue(statsB, "pre_fight_elo"), ""],
      ["Reach", statValue(statsA, "reach_cm"), statValue(statsB, "reach_cm"), " cm"],
      ["Sig Strike %", statValue(statsA, "avg_sig_str_pct_5f") * 100, statValue(statsB, "avg_sig_str_pct_5f") * 100, "%"],
      ["Sig Defense", statValue(statsA, "avg_sig_str_def_5f") * 100, statValue(statsB, "avg_sig_str_def_5f") * 100, "%"],
      ["Sig Absorbed", statValue(statsA, "avg_sig_str_absorbed_5f"), statValue(statsB, "avg_sig_str_absorbed_5f"), ""],
      ["Takedown %", statValue(statsA, "avg_td_pct_5f") * 100, statValue(statsB, "avg_td_pct_5f") * 100, "%"],
      ["TD Defense", statValue(statsA, "avg_td_def_5f") * 100, statValue(statsB, "avg_td_def_5f") * 100, "%"],
      ["Striking Score", statValue(statsA, "striking_score"), statValue(statsB, "striking_score"), ""],
      ["Grappling Score", statValue(statsA, "grappling_score"), statValue(statsB, "grappling_score"), ""],
      ["Win Streak", statValue(statsA, "win_streak"), statValue(statsB, "win_streak"), ""],
    ],
    [statsA, statsB],
  );

  async function runPrediction() {
    setLoading(true);
    setError("");
    setPrediction(null);
    try {
      const usingRefs = Boolean(refsReady);
      const [resolvedA, resolvedB] = usingRefs
        ? await Promise.all([getFighterByRef(refA.trim()), getFighterByRef(refB.trim())])
        : [fighterA, fighterB];
      const [result, aStats, bStats] = await Promise.all([
        usingRefs ? predictFightByRef(refA.trim(), refB.trim()) : predictFight(resolvedA.fighter_id, resolvedB.fighter_id),
        getFighterStats(usingRefs ? refA.trim() : resolvedA.fighter_id),
        getFighterStats(usingRefs ? refB.trim() : resolvedB.fighter_id),
      ]);
      setFighterA(resolvedA);
      setFighterB(resolvedB);
      setPrediction(result);
      setStatsA(aStats);
      setStatsB(bStats);
    } catch (err) {
      setError(err.message || "Prediction failed");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFighterA(null);
    setFighterB(null);
    setPrediction(null);
    setStatsA(null);
    setStatsB(null);
    setRefA("");
    setRefB("");
    setError("");
  }

  function selectFighterA(fighter) {
    setFighterA(fighter);
    if (fighter) setRefA("");
  }

  function selectFighterB(fighter) {
    setFighterB(fighter);
    if (fighter) setRefB("");
  }

  return (
    <main className="app-shell">
      <section className="predictor-card">
        <header className="app-header">
          <div>
            <p>UFC Fight Predictor</p>
            <h1>Matchup Forecast</h1>
          </div>
          <button type="button" className="reset-button" onClick={reset} title="Reset">
            <RotateCcw size={18} aria-hidden="true" />
            <span>Reset</span>
          </button>
        </header>

        <div className="search-grid">
          <FighterSearch label="Fighter A" value={fighterA} onSelect={selectFighterA} />
          <FighterSearch label="Fighter B" value={fighterB} onSelect={selectFighterB} />
        </div>

        <div className="ref-grid">
          <label className="ref-field">
            <span>Fighter A Ref No</span>
            <input
              type="number"
              min="1"
              value={refA}
              onChange={(event) => setRefA(event.target.value)}
              placeholder="Ref no"
            />
          </label>
          <label className="ref-field">
            <span>Fighter B Ref No</span>
            <input
              type="number"
              min="1"
              value={refB}
              onChange={(event) => setRefB(event.target.value)}
              placeholder="Ref no"
            />
          </label>
        </div>

        <button type="button" className="predict-button" disabled={!canPredict} onClick={runPrediction}>
          {loading ? <Loader2 className="spin" size={20} aria-hidden="true" /> : <Swords size={20} aria-hidden="true" />}
          <span>{loading ? "Predicting" : "Predict"}</span>
        </button>

        {error && <div className="error-box">{error}</div>}

        {prediction && (
          <div className="results">
            <ProbabilityMeter
              probA={prediction.fighter_a_win_prob}
              nameA={fighterA.name}
              nameB={fighterB.name}
              confidence={prediction.confidence}
            />
            <section className="stat-panel">
              {statRows.map(([label, valueA, valueB, unit]) => (
                <StatBar key={label} label={label} valueA={valueA} valueB={valueB} unit={unit} />
              ))}
            </section>
            <ShapExplainer features={prediction.top_shap_features} />
          </div>
        )}
      </section>
    </main>
  );
}
