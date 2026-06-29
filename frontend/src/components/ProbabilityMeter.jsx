export default function ProbabilityMeter({ probA, nameA, nameB, confidence }) {
  const pctA = Math.round((Number(probA) || 0) * 100);
  const winner =
    probA > 0.6 ? { name: nameA, className: "winner" } : probA < 0.4 ? { name: nameB, className: "winner" } : null;

  return (
    <section className="probability">
      <div className="probability-top">
        <div>
          <div className="meter-number">{pctA}%</div>
          <div className="meter-label">{nameA}</div>
        </div>
        <span className={`confidence ${confidence || "low"}`}>{confidence || "low"}</span>
      </div>
      <div className="meter-track">
        <div className="meter-red" style={{ width: `${pctA}%` }} />
        <div className="meter-blue" style={{ width: `${100 - pctA}%` }} />
      </div>
      <div className={winner ? winner.className : "too-close"}>{winner ? winner.name : "Too close to call"}</div>
      <div className="meter-names">
        <span>{nameA}</span>
        <span>{nameB}</span>
      </div>
    </section>
  );
}
