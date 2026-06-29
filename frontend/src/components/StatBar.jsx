function formatValue(value, unit = "") {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  const decimals = Math.abs(number) < 10 && !Number.isInteger(number) ? 2 : 1;
  return `${number.toFixed(decimals)}${unit}`;
}

export default function StatBar({ label, valueA, valueB, unit }) {
  const a = Number(valueA);
  const b = Number(valueB);
  const safeA = Number.isFinite(a) ? Math.max(a, 0) : 0;
  const safeB = Number.isFinite(b) ? Math.max(b, 0) : 0;
  const total = safeA + safeB;
  const aShare = total > 0 ? (safeA / total) * 100 : 50;

  return (
    <div className="stat-row">
      <div className="stat-value left">{formatValue(valueA, unit)}</div>
      <div className="stat-mid">
        <div className="stat-label">{label}</div>
        <div className="stat-track" aria-hidden="true">
          <div className="stat-fill-a" style={{ width: `${aShare}%` }} />
          <div className="stat-fill-b" style={{ width: `${100 - aShare}%` }} />
        </div>
      </div>
      <div className="stat-value right">{formatValue(valueB, unit)}</div>
    </div>
  );
}
