import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

function humanizeFeature(feature) {
  const known = {
    elo_diff: "ELO advantage",
    reach_diff: "Reach advantage",
    height_diff: "Height advantage",
    age_diff: "Age difference",
    diff_win_streak: "Win streak edge",
    weight_class_kg: "Weight class",
  };
  if (known[feature]) return known[feature];
  return feature
    .replace(/^diff_avg_/, "")
    .replace(/^a_avg_/, "A ")
    .replace(/^b_avg_/, "B ")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export default function ShapExplainer({ features = [] }) {
  const data = features.map((item) => ({
    feature: humanizeFeature(item.feature),
    impact: Number(item.impact),
  }));

  if (!data.length) {
    return <div className="empty-panel">SHAP unavailable</div>;
  }

  return (
    <section className="shap-panel">
      <h2>Key Factors</h2>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 96 }}>
          <XAxis type="number" tick={{ fill: "#cfcfcf" }} />
          <YAxis dataKey="feature" type="category" width={132} tick={{ fill: "#f5f5f5", fontSize: 12 }} />
          <Tooltip
            cursor={{ fill: "rgba(255,255,255,0.06)" }}
            contentStyle={{ background: "#181818", border: "1px solid #333", color: "#f5f5f5" }}
          />
          <Bar dataKey="impact" radius={[4, 4, 4, 4]}>
            {data.map((entry) => (
              <Cell key={entry.feature} fill={entry.impact >= 0 ? "#2e9f5b" : "#D4462C"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </section>
  );
}
