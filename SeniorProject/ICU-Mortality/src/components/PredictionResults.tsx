import {
  PieChart, Pie, Cell,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts';
import { PredictionResponse, ShapEntry } from '../App';
import {
  Activity, BedDouble, RefreshCw,
  AlertTriangle, CheckCircle2, Loader2, Brain,
} from 'lucide-react';

interface PredictionResultsProps {
  result: PredictionResponse | null;
  isLoading: boolean;
  error: string | null;
  suggestions: string | null;
  isSuggesting: boolean;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

type RiskLevel = 'LOW' | 'MODERATE' | 'HIGH';

function riskColors(level: RiskLevel | undefined) {
  switch (level) {
    case 'HIGH':     return { fill: '#ef4444', bg: 'bg-red-100',    text: 'text-red-700',   border: 'border-red-200',   card: 'from-red-50 to-red-50/30 border-red-200'       };
    case 'MODERATE': return { fill: '#f59e0b', bg: 'bg-amber-100',  text: 'text-amber-700', border: 'border-amber-200', card: 'from-amber-50 to-amber-50/30 border-amber-200' };
    default:         return { fill: '#22c55e', bg: 'bg-green-100',  text: 'text-green-700', border: 'border-green-200', card: 'from-green-50 to-green-50/30 border-green-200' };
  }
}

const MODEL_LABELS: Record<string, string> = {
  svm_calibrated:      'SVM',
  decision_tree:       'Decision Tree',
  random_forest:       'Random Forest',
  gradient_boosting:   'Grad. Boosting',
  logistic_regression: 'Logistic Reg.',
};

function losCategory(days: number) {
  if (days <= 3)  return { sub: 'Short stay' };
  if (days <= 7)  return { sub: 'Moderate stay' };
  if (days <= 14) return { sub: 'Extended stay' };
  return               { sub: 'Prolonged stay' };
}

// ── Gauge (half-donut) ────────────────────────────────────────────────────────

function GaugeChart({ pct, fill }: { pct: number; fill: string }) {
  const clamped = Math.min(100, Math.max(0, pct));
  return (
    <PieChart width={160} height={100}>
      <Pie
        data={[{ value: clamped }, { value: 100 - clamped }]}
        cx={80} cy={84}
        startAngle={180} endAngle={0}
        innerRadius={52} outerRadius={72}
        paddingAngle={0} dataKey="value"
        strokeWidth={0} isAnimationActive
      >
        <Cell fill={fill} />
        <Cell fill="#e5e7eb" />
      </Pie>
      <text x={80} y={92} textAnchor="middle" dominantBaseline="middle"
        fontSize={20} fontWeight={700} fill={fill}>
        {`${pct.toFixed(1)}%`}
      </text>
    </PieChart>
  );
}

// ── Model breakdown bar chart ─────────────────────────────────────────────────

function ModelBarChart({ probabilities, votes }: {
  probabilities: Record<string, number>;
  votes: Record<string, number>;
}) {
  const data = Object.entries(probabilities).map(([key, prob]) => ({
    name: MODEL_LABELS[key] ?? key,
    pct: parseFloat((prob * 100).toFixed(1)),
    voted: votes[key] === 1,
  }));

  return (
    <ResponsiveContainer width="100%" height={130}>
      <BarChart layout="vertical" data={data}
        margin={{ top: 0, right: 36, bottom: 0, left: 0 }} barSize={10}>
        <XAxis type="number" domain={[0, 100]} hide />
        <YAxis type="category" dataKey="name" width={104}
          tick={{ fontSize: 11, fill: '#6b7280' }} tickLine={false} axisLine={false} />
        <Tooltip formatter={(val: number) => [`${val.toFixed(1)}%`, 'Probability']}
          cursor={{ fill: '#f3f4f6' }} contentStyle={{ fontSize: 12, borderRadius: 8 }} />
        <Bar dataKey="pct" radius={4} isAnimationActive>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.voted ? '#f87171' : '#86efac'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── SHAP bar chart ────────────────────────────────────────────────────────────

function ShapChart({ data }: { data: ShapEntry[] }) {
  if (!data || data.length === 0) return null;

  const chartData = data.map(d => ({
    name: d.display_name || d.feature,
    value: d.value,
  }));

  return (
    <div className="mt-3 pt-3 border-t border-white/60">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1.5">
        Feature Impact (SHAP)
      </p>
      <div className="flex gap-3 mb-1.5">
        <span className="flex items-center gap-1 text-xs text-gray-400">
          <span className="inline-block w-2.5 h-2.5 rounded-sm bg-red-300" /> Increases risk
        </span>
        <span className="flex items-center gap-1 text-xs text-gray-400">
          <span className="inline-block w-2.5 h-2.5 rounded-sm bg-blue-300" /> Decreases risk
        </span>
      </div>
      <ResponsiveContainer width="100%" height={chartData.length * 22 + 10}>
        <BarChart layout="vertical" data={chartData}
          margin={{ top: 0, right: 8, bottom: 0, left: 0 }} barSize={9}>
          <XAxis type="number" domain={['auto', 'auto']} hide />
          <YAxis type="category" dataKey="name" width={140}
            tick={{ fontSize: 10, fill: '#6b7280' }} tickLine={false} axisLine={false} />
          <Tooltip
            formatter={(val: number) => [
              `${val >= 0 ? '+' : ''}${Number(val).toFixed(4)}`,
              val >= 0 ? '↑ Increases risk' : '↓ Decreases risk',
            ]}
            cursor={{ fill: '#f9fafb' }}
            contentStyle={{ fontSize: 11, borderRadius: 8 }}
          />
          <Bar dataKey="value" radius={3} isAnimationActive>
            {chartData.map((entry, i) => (
              <Cell key={i} fill={entry.value >= 0 ? '#f87171' : '#60a5fa'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── LOS legend ────────────────────────────────────────────────────────────────

const LOS_ZONES = [
  { label: '≤ 3 days',  dot: 'bg-green-400', active: (d: number) => d <= 3 },
  { label: '4–7 days',  dot: 'bg-blue-400',  active: (d: number) => d > 3 && d <= 7 },
  { label: '8–14 days', dot: 'bg-amber-400', active: (d: number) => d > 7 && d <= 14 },
  { label: '> 14 days', dot: 'bg-red-400',   active: (d: number) => d > 14 },
];

function LOSLegend({ days }: { days: number }) {
  return (
    <div className="flex flex-col gap-1">
      {LOS_ZONES.map((z) => {
        const on = z.active(days);
        return (
          <div key={z.label} className={`flex items-center gap-1.5 text-xs ${on ? 'font-semibold text-gray-800' : 'text-gray-400'}`}>
            <span className={`w-2 h-2 rounded-full flex-shrink-0 ${z.dot} ${on ? '' : 'opacity-30'}`} />
            <span>{z.label}</span>
            {on && <span className="ml-0.5 text-gray-500 font-normal">← now</span>}
          </div>
        );
      })}
    </div>
  );
}

// ── Risk badge ────────────────────────────────────────────────────────────────

function RiskBadge({ level }: { level: RiskLevel | undefined }) {
  if (!level) return null;
  const c = riskColors(level);
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold border ${c.bg} ${c.text} ${c.border}`}>
      {level}
    </span>
  );
}

// ── Cards ─────────────────────────────────────────────────────────────────────

function MortalityCard({ mortality, shap }: {
  mortality: PredictionResponse['mortality'];
  shap: ShapEntry[];
}) {
  const c = riskColors(mortality.risk_level);
  const pct = mortality.death_percentage;
  const votesYes = Object.values(mortality.votes).filter(Boolean).length;
  const total = Object.keys(mortality.votes).length;

  return (
    <div className={`rounded-xl border p-4 bg-gradient-to-br ${c.card}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg" style={{ background: c.fill }}>
            <Activity className="size-4 text-white" />
          </div>
          <span className="font-semibold text-gray-800 text-sm">In-hospital Mortality</span>
        </div>
        <RiskBadge level={mortality.risk_level} />
      </div>

      <div className="flex items-end gap-4">
        <div className="flex-shrink-0">
          <GaugeChart pct={pct} fill={c.fill} />
        </div>
        <div className="pb-2 flex-1">
          <p className="text-xs text-gray-500 leading-snug">probability of in-hospital death</p>
          <p className="text-xs text-gray-500 mt-1">
            <span className="font-semibold text-gray-700">{votesYes}/{total}</span> models predict death
          </p>
        </div>
      </div>

      <div className="mt-2">
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Model breakdown</p>
        <ModelBarChart probabilities={mortality.model_probabilities} votes={mortality.votes} />
        <div className="flex gap-3 mt-1">
          <span className="flex items-center gap-1 text-xs text-gray-400">
            <span className="inline-block w-2.5 h-2.5 rounded-sm bg-red-300" /> Predicts death
          </span>
          <span className="flex items-center gap-1 text-xs text-gray-400">
            <span className="inline-block w-2.5 h-2.5 rounded-sm bg-green-300" /> Predicts survival
          </span>
        </div>
      </div>

      <ShapChart data={shap} />
    </div>
  );
}

function LOSCard({ los, shap }: {
  los: PredictionResponse['length_of_stay'];
  shap: ShapEntry[];
}) {
  const days = los.predicted_los_days;

  return (
    <div className="rounded-xl border border-blue-200 p-4 bg-gradient-to-br from-blue-50 to-indigo-50/40">
      <div className="flex items-center gap-2 mb-3">
        <div className="p-1.5 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg">
          <BedDouble className="size-4 text-white" />
        </div>
        <span className="font-semibold text-gray-800 text-sm">Length of Stay</span>
      </div>

      {los.error || days === null ? (
        <p className="text-sm text-gray-400 italic">
          {los.error ? `Not available: ${los.error}` : 'LOS model not in bundle'}
        </p>
      ) : (
        <>
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="flex items-baseline gap-1.5">
                <span className="text-4xl font-bold text-blue-600">{days.toFixed(1)}</span>
                <span className="text-lg text-blue-400 font-medium">days</span>
              </div>
              <p className="text-xs text-gray-400 mt-0.5">{losCategory(days).sub}</p>
            </div>
            <LOSLegend days={days} />
          </div>
          <ShapChart data={shap} />
        </>
      )}
    </div>
  );
}

function ReadmissionCard({ ra, shap }: {
  ra: PredictionResponse['readmission_30d'];
  shap: ShapEntry[];
}) {
  const prob = ra.readmission_probability;
  const c = riskColors(ra.risk_level);

  return (
    <div className={`rounded-xl border p-4 bg-gradient-to-br ${c.card}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg" style={{ background: c.fill }}>
            <RefreshCw className="size-4 text-white" />
          </div>
          <span className="font-semibold text-gray-800 text-sm">30-day Readmission</span>
        </div>
        <RiskBadge level={ra.risk_level} />
      </div>

      {ra.error || prob === null ? (
        <p className="text-sm text-gray-400 italic">
          {ra.error ? `Not available: ${ra.error}` : 'Readmission model not in bundle'}
        </p>
      ) : (
        <>
          <div className="flex items-end gap-4">
            <div className="flex-shrink-0">
              <GaugeChart pct={prob * 100} fill={c.fill} />
            </div>
            <div className="pb-2">
              <p className="text-xs text-gray-500 leading-snug">probability of readmission within 30 days</p>
            </div>
          </div>
          <ShapChart data={shap} />
        </>
      )}
    </div>
  );
}

// ── AI Suggestions ────────────────────────────────────────────────────────────

function AISuggestions({ suggestions, isSuggesting }: {
  suggestions: string | null;
  isSuggesting: boolean;
}) {
  if (!isSuggesting && !suggestions) return null;

  const parseBullets = (text: string): string[] | null => {
    const lines = text.split('\n').map(l => l.trim()).filter(Boolean);
    const bullets = lines.filter(l => /^[•\-\*]/.test(l) || /^\d[\.\)]/.test(l));
    if (bullets.length === 0) return null;
    return bullets.map(b => b.replace(/^[•\-\*\d\.\)\s]+/, '').trim());
  };

  return (
    <div className="rounded-xl border border-violet-200 p-4 bg-gradient-to-br from-violet-50 to-purple-50/40">
      <div className="flex items-center gap-2 mb-3">
        <div className="p-1.5 bg-gradient-to-br from-violet-500 to-purple-600 rounded-lg">
          <Brain className="size-4 text-white" />
        </div>
        <span className="font-semibold text-gray-800 text-sm">AI Treatment Suggestions</span>
        <span className="ml-auto text-xs text-gray-400">Gemini 2.5</span>
      </div>

      {isSuggesting ? (
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <Loader2 className="size-4 animate-spin text-violet-400" />
          Generating suggestions…
        </div>
      ) : suggestions?.startsWith('Error:') ? (
        <p className="text-xs text-red-500">{suggestions}</p>
      ) : suggestions ? (
        (() => {
          const bullets = parseBullets(suggestions);
          if (!bullets) {
            return <p className="text-sm text-gray-600 leading-relaxed">{suggestions}</p>;
          }
          return (
            <ul className="space-y-2.5">
              {bullets.map((b, i) => (
                <li key={i} className="flex gap-2.5 text-sm text-gray-700 leading-relaxed">
                  <span className="text-violet-500 font-bold mt-0.5 flex-shrink-0">•</span>
                  <span>{b}</span>
                </li>
              ))}
            </ul>
          );
        })()
      ) : null}
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────

export function PredictionResults({ result, isLoading, error, suggestions, isSuggesting }: PredictionResultsProps) {
  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-400 gap-4">
        <Loader2 className="size-12 animate-spin text-indigo-400" />
        <p className="text-base">Running prediction…</p>
        <p className="text-sm">Contacting Lambda endpoint</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center px-4">
        <div className="p-4 bg-red-50 rounded-2xl border border-red-200">
          <AlertTriangle className="size-10 text-red-400 mx-auto mb-2" />
          <p className="text-sm font-semibold text-red-600 mb-1">Prediction failed</p>
          <p className="text-xs text-red-500 font-mono break-all">{error}</p>
        </div>
        <p className="text-xs text-gray-400">Check that the Lambda is deployed and the API key is correct.</p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-400 gap-3">
        <div className="relative mb-2">
          <div className="absolute inset-0 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full blur-xl opacity-20" />
          <Activity className="relative size-20 opacity-20" />
        </div>
        <p className="text-lg text-gray-500">No results yet</p>
        <p className="text-sm text-gray-400 text-center max-w-xs">
          Fill in patient details on the left and click <strong>Run Prediction</strong>
        </p>
      </div>
    );
  }

  const shap = result.shap ?? { mortality: [], los: [], readmission: [] };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 text-xs text-gray-500">
          <CheckCircle2 className="size-3.5 text-green-500" />
          Prediction complete
        </div>
        <span className="text-xs text-gray-400 font-mono">v {result.model_version.slice(0, 10)}</span>
      </div>

      <MortalityCard mortality={result.mortality} shap={shap.mortality} />
      <LOSCard los={result.length_of_stay} shap={shap.los} />
      <ReadmissionCard ra={result.readmission_30d} shap={shap.readmission} />

      <AISuggestions suggestions={suggestions} isSuggesting={isSuggesting} />

      <div className="rounded-lg bg-gray-50 border border-gray-200 p-3 text-xs text-gray-500">
        <strong className="text-gray-600">Research use only.</strong> Trained on MIMIC-IV data.
        Not validated for clinical decision-making.
      </div>
    </div>
  );
}
