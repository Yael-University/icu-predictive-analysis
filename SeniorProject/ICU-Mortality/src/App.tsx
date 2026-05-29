import { useState } from 'react';
import { PatientDataForm } from './components/PatientDataForm';
import { PredictionResults } from './components/PredictionResults';
import { Activity } from 'lucide-react';
import { Toaster } from './components/ui/sonner';
import { toast } from 'sonner';
import { LAMBDA_URL, LAMBDA_API_KEY } from './config';

export interface MortalityResult {
  predicted_class: number;
  death_probability: number;
  death_percentage: number;
  risk_level: 'LOW' | 'MODERATE' | 'HIGH';
  vote_fraction: number;
  votes: Record<string, number>;
  model_probabilities: Record<string, number>;
}

export interface LOSResult {
  predicted_los_days: number | null;
  error?: string;
}

export interface ReadmissionResult {
  predicted_class?: number;
  readmission_probability: number | null;
  readmission_percentage?: number;
  risk_level?: 'LOW' | 'MODERATE' | 'HIGH';
  error?: string;
}

export interface ShapEntry {
  feature: string;
  display_name: string;
  value: number;
}

export interface ShapResult {
  mortality: ShapEntry[];
  los: ShapEntry[];
  readmission: ShapEntry[];
}

export interface PredictionResponse {
  mortality: MortalityResult;
  length_of_stay: LOSResult;
  readmission_30d: ReadmissionResult;
  shap: ShapResult;
  engineered_features: Record<string, unknown>;
  model_version: string;
}

export interface PatientPayload {
  demographics: {
    age: number;
    gender: string;
    race: string;
    language: string;
  };
  admission: {
    admission_type: string;
    admission_location: string;
    insurance: string;
    marital_status: string;
    has_ed_visit: boolean;
    ed_los_hours: number;
    admit_hour: number;
    admit_day_of_week: number;
    los_days: number;
    previous_admissions: number;
    days_since_last_admission: number;
    discharge_location: string;
  };
  diagnoses: string[];
  procedures: string[];
  medications: string[];
}

export default function App() {
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<string | null>(null);
  const [isSuggesting, setIsSuggesting] = useState(false);

  const runSuggest = async (patient: PatientPayload, prediction: PredictionResponse) => {
    setIsSuggesting(true);
    setSuggestions(null);
    try {
      const res = await fetch(`${LAMBDA_URL}/suggest`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-api-key': LAMBDA_API_KEY },
        body: JSON.stringify({ patient, prediction }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      setSuggestions(data.suggestions as string);
    } catch (e) {
      setSuggestions(`Error: ${e instanceof Error ? e.message : 'Unknown error'}`);
    } finally {
      setIsSuggesting(false);
    }
  };

  const handlePredict = async (patient: PatientPayload) => {
    setIsLoading(true);
    setError(null);
    setPredictionResult(null);
    setSuggestions(null);

    try {
      const res = await fetch(`${LAMBDA_URL}/predict`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-api-key': LAMBDA_API_KEY,
        },
        body: JSON.stringify({ patient }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      const prediction = data as PredictionResponse;
      setPredictionResult(prediction);
      toast.success('Prediction complete');
      runSuggest(patient, prediction);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Unknown error';
      setError(msg);
      toast.error('Prediction failed', { description: msg });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <header className="backdrop-blur-xl bg-white/80 border-b border-white/20 shadow-lg sticky top-0 z-50">
        <div className="max-w-[1800px] mx-auto px-8 py-5" style={{ margin: "10px"}}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl blur-lg opacity-50"></div>
                <div className="relative p-3 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl">
                  <Activity className="size-7 text-white" />
                </div>
              </div>
              <div>
                <h1 className="bg-gradient-to-r from-blue-700 via-indigo-700 to-purple-700 bg-clip-text text-transparent text-2xl font-bold">
                  ICU Predictive Analysis
                </h1>
                <p className="text-sm text-gray-600 mt-0.5">Mortality · Length of Stay · 30-day Readmission</p>
              </div>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-green-50 to-emerald-50 rounded-full border border-green-200">
              <div className="size-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-green-700">Lambda Connected</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-[1800px] mx-auto px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6" style={{ minHeight: 'calc(100vh - 180px)' }}>
          {/* Left Panel — Input */}
          <div className="group relative">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl blur opacity-20 group-hover:opacity-30 transition"></div>
            <div className="relative bg-white/90 backdrop-blur-xl rounded-2xl shadow-2xl border border-white/20 overflow-hidden flex flex-col">
              <div className="p-6 border-b border-gray-100 bg-gradient-to-br from-blue-50/50 via-indigo-50/30 to-transparent flex-shrink-0">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg shadow-lg">
                    <Activity className="size-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-gray-900 font-semibold">Patient Data Input</h2>
                    <p className="text-sm text-gray-500 mt-0.5">Demographics, admission details, and clinical codes</p>
                  </div>
                </div>
              </div>
              <div className="p-6 overflow-auto flex-1">
                <PatientDataForm onPredict={handlePredict} isLoading={isLoading} />
              </div>
            </div>
          </div>

          {/* Right Panel — Results */}
          <div className="group relative">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl blur opacity-20 group-hover:opacity-30 transition"></div>
            <div className="relative bg-white/90 backdrop-blur-xl rounded-2xl shadow-2xl border border-white/20 overflow-hidden flex flex-col">
              <div className="p-6 border-b border-gray-100 bg-gradient-to-br from-purple-50/50 via-pink-50/30 to-transparent flex-shrink-0">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg shadow-lg">
                    <Activity className="size-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-gray-900 font-semibold">Prediction Results</h2>
                    <p className="text-sm text-gray-500 mt-0.5">Mortality risk, length of stay, readmission probability</p>
                  </div>
                </div>
              </div>
              <div className="p-6 overflow-auto flex-1">
                <PredictionResults
                  result={predictionResult}
                  isLoading={isLoading}
                  error={error}
                  suggestions={suggestions}
                  isSuggesting={isSuggesting}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      <Toaster />
    </div>
  );
}
