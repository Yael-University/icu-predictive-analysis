import { useState, KeyboardEvent } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Switch } from './ui/switch';
import { Badge } from './ui/badge';
import { Separator } from './ui/separator';
import { UserPlus, X, Stethoscope, ClipboardList, Pill, Loader2, ChevronDown, ChevronUp } from 'lucide-react';
import { PatientPayload } from '../App';
import { toast } from 'sonner';

interface PatientDataFormProps {
  onPredict: (patient: PatientPayload) => void;
  isLoading: boolean;
}

// ── Tag Input ────────────────────────────────────────────────────────────────

interface TagInputProps {
  tags: string[];
  onChange: (tags: string[]) => void;
  placeholder: string;
}

function TagInput({ tags, onChange, placeholder }: TagInputProps) {
  const [input, setInput] = useState('');

  const addTag = (value: string) => {
    const trimmed = value.trim().toUpperCase();
    if (trimmed && !tags.includes(trimmed)) {
      onChange([...tags, trimmed]);
    }
    setInput('');
  };

  const handleKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      addTag(input);
    } else if (e.key === 'Backspace' && !input && tags.length > 0) {
      onChange(tags.slice(0, -1));
    }
  };

  const removeTag = (index: number) => {
    onChange(tags.filter((_, i) => i !== index));
  };

  return (
    <div className="border border-gray-300 rounded-md p-2 min-h-[42px] flex flex-wrap gap-1.5 focus-within:ring-2 focus-within:ring-blue-400/30 focus-within:border-blue-400 transition bg-white">
      {tags.map((tag, i) => (
        <span
          key={i}
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-indigo-100 text-indigo-700 text-xs font-medium"
        >
          {tag}
          <button
            type="button"
            onClick={() => removeTag(i)}
            className="hover:text-indigo-900 transition"
          >
            <X className="size-3" />
          </button>
        </span>
      ))}
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKey}
        onBlur={() => input.trim() && addTag(input)}
        placeholder={tags.length === 0 ? placeholder : ''}
        className="flex-1 min-w-[120px] outline-none text-sm bg-transparent placeholder:text-gray-400"
      />
    </div>
  );
}

// ── MedicationTagInput (lowercase) ───────────────────────────────────────────

interface MedTagInputProps {
  tags: string[];
  onChange: (tags: string[]) => void;
}

function MedTagInput({ tags, onChange }: MedTagInputProps) {
  const [input, setInput] = useState('');

  const addTag = (value: string) => {
    const trimmed = value.trim().toLowerCase();
    if (trimmed && !tags.includes(trimmed)) {
      onChange([...tags, trimmed]);
    }
    setInput('');
  };

  const handleKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      addTag(input);
    } else if (e.key === 'Backspace' && !input && tags.length > 0) {
      onChange(tags.slice(0, -1));
    }
  };

  return (
    <div className="border border-gray-300 rounded-md p-2 min-h-[42px] flex flex-wrap gap-1.5 focus-within:ring-2 focus-within:ring-blue-400/30 focus-within:border-blue-400 transition bg-white">
      {tags.map((tag, i) => (
        <span
          key={i}
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-purple-100 text-purple-700 text-xs font-medium"
        >
          {tag}
          <button
            type="button"
            onClick={() => onChange(tags.filter((_, j) => j !== i))}
            className="hover:text-purple-900 transition"
          >
            <X className="size-3" />
          </button>
        </span>
      ))}
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKey}
        onBlur={() => input.trim() && addTag(input)}
        placeholder={tags.length === 0 ? 'aspirin, metformin, …' : ''}
        className="flex-1 min-w-[120px] outline-none text-sm bg-transparent placeholder:text-gray-400"
      />
    </div>
  );
}

// ── Section Header ────────────────────────────────────────────────────────────

function SectionDivider({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 mt-2">
      <div className="h-px flex-1 bg-gradient-to-r from-blue-200 to-transparent"></div>
      <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">{label}</span>
      <div className="h-px flex-1 bg-gradient-to-l from-blue-200 to-transparent"></div>
    </div>
  );
}

// ── Main Form ─────────────────────────────────────────────────────────────────

const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

export function PatientDataForm({ onPredict, isLoading }: PatientDataFormProps) {
  const [showOptional, setShowOptional] = useState(true);

  // Demographics
  const [age, setAge] = useState('67');
  const [gender, setGender] = useState('M');
  const [race, setRace] = useState('WHITE');
  const [language, setLanguage] = useState('ENGLISH');

  // Admission
  const [admissionType, setAdmissionType] = useState('EMERGENCY');
  const [admissionLocation, setAdmissionLocation] = useState('EMERGENCY ROOM');
  const [insurance, setInsurance] = useState('Medicare');
  const [maritalStatus, setMaritalStatus] = useState('MARRIED');
  const [hasEdVisit, setHasEdVisit] = useState(true);
  const [edLosHours, setEdLosHours] = useState('4.5');

  // Optional timing
  const [admitHour, setAdmitHour] = useState('14');
  const [admitDayOfWeek, setAdmitDayOfWeek] = useState('2');

  // Optional discharge / history (for readmission model)
  const [losDays, setLosDays] = useState('5.0');
  const [previousAdmissions, setPreviousAdmissions] = useState('2');
  const [daysSinceLast, setDaysSinceLast] = useState('45');
  const [dischargeLocation, setDischargeLocation] = useState('HOME');

  // Clinical codes
  const [diagnoses, setDiagnoses] = useState<string[]>(['I10', 'E11.9', 'J18.9']);
  const [procedures, setProcedures] = useState<string[]>(['5A1955Z', '0BH17EZ']);
  const [medications, setMedications] = useState<string[]>(['aspirin', 'metformin', 'ceftriaxone']);

  const inputClass = 'border-gray-300 focus:border-blue-400 focus:ring-blue-400/20 h-9';
  const selectClass = 'border-gray-300 focus:border-blue-400 focus:ring-blue-400/20 h-9';

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!age || !gender || !race || !admissionType || !insurance) {
      toast.error('Fill in all required fields', {
        description: 'Age, gender, race, admission type, and insurance are required.',
      });
      return;
    }

    const patient: PatientPayload = {
      demographics: {
        age: Number(age),
        gender,
        race,
        language,
      },
      admission: {
        admission_type: admissionType,
        admission_location: admissionLocation,
        insurance,
        marital_status: maritalStatus,
        has_ed_visit: hasEdVisit,
        ed_los_hours: hasEdVisit ? Number(edLosHours) || 0 : 0,
        admit_hour: Number(admitHour),
        admit_day_of_week: Number(admitDayOfWeek),
        los_days: losDays ? Number(losDays) : 0,
        previous_admissions: Number(previousAdmissions) || 0,
        days_since_last_admission: daysSinceLast ? Number(daysSinceLast) : 0,
        discharge_location: dischargeLocation,
      },
      diagnoses,
      procedures,
      medications,
    };

    onPredict(patient);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-5">

      {/* ── Demographics ── */}
      <SectionDivider label="Demographics" />

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1.5">
          <Label htmlFor="age" className="text-gray-700 text-sm">Age <span className="text-red-400">*</span></Label>
          <Input
            id="age"
            type="number"
            placeholder="67"
            value={age}
            onChange={(e) => setAge(e.target.value)}
            min="0"
            max="120"
            required
            className={inputClass}
          />
        </div>

        <div className="space-y-1.5">
          <Label className="text-gray-700 text-sm">Gender <span className="text-red-400">*</span></Label>
          <Select value={gender} onValueChange={setGender} required>
            <SelectTrigger className={selectClass}>
              <SelectValue placeholder="Select" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="M">Male</SelectItem>
              <SelectItem value="F">Female</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1.5">
          <Label className="text-gray-700 text-sm">Race / Ethnicity <span className="text-red-400">*</span></Label>
          <Select value={race} onValueChange={setRace} required>
            <SelectTrigger className={selectClass}>
              <SelectValue placeholder="Select" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="WHITE">White</SelectItem>
              <SelectItem value="BLACK/AFRICAN AMERICAN">Black / African American</SelectItem>
              <SelectItem value="ASIAN">Asian</SelectItem>
              <SelectItem value="HISPANIC/LATINO">Hispanic / Latino</SelectItem>
              <SelectItem value="AMERICAN INDIAN/ALASKA NATIVE">American Indian / Alaska Native</SelectItem>
              <SelectItem value="OTHER">Other / Unknown</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label className="text-gray-700 text-sm">Language</Label>
          <Select value={language} onValueChange={setLanguage}>
            <SelectTrigger className={selectClass}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="ENGLISH">English</SelectItem>
              <SelectItem value="SPANISH">Spanish</SelectItem>
              <SelectItem value="OTHER">Other</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* ── Admission ── */}
      <SectionDivider label="Admission" />

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1.5">
          <Label className="text-gray-700 text-sm">Admission Type <span className="text-red-400">*</span></Label>
          <Select value={admissionType} onValueChange={setAdmissionType} required>
            <SelectTrigger className={selectClass}>
              <SelectValue placeholder="Select" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="EMERGENCY">Emergency</SelectItem>
              <SelectItem value="ELECTIVE">Elective</SelectItem>
              <SelectItem value="URGENT">Urgent</SelectItem>
              <SelectItem value="EW_EMER">EW Emergency</SelectItem>
              <SelectItem value="OBSERVATION ADMIT">Observation Admit</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label className="text-gray-700 text-sm">Insurance <span className="text-red-400">*</span></Label>
          <Select value={insurance} onValueChange={setInsurance} required>
            <SelectTrigger className={selectClass}>
              <SelectValue placeholder="Select" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Medicare">Medicare</SelectItem>
              <SelectItem value="Medicaid">Medicaid</SelectItem>
              <SelectItem value="Private">Private</SelectItem>
              <SelectItem value="Other">Other</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-1.5">
          <Label className="text-gray-700 text-sm">Admission Location</Label>
          <Select value={admissionLocation} onValueChange={setAdmissionLocation}>
            <SelectTrigger className={selectClass}>
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="EMERGENCY ROOM">Emergency Room</SelectItem>
              <SelectItem value="PHYSICIAN REFERRAL">Physician Referral</SelectItem>
              <SelectItem value="TRANSFER FROM HOSPITAL">Transfer from Hospital</SelectItem>
              <SelectItem value="CLINIC REFERRAL">Clinic Referral</SelectItem>
              <SelectItem value="WALK-IN/SELF REFERRAL">Walk-in / Self Referral</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label className="text-gray-700 text-sm">Marital Status</Label>
          <Select value={maritalStatus} onValueChange={setMaritalStatus}>
            <SelectTrigger className={selectClass}>
              <SelectValue placeholder="Select" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="MARRIED">Married</SelectItem>
              <SelectItem value="SINGLE">Single</SelectItem>
              <SelectItem value="WIDOWED">Widowed</SelectItem>
              <SelectItem value="DIVORCED">Divorced</SelectItem>
              <SelectItem value="UNKNOWN (DEFAULT)">Unknown</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* ED Visit */}
      <div className="flex items-center gap-3 p-3 rounded-lg bg-gray-50 border border-gray-200">
        <Switch
          id="hasEdVisit"
          checked={hasEdVisit}
          onCheckedChange={setHasEdVisit}
        />
        <Label htmlFor="hasEdVisit" className="text-gray-700 text-sm cursor-pointer">
          ED visit prior to this admission
        </Label>
        {hasEdVisit && (
          <div className="ml-auto flex items-center gap-2">
            <Label className="text-gray-600 text-xs whitespace-nowrap">ED hours:</Label>
            <Input
              type="number"
              step="0.5"
              placeholder="4.5"
              value={edLosHours}
              onChange={(e) => setEdLosHours(e.target.value)}
              min="0"
              max="72"
              className="w-24 h-8 text-sm border-gray-300"
            />
          </div>
        )}
      </div>

      {/* ── Clinical Codes ── */}
      <SectionDivider label="Clinical Codes" />

      <div className="space-y-3">
        <div className="space-y-1.5">
          <Label className="text-gray-700 text-sm flex items-center gap-1.5">
            <Stethoscope className="size-3.5 text-indigo-500" />
            Diagnoses (ICD codes)
          </Label>
          <TagInput
            tags={diagnoses}
            onChange={setDiagnoses}
            placeholder="I10, E11.9, J18.9 — press Enter or comma"
          />
          <p className="text-xs text-gray-400">Type an ICD code and press Enter or comma to add</p>
        </div>

        <div className="space-y-1.5">
          <Label className="text-gray-700 text-sm flex items-center gap-1.5">
            <ClipboardList className="size-3.5 text-indigo-500" />
            Procedures (ICD codes)
          </Label>
          <TagInput
            tags={procedures}
            onChange={setProcedures}
            placeholder="5A1955Z, 0BH17EZ — press Enter or comma"
          />
        </div>

        <div className="space-y-1.5">
          <Label className="text-gray-700 text-sm flex items-center gap-1.5">
            <Pill className="size-3.5 text-purple-500" />
            Medications
          </Label>
          <MedTagInput tags={medications} onChange={setMedications} />
        </div>
      </div>

      {/* ── Optional: Timing & History ── */}
      <button
        type="button"
        onClick={() => setShowOptional((v) => !v)}
        className="w-full flex items-center justify-between text-sm text-gray-500 hover:text-gray-700 transition py-1"
      >
        <span className="font-medium">Optional: timing &amp; history (improves LOS &amp; readmission)</span>
        {showOptional ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}
      </button>

      {showOptional && (
        <div className="space-y-4 border border-dashed border-gray-200 rounded-xl p-4 bg-gray-50/50">
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label className="text-gray-700 text-sm">Admit Hour (0–23)</Label>
              <Input
                type="number"
                placeholder="14"
                value={admitHour}
                onChange={(e) => setAdmitHour(e.target.value)}
                min="0"
                max="23"
                className={inputClass}
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-gray-700 text-sm">Day of Week Admitted</Label>
              <Select value={admitDayOfWeek} onValueChange={setAdmitDayOfWeek}>
                <SelectTrigger className={selectClass}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {DAYS.map((d, i) => (
                    <SelectItem key={i} value={String(i)}>{d}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <SectionDivider label="Discharge / History" />

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label className="text-gray-700 text-sm">LOS this admission (days)</Label>
              <Input
                type="number"
                step="0.1"
                placeholder="5.0"
                value={losDays}
                onChange={(e) => setLosDays(e.target.value)}
                min="0"
                className={inputClass}
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-gray-700 text-sm">Previous admissions</Label>
              <Input
                type="number"
                placeholder="2"
                value={previousAdmissions}
                onChange={(e) => setPreviousAdmissions(e.target.value)}
                min="0"
                className={inputClass}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label className="text-gray-700 text-sm">Days since last admission</Label>
              <Input
                type="number"
                placeholder="45"
                value={daysSinceLast}
                onChange={(e) => setDaysSinceLast(e.target.value)}
                min="0"
                className={inputClass}
              />
            </div>
            <div className="space-y-1.5">
              <Label className="text-gray-700 text-sm">Discharge location</Label>
              <Select value={dischargeLocation} onValueChange={setDischargeLocation}>
                <SelectTrigger className={selectClass}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="HOME">Home</SelectItem>
                  <SelectItem value="HOME WITH SERVICE">Home with Service</SelectItem>
                  <SelectItem value="SKILLED NURSING FACILITY">Skilled Nursing Facility</SelectItem>
                  <SelectItem value="REHAB">Rehab</SelectItem>
                  <SelectItem value="DIED">Died</SelectItem>
                  <SelectItem value="OTHER FACILITY">Other Facility</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      )}

      <Separator className="my-1" />

      <Button
        type="submit"
        disabled={isLoading}
        className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-lg hover:shadow-xl transition-all h-11"
      >
        {isLoading ? (
          <>
            <Loader2 className="size-4 mr-2 animate-spin" />
            Running prediction…
          </>
        ) : (
          <>
            <UserPlus className="size-4 mr-2" />
            Run Prediction
          </>
        )}
      </Button>
    </form>
  );
}
