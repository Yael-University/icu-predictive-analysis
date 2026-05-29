import { Button } from './ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from './ui/table';
import { Badge } from './ui/badge';
import { Trash2, Download, TrendingUp, Users } from 'lucide-react';
import { PatientData } from '../App';

interface DataPreviewProps {
  data: PatientData[];
  onClear: () => void;
}

export function DataPreview({ data, onClear }: DataPreviewProps) {
  const handleDownloadCSV = () => {
    const headers = ['age', 'gender', 'race', 'bmi', 'systolic', 'diastolic', 'heart_rate'];
    const csvContent = [
      headers.join(','),
      ...data.map(patient => 
        `${patient.age},${patient.gender},${patient.race},${patient.bmi},${patient.systolicBP},${patient.diastolicBP},${patient.heartRate}`
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `patient_data_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const getGenderBadgeColor = (gender: string) => {
    if (gender === 'M') return 'bg-blue-100 text-blue-700';
    if (gender === 'F') return 'bg-pink-100 text-pink-700';
    return 'bg-gray-100 text-gray-700';
  };

  const getBPStatus = (systolic: number, diastolic: number) => {
    if (systolic >= 140 || diastolic >= 90) {
      return { label: 'High', color: 'bg-red-100 text-red-700' };
    } else if (systolic >= 130 || diastolic >= 80) {
      return { label: 'Elevated', color: 'bg-orange-100 text-orange-700' };
    }
    return { label: 'Normal', color: 'bg-green-100 text-green-700' };
  };

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-2 gap-4">
        <div className="relative group">
          <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-400 to-pink-400 rounded-xl blur opacity-20"></div>
          <div className="relative bg-gradient-to-br from-purple-50 to-pink-50 p-4 rounded-xl border border-purple-200/50">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-purple-500 to-pink-600 rounded-lg shadow-md">
                <Users className="size-5 text-white" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Total Patients</p>
                <p className="text-2xl text-gray-900">{data.length}</p>
              </div>
            </div>
          </div>
        </div>

        <div className="relative group">
          <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-400 to-indigo-400 rounded-xl blur opacity-20"></div>
          <div className="relative bg-gradient-to-br from-blue-50 to-indigo-50 p-4 rounded-xl border border-blue-200/50">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg shadow-md">
                <TrendingUp className="size-5 text-white" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Avg Age</p>
                <p className="text-2xl text-gray-900">
                  {data.length > 0 ? Math.round(data.reduce((sum, p) => sum + (p.age || 0), 0) / data.length) : 0}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex items-center justify-between">
        <h3 className="text-gray-900">Patient Records</h3>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={handleDownloadCSV} className="border-gray-300 hover:border-blue-400 hover:bg-blue-50 transition-colors shadow-sm">
            <Download className="size-4 mr-2" />
            Export CSV
          </Button>
          <Button variant="outline" size="sm" onClick={onClear} className="border-gray-300 hover:border-red-400 hover:bg-red-50 hover:text-red-600 transition-colors shadow-sm">
            <Trash2 className="size-4 mr-2" />
            Clear All
          </Button>
        </div>
      </div>

      {/* Table */}
      <div className="relative group">
        <div className="absolute -inset-0.5 bg-gradient-to-r from-gray-200 to-gray-300 rounded-xl blur opacity-10"></div>
        <div className="relative border border-gray-200/50 rounded-xl overflow-hidden shadow-lg bg-white">
          <div className="overflow-x-auto max-h-[500px]">
            <Table>
              <TableHeader>
                <TableRow className="bg-gradient-to-r from-gray-50 to-gray-100/50 border-b-2 border-gray-200">
                  <TableHead className="w-12">#</TableHead>
                  <TableHead>Age</TableHead>
                  <TableHead>Gender</TableHead>
                  <TableHead>Race</TableHead>
                  <TableHead>BMI</TableHead>
                  <TableHead>BP (mmHg)</TableHead>
                  <TableHead>HR (bpm)</TableHead>
                  <TableHead>BP Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.map((patient, index) => {
                  const bpStatus = getBPStatus(patient.systolicBP, patient.diastolicBP);
                  return (
                    <TableRow key={patient.id} className="hover:bg-gradient-to-r hover:from-purple-50/30 hover:to-pink-50/30 transition-colors">
                      <TableCell className="text-gray-500">{index + 1}</TableCell>
                      <TableCell>{patient.age || 'N/A'}</TableCell>
                      <TableCell>
                        <Badge variant="secondary" className={`${getGenderBadgeColor(patient.gender)} shadow-sm`}>
                          {patient.gender || 'N/A'}
                        </Badge>
                      </TableCell>
                      <TableCell className="capitalize">{patient.race || 'N/A'}</TableCell>
                      <TableCell>{patient.bmi ? patient.bmi.toFixed(1) : 'N/A'}</TableCell>
                      <TableCell>
                        {patient.systolicBP && patient.diastolicBP 
                          ? `${patient.systolicBP}/${patient.diastolicBP}`
                          : 'N/A'}
                      </TableCell>
                      <TableCell>{patient.heartRate || 'N/A'}</TableCell>
                      <TableCell>
                        <Badge variant="secondary" className={`${bpStatus.color} shadow-sm`}>
                          {bpStatus.label}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </div>
      </div>

      {/* Info Banner */}
      <div className="relative group">
        <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-400 to-indigo-400 rounded-xl blur opacity-10"></div>
        <div className="relative bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 p-5 rounded-xl border border-blue-200/50 shadow-md">
          <div className="flex items-start gap-3">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg shadow-md mt-0.5">
              <TrendingUp className="size-4 text-white" />
            </div>
            <div>
              <p className="text-sm text-gray-900 mb-1">
                <strong>Ready for ML Analysis</strong>
              </p>
              <p className="text-sm text-gray-600">
                This dataset is prepared for mortality prediction. Future versions will integrate the logistic regression model to display risk predictions, confusion matrices, and performance metrics in this panel.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}