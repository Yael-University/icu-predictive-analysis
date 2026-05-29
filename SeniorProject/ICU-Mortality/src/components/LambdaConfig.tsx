import { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from './ui/dialog';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { LambdaConfig as Config } from '../App';
import { Zap, Eye, EyeOff } from 'lucide-react';

interface LambdaConfigProps {
  open: boolean;
  config: Config;
  onSave: (config: Config) => void;
  onClose: () => void;
}

export function LambdaConfig({ open, config, onSave, onClose }: LambdaConfigProps) {
  const [url, setUrl] = useState(config.url);
  const [apiKey, setApiKey] = useState(config.apiKey);
  const [showKey, setShowKey] = useState(false);

  useEffect(() => {
    setUrl(config.url);
    setApiKey(config.apiKey);
  }, [config, open]);

  const handleSave = () => {
    if (!url.trim()) return;
    onSave({ url: url.trim(), apiKey: apiKey.trim() });
  };

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <div className="p-1.5 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-lg">
              <Zap className="size-4 text-white" />
            </div>
            Lambda Configuration
          </DialogTitle>
          <DialogDescription>
            Enter your AWS Lambda Function URL and API key. These are saved to your browser's local storage.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 pt-2">
          <div className="space-y-1.5">
            <Label htmlFor="lambda-url" className="text-gray-700">
              Lambda Function URL <span className="text-red-400">*</span>
            </Label>
            <Input
              id="lambda-url"
              type="url"
              placeholder="https://xxxxxxxx.lambda-url.us-east-1.on.aws"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              className="font-mono text-sm"
            />
            <p className="text-xs text-gray-400">
              Run <code className="bg-gray-100 px-1 rounded">bash deploy/deploy_lambda.sh</code> and copy the printed URL
            </p>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="api-key" className="text-gray-700">API Key (x-api-key)</Label>
            <div className="relative">
              <Input
                id="api-key"
                type={showKey ? 'text' : 'password'}
                placeholder="your-long-random-secret"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                className="font-mono text-sm pr-10"
              />
              <button
                type="button"
                onClick={() => setShowKey((v) => !v)}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition"
              >
                {showKey ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
              </button>
            </div>
            <p className="text-xs text-gray-400">
              Matches the <code className="bg-gray-100 px-1 rounded">API_KEY</code> environment variable set during deploy
            </p>
          </div>

          <div className="rounded-lg bg-amber-50 border border-amber-200 p-3 text-xs text-amber-700">
            <strong>Note:</strong> The API key is stored in browser localStorage. Do not use this tool on
            shared or public computers.
          </div>

          <div className="flex gap-2 pt-1">
            <Button
              onClick={handleSave}
              disabled={!url.trim()}
              className="flex-1 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
            >
              Save & Connect
            </Button>
            {config.url && (
              <Button variant="outline" onClick={onClose} className="flex-1">
                Cancel
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
