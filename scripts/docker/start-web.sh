#!/usr/bin/env bash
set -euo pipefail

cd /workspace

if [ ! -d /workspace/web/node_modules/lucide-react ]; then
  echo "Installing web dependencies in container..."
  npm --prefix /workspace/web ci
fi

echo "Starting Next.js on http://localhost:3000 ..."
exec npm --prefix /workspace/web run dev -- --hostname 0.0.0.0 --port 3000
