#!/usr/bin/env bash
set -e

# Start PostgreSQL if not already running
if ! pg_isready -q 2>/dev/null; then
    service postgresql start
    # Wait until accepting connections
    until pg_isready -q; do sleep 0.2; done
fi

exec "$@"
