#!/usr/bin/env bash
set -euo pipefail

cd /home/metricspace/ctox-public-20260322-full
~/.cargo/bin/cargo build --quiet
