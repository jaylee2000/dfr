#!/usr/bin/env bash

if [[ -z "${ROOT_DIR:-}" ]]; then
  echo "scripts/load_local_env.sh requires ROOT_DIR to be set" >&2
  exit 2
fi

LOCAL_ENV="${LOCAL_ENV:-${ROOT_DIR}/local.env}"
if [[ -f "${LOCAL_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${LOCAL_ENV}"
  set +a
fi
