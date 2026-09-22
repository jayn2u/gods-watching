#!/bin/sh
set -eu

: "${FIXTURE_INPUT:?FIXTURE_INPUT is required}"
: "${FIXTURE_PATH:?FIXTURE_PATH is required}"
: "${RTSP_HOST:?RTSP_HOST is required}"
: "${RTSP_PORT:=8554}"

destination="rtsp://${RTSP_HOST}:${RTSP_PORT}/${FIXTURE_PATH}"

while true; do
  ffmpeg \
    -hide_banner \
    -loglevel warning \
    -nostdin \
    -re \
    -protocol_whitelist file \
    -i "${FIXTURE_INPUT}" \
    -map 0:v:0 \
    -an \
    -c:v copy \
    -f rtsp \
    -rtsp_transport tcp \
    "${destination}"

  while timeout 3 ffprobe \
    -v error \
    -rtsp_transport tcp \
    -rw_timeout 2000000 \
    -protocol_whitelist rtsp,tcp,udp,rtp \
    -select_streams v:0 \
    -read_intervals '%+0.1' \
    -show_entries stream=codec_name \
    -of csv=p=0 \
    "${destination}" >/dev/null 2>&1
  do
    sleep 0.1
  done

  printf 'fixture_session_boundary path=%s\n' "${FIXTURE_PATH}"
  sleep 0.1
done
