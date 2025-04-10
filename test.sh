#!/bin/bash

# Check if an audio file path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_audio_file>"
  exit 1
fi

AUDIO_FILE=$1
SERVICE_URL="http://localhost:8123/"

# Check if the audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
  echo "Error: Audio file not found at $AUDIO_FILE"
  exit 1
fi

# Send the request to the service
echo "Sending transcription request for $AUDIO_FILE to $SERVICE_URL"
curl -X POST \
     -F "file=@$AUDIO_FILE" \
     $SERVICE_URL

echo "\nRequest finished."
