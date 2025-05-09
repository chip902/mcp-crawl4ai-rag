#!/bin/bash

# Set the Supabase URL and key directly
SUPABASE_URL="http://192.168.1.10:8000"
SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.ewogICJyb2xlIjogInNlcnZpY2Vfcm9sZSIsCiAgImlzcyI6ICJzdXBhYmFzZSIsCiAgImlhdCI6IDE3Mzg5MDQ0MDAsCiAgImV4cCI6IDE4OTY2NzA4MDAKfQ.0NWkt9VTZABAo76O3KtKDrt2-fRZJPt-TXlaCRmkzMM"

echo "Setting up database tables at $SUPABASE_URL"

# Execute the SQL file against Supabase
curl -X POST \
  "$SUPABASE_URL/rest/v1/rpc/exec_sql" \
  -H "apikey: $SUPABASE_KEY" \
  -H "Authorization: Bearer $SUPABASE_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$(cat crawled_pages.sql | tr -d '\n' | sed 's/"/\\"/g')\"}"

echo -e "\nDatabase setup complete!"
