curl http://localhost:8787/v1/completions \
	-H "Content-Type: application/json" \
	 -d '{
        "model": "google/gemma-3-12b-it",
	"prompt": "what is ai?",
	"max_tokens": 7,
	"temperature": 0
	 }'
