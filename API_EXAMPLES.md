# CABI Outfit Recommender API Examples

## Starting the Server

```bash
python run_server.py
```

The server will start at `http://localhost:8000`

## API Endpoints

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. API Info
```bash
curl http://localhost:8000/info
```

### 3. Root Endpoint
```bash
curl http://localhost:8000/
```

## Making Recommendations

### GET Request (Browser/Curl Friendly)

**Basic recommendation:**
```bash
curl "http://localhost:8000/recommend?prompt=smart%20casual%20dinner%20outfit&k=3"
```

**With specific colors:**
```bash
curl "http://localhost:8000/recommend?prompt=elegant%20work%20outfit%20in%20navy%20and%20cream&k=5"
```

**With required items:**
```bash
curl "http://localhost:8000/recommend?prompt=casual%20weekend%20outfit&k=3&must_include=444656,444698"
```

### POST Request (JSON)

**Using curl:**
```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "floral summer dress for weekend brunch",
    "k": 5
  }'
```

**Using Python requests:**
```python
import requests
import json

# Basic request
response = requests.post("http://localhost:8000/recommend", 
    json={
        "prompt": "smart casual dinner outfit in fall tones",
        "k": 3
    }
)

outfits = response.json()
print(f"Found {outfits['num_results']} outfits for: {outfits['query']}")

for i, outfit in enumerate(outfits['outfits'], 1):
    print(f"\nOutfit {i} (score: {outfit['score']}):")
    print(f"Palette: {', '.join(outfit['palette'])}")
    print(f"Base: {outfit['base']}")
    for item in outfit['items']:
        print(f"  - {item['product_id']}: {item['name']} ({item['category']})")
```

**With required items:**
```python
response = requests.post("http://localhost:8000/recommend", 
    json={
        "prompt": "professional work outfit",
        "k": 3,
        "must_include": ["444656", "444698"]
    }
)
```

## Example Prompts

### Work & Professional
- `"elegant work outfit in navy and cream"`
- `"professional blazer outfit for important meeting"`
- `"black slim trousers for work with a camel blazer"`

### Casual & Weekend
- `"casual weekend brunch outfit in earth tones"`
- `"comfortable weekend outfit for running errands"`
- `"relaxed Sunday outfit in soft colors"`

### Evening & Dinner
- `"smart casual dinner outfit in fall tones"`
- `"elegant dinner date outfit in jewel tones"`
- `"sophisticated evening look with bold colors"`

### Seasonal & Occasion
- `"floral summer dress for weekend brunch"`
- `"cozy fall outfit with warm layers"`
- `"fresh spring outfit with light fabrics"`

## Response Format

```json
{
  "query": "smart casual dinner outfit in fall tones",
  "num_results": 3,
  "outfits": [
    {
      "score": 0.849,
      "palette": ["red", "camel", "gray"],
      "base": "top+bottom",
      "items": [
        {
          "product_id": "444733",
          "name": "Scorch Top",
          "subtype": "Shirts & Blouses", 
          "category": "top",
          "colors": ["red"],
          "patterns": [],
          "occasions": ["smart casual", "work", "dinner", "weekend", "casual", "travel"]
        },
        {
          "product_id": "464965",
          "name": "Tulip Skirt",
          "subtype": "Skirts",
          "category": "bottom", 
          "colors": ["red"],
          "patterns": [],
          "occasions": ["smart casual", "work", "dinner", "weekend", "casual", "travel"]
        }
      ]
    }
  ]
}
```

## Interactive Documentation

Visit `http://localhost:8000/docs` for the interactive Swagger UI documentation where you can:
- Explore all endpoints
- Test API calls directly in the browser
- See detailed request/response schemas
- Try different prompts and parameters

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid parameters)
- `500`: Internal server error
- `503`: Service unavailable (models not loaded)

Example error response:
```json
{
  "detail": "Models not loaded. Please check server startup logs."
}
```
