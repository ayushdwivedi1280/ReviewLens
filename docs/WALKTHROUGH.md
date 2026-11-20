# ReviewLens Walkthrough & Interview Guide

This guide is written for the candidate to walk through the codebase during a live technical interview, explain the architectural decisions, and make live code changes on the spot with confidence.

---

## Architecture Overview

```
[ User Browser ]
       |
       | Uploads reviews.csv (Multipart Form)
       v
[ FastAPI: app/main.py ]
       |
       v
[ app/pipeline.py: analyze_file() ]
       |
       +---> 1. Data Cleaning [ app/cleaning.py ]
       |        - Parses CSV with utf-8-sig
       |        - Strips HTML & collapses whitespace
       |        - Normalizes ratings (1-5 or None)
       |        - Detects duplicates by casefolded SHA-256 hash
       |
       +---> 2. Batch Classification [ app/analyzer.py ]
       |        - Checks SQLite Cache [ app/cache.py ]
       |        - Chunks uncached reviews into batches of 5
       |        - Injects reviews securely into Prompt [ app/prompts.py ]
       |        - Calls LLM with backoff [ app/llm.py ]
       |        - Validates with strict schema [ app/schema.py ]
       |        - Retries failed items once; marks unrecoverable as analysis_failed
       |        - Checks star rating vs sentiment mismatch
       |
       +---> 3. Aggregate Insights [ app/insights.py ]
       |        - Extracts Top 3 Complaints & Top 3 Praises
       |        - Deterministic fallback on failures
       |        - Caches insights for zero repeated calls
       |
       v
[ JSON Response ] ---> Rendered in Vanilla JS Dashboard [ static/index.html ]
       |
       | User clicks "Suggest reply"
       v
[ POST /api/reply ] ---> Suggested Store Owner Response [ app/replies.py ]
```

---

## File-by-File Breakdown

### 1. `app/config.py`
- **What it does**: Manages application settings, provider tables (Groq and Gemini), and environment variables.
- **Key Functions**:
  - `PROVIDERS`: Dictionary registering endpoint URLs, default model identifiers (`llama-3.3-70b-versatile` for Groq, `gemini-2.5-flash` for Gemini), and required API key environment variable names.
  - `get_settings()`: Reads environment variables lazily. If an API key is missing, raises `ConfigError` with helpful instructions rather than crashing on server boot.
- **Why**: Keeps secrets externalized in environment variables and lets the server start even before an API key is provided, returning an informative 503 error on first use.

### 2. `app/cleaning.py`
- **What it does**: Parses uploaded CSV bytes and cleans messy text before it touches any AI model.
- **Key Functions**:
  - `parse_reviews(file_bytes)`: Reads CSV with `utf-8-sig`. Validates required header columns (`review_id`, `product`, `rating`, `date`, `review_text`). Returns `CleanedData(reviews, skipped, duplicates)`.
  - `clean_text(raw_text)`: Strips HTML tags (e.g. `<b>` in R024), unescapes HTML entities, collapses extra whitespace, and caps length at 2000 characters while preserving emojis and Hindi script. Empty text is routed to `skipped`.
  - `parse_rating(val)`: Ensures ratings are strictly integers 1 to 5. Missing values or non-integers (like `"five"` in R022) become `None` ("unknown").
  - `compute_text_hash(text)`: Produces a SHA-256 hash of casefolded cleaned text to identify duplicates (e.g. R015 duplicates R012).
  - `check_injection_suspected(text)`: Heuristic regex flag marking prompt injection patterns for UI display.
- **Why**: Protects downstream systems from crashing on malformed input, avoids paying for duplicate AI queries, and sanitizes dangerous HTML characters.

### 3. `app/schema.py`
- **What it does**: Defines strict Pydantic v2 validation models for LLM responses.
- **Key Functions**:
  - `ReviewAnalysis`: Model with `extra="forbid"` and `strict=True`. Enforces allowed sentiments (`positive`, `neutral`, `negative`), allowed topics (`quality`, `delivery`, `price`, `packaging`, `support`, `size_fit`, `other`), single-line English summary, and boolean `needs_reply`.
  - `TopInsights`: Model containing lists of `top_complaints` and `top_praises` with title, count, and review IDs.
  - `load_json(raw_text)`: Safely extracts JSON from model responses, stripping markdown code fences (` ```json `) and isolating braces.
- **Why**: Guarantees that hallucinated fields, invalid categories, or hijacked outputs are caught immediately and rejected.

### 4. `app/prompts.py`
- **What it does**: Contains system prompts and secure user-message builders.
- **Key Functions**:
  - `ANALYSIS_SYSTEM_PROMPT`: Directs the LLM on classification criteria, sarcasm recognition (e.g. R003 is negative), Hindi/Hinglish translation, and explicitly declares customer text inside tags as untrusted DATA.
  - `build_analysis_user_message(batch_items)`: Serializes reviews using `json.dumps` inside `<reviews>...</reviews>` tags. Star ratings are deliberately excluded. IDs are batch-local integers (1..N).
  - `REPLY_SYSTEM_PROMPT`: Instructs the LLM to write an empathetic merchant reply under 70 words without making up discounts or return policies.
- **Why**: Implements defense-in-depth against prompt injection attacks (such as R012). Malicious instructions remain safely encapsulated as JSON strings inside XML data tags.

### 5. `app/llm.py`
- **What it does**: Provides a unified HTTP client for OpenAI-compatible endpoints with exponential backoff and jitter.
- **Key Functions**:
  - `backoff_delay(attempt, retry_after)`: Calculates `base = min(30.0, 2 ** (attempt - 1))` plus up to 25% random jitter. Honours `Retry-After` headers if present.
  - `LLMClient.complete(system, user, json_mode)`: Sends requests to `/chat/completions`. Automatically retries on 408, 429, 500, 502, 503, 504, or connection timeouts up to 5 attempts. Immediately raises `LLMRequestRejected` on permanent 4xx errors (e.g. 401, 403). Uses injectable `sleep_fn` for fast unit tests.
- **Why**: Handles provider rate limits gracefully without stalling or failing prematurely, while avoiding unnecessary retries on permanent authentication errors.

### 6. `app/cache.py`
- **What it does**: Provides an SQLite key-value store for caching analysis and insights results.
- **Key Functions**:
  - `Cache(db_path)`: Connects to SQLite (file-based or `:memory:`) and initializes the `cache` table.
  - `get(key)` / `set(key, value)`: Stores JSON serialized results. Keys follow the format `review:{PROMPT_VERSION}:{model}:{text_hash}`.
- **Why**: Guarantees zero external AI calls when re-uploading identical files or previously analysed reviews, saving time and API quota.

### 7. `app/analyzer.py`
- **What it does**: Manages batching of reviews, cache lookups, single retry logic, and fallback to `analysis_failed`.
- **Key Functions**:
  - `ReviewAnalyzer.analyze(reviews)`: Checks cache for each review first. Chunks uncached reviews into batches of 5. Calls LLM sequentially (no concurrency).
  - `_process_batch(chunk, model_name)`: Maps batch-local IDs, parses response, validates schema. Any missing or malformed item is retried once in a secondary request. Items that still fail are recorded with `status="analysis_failed"`. Never raises or crashes.
  - `check_rating_mismatch(rating, sentiment)`: Compares star rating with sentiment (rating >= 4 expects positive; rating == 3 expects neutral; rating <= 2 expects negative; missing rating is never a mismatch).
- **Why**: Ensures reliable end-to-end processing even when third-party AI models return malformed or partial output.

### 8. `app/insights.py`
- **What it does**: Aggregates reviews across all products to generate top 3 complaints and top 3 praises.
- **Key Functions**:
  - `extract_top_insights(llm, cache, analyzed_reviews)`: Asks LLM for top themes, validates IDs, caches result by hash of all inputs, and retries once on failure.
  - `generate_deterministic_insights(reviews)`: Fallback method that counts topic frequencies for negative reviews (complaints) and positive reviews (praises) without calling any external API.
- **Why**: Ensures store owners always see aggregate actionable themes even if the LLM is down or rate-limited.

### 9. `app/replies.py`
- **What it does**: Generates on-demand store owner replies when clicked by the user.
- **Key Functions**:
  - `generate_suggested_reply(llm, product, review_text, sentiment, summary)`: Generates a polite, empathetic store response under 70 words. If the API fails, returns a polite fallback response without inventing discounts.
- **Why**: Keeps reply generation on-demand to save tokens and prevent upfront latency.

### 10. `app/pipeline.py`
- **What it does**: Coordinates the entire workflow from CSV bytes to summary statistics, metrics, and in-memory run storage.
- **Key Functions**:
  - `analyze_file(file_bytes, llm, cache, batch_size)`: Executes cleaning -> analyzer -> insights -> summary metrics (sentiment split, rating agreement %, AI calls, seconds elapsed).
  - `RUN_STORE`: In-memory dictionary mapping `run_id -> {review_id -> review_data}` for safe server-side reply generation.
- **Why**: Clean separation between application business logic and HTTP transport handlers.

### 11. `app/main.py`
- **What it does**: FastAPI application factory providing REST endpoints and static file serving.
- **Key Functions**:
  - `create_app(llm, cache)`: Dependency-injected application factory allowing tests to provide fakes.
  - `POST /api/analyze`: Multipart CSV upload endpoint with 2 MB size cap.
  - `POST /api/reply`: Endpoint for on-demand merchant reply generation.
  - `GET /api/health` and `GET /`: Health check probe and dashboard entry point.
- **Why**: Exposes a clean, RESTful API and serves the dashboard with zero build steps or external dependencies.

### 12. `static/index.html`
- **What it does**: Single-page dashboard built with vanilla HTML5, CSS3, and JavaScript.
- **Key Features**:
  - Drag-and-drop or file browser upload for `reviews.csv`.
  - Summary KPI cards, rating-sentiment agreement card, and run metrics line.
  - Notices for skipped rows (R005) and duplicate rows (R015).
  - Top 3 complaints and praises cards with associated review IDs.
  - Multi-filter table (Product, Sentiment, Topic) working simultaneously.
  - On-click "Suggest reply" button with expandable reply card and copy-to-clipboard functionality.
  - Strict XSS protection using `textContent` for all customer data.
- **Why**: Fast, lightweight, zero build-step UI that runs anywhere offline.

---

## Live Change Drills (Interview Prep)

Here are 6 common modifications an interviewer may ask you to make live on the spot:

### Drill 1: Change the Batch Size
- **Question**: "Can you change the batch size from 5 to 10?"
- **Where to edit**:
  1. Open `app/config.py`: In `Settings`, change `batch_size: int = 10`, or set `BATCH_SIZE=10` in `.env`.
  2. In `app/analyzer.py`: Default argument in `__init__` can be updated: `batch_size: int = 10`.

### Drill 2: Add a New Topic (e.g. "value" or "comfort")
- **Question**: "Store owners want to track a 'comfort' topic for footwear and bedding. How do you add it?"
- **Where to edit**:
  1. Open `app/schema.py`:
     - Add `"comfort"` to `TopicType = Literal[..., "comfort"]`.
     - Add `"comfort"` to `ALLOWED_TOPICS = {..., "comfort"}`.
  2. Open `app/prompts.py`:
     - Add `"comfort"` to the prompt topics list in `ANALYSIS_SYSTEM_PROMPT`.
  3. Open `static/index.html`:
     - Add `<option value="comfort">Comfort</option>` inside the `#filterTopic` select element.
  4. Bump `PROMPT_VERSION = "v1.1"` in `app/prompts.py` so old cached entries without the new topic are invalidated.

### Drill 3: Change the Backoff Timing / Base
- **Question**: "The API rate limits are aggressive. Can you start backoff at 2s instead of 1s and increase max delay to 60s?"
- **Where to edit**:
  1. Open `app/llm.py`:
     - In `backoff_delay(attempt, retry_after)`:
       Change `base = min(60.0, float(2 ** attempt))` (starts at 2s for attempt 1, 4s for attempt 2, etc.) and cap at `60.0`.

### Drill 4: Change the Reply Word Count Limit
- **Question**: "We want longer, more detailed replies (up to 120 words)."
- **Where to edit**:
  1. Open `app/prompts.py`:
     - In `REPLY_SYSTEM_PROMPT`: Change `Maximum length: 70 words.` to `Maximum length: 120 words.`
     - In `build_reply_user_message`: Update prompt text to `(under 120 words)`.

### Drill 5: Add a 'mixed' Sentiment
- **Question**: "Reviews that have both strong praise and complaints should be classified as 'mixed' instead of forcing positive or negative."
- **Where to edit**:
  1. Open `app/schema.py`:
     - Update `SentimentType = Literal["positive", "neutral", "negative", "mixed"]`.
     - Update `ALLOWED_SENTIMENTS = {"positive", "neutral", "negative", "mixed"}`.
  2. Open `app/prompts.py`:
     - In `ANALYSIS_SYSTEM_PROMPT`: Add `"mixed"` to the Sentiment list and explain criteria (reviews having both strong positive and negative aspects).
  3. Open `app/analyzer.py`:
     - In `check_rating_mismatch`: Define expectation for `mixed` (e.g., rating == 3 or rating in [2, 3, 4] depending on policy).
  4. Open `static/index.html`:
     - Add `<option value="mixed">Mixed</option>` to `#filterSentiment`.
     - Add `.badge-mixed { background: rgba(168, 85, 247, 0.15); color: #c084fc; }` in CSS.

### Drill 6: Filter Reviews by Date
- **Question**: "Can we filter the dashboard reviews to only show reviews after a specific date?"
- **Where to edit**:
  1. Open `static/index.html`:
     - Add `<input type="date" id="filterDate" class="filter-select">` next to the filter dropdowns in the table header.
     - In `applyFilters()`:
       ```javascript
       const dVal = document.getElementById("filterDate").value;
       // Inside filtered = currentData.reviews.filter(r => { ... })
       if (dVal && r.date < dVal) return false;
       ```
     - Bind event: `document.getElementById("filterDate").onchange = applyFilters;`.
