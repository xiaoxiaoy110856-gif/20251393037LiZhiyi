# Context Compression

Long conversations slow down local model calls because the whole history can grow beyond the useful context window. This project now keeps raw messages intact in MySQL while model calls use a dynamic context package:

1. rolling conversation state
2. relevant segment summaries
3. relevant old raw messages when useful
4. recent messages
5. current user message

Original `chat_messages` rows are never deleted or overwritten by compression.

## New MySQL Tables

### `conversation_summaries`

Stores segment, rolling, or manual summaries.

Important fields:

- `conversation_id`
- `summary_type`
- `start_message_id`
- `end_message_id`
- `content`
- `structured_json`
- `token_count`
- `source_message_ids`
- `model`
- `version`

### `conversation_states`

Stores rolling state for a session:

- current goal
- user requirements
- decisions
- constraints
- open tasks
- important entities
- user preferences
- files or artifacts
- `last_compressed_message_id`

### `context_build_logs`

Debug table recording ids selected for one model request:

- recent message ids
- summary ids
- relevant old message ids
- total estimated input tokens

`chat_messages` also gets a `token_count` column.

## Environment Variables

```powershell
$env:CONTEXT_MAX_INPUT_TOKENS="12000"
$env:CONTEXT_RECENT_MESSAGE_COUNT="12"
$env:CONTEXT_COMPRESSION_TRIGGER_TOKENS="8000"
$env:CONTEXT_SEGMENT_MESSAGE_COUNT="30"
$env:CONTEXT_SUMMARY_MAX_TOKENS="1500"
$env:CONTEXT_RETRIEVAL_MAX_ITEMS="8"
$env:CONTEXT_RETRIEVAL_MAX_TOKENS="2500"
$env:CONTEXT_ENABLE_COMPRESSION="true"
$env:CONTEXT_ENABLE_BUILD_LOG="true"
```

Set `CONTEXT_ENABLE_COMPRESSION=false` to fall back to the previous recent-history behavior with token trimming.

## Compression Trigger

Compression runs defensively:

- before a new chat request is assembled
- after an assistant reply is saved

If compression fails, the chat request continues. The failure is logged without dumping private content.

## Context Assembler Order

The assembler builds model input in this order:

1. system / Agent prompt from the existing LLM or Agent path
2. `[Conversation rolling state]`
3. relevant segment summaries
4. relevant old raw messages
5. recent messages
6. current user message, appended by the existing caller

When the budget is tight, relevant raw messages are skipped first, then summaries. Recent messages and the current user message have priority.

Core implementation files:

- `backend/context_assembler.py`: builds the model context for one request.
- `backend/conversation_compressor.py`: creates segment summaries and updates rolling state.
- `backend/context_repositories.py`: reads/writes summaries, states, and build logs.
- `backend/context_state.py`: merges rolling state.
- `backend/relevant_context.py`: keyword-based relevance retrieval.
- `backend/context_utils.py`: token estimation, compression prompt, and sensitive-data redaction.

## Relevance Retrieval

The first version uses keyword scoring over:

- `conversation_summaries.content`
- `conversation_summaries.structured_json`
- older `chat_messages.content`

It avoids pulling all old messages back into the prompt. Future versions can replace this with embeddings or a vector store.

## Sensitive Data

The compressor uses `redact_sensitive_text` from `backend/context_utils.py` before saving summaries:

- API keys
- tokens
- passwords
- cookies
- private keys

Raw messages remain unchanged in MySQL for auditability, but summaries and logs should not contain secrets.

## Migration

Run:

```powershell
python scripts\init_context_compression.py
```

This calls the existing `init_db()` flow and creates the new tables if MySQL is enabled.

## Tests

```powershell
python -m unittest tests.test_context_compression
```

Run all current backend tests:

```powershell
python -m unittest tests.test_context_compression tests.test_workspace_tools tests.test_image_generation tests.test_image_quality_pipeline
```

## Future Work

- Embedding/vector retrieval for old messages and summaries.
- Background compression queue.
- Multi-model summarization policy.
- Stronger tool-call pair expansion for providers with native tool-call message objects.
