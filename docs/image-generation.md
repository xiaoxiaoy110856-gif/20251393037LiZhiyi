# Image Generation

The Agent now has two image-generation paths:

- `generate_image_advanced`: the default Agent tool for user-facing image requests. It rewrites the prompt, adds a negative prompt, generates ComfyUI candidates, scores them, and returns the best image.
- `generate_image`: legacy/simple provider path used by `/api/images/generate` and as a fallback for OpenAI or simple ComfyUI generation.

For normal chat requests such as "generate a sports car image", the Agent should call `generate_image_advanced`, not the legacy simple tool.

## Local ComfyUI Generation

Recommended local setup:

```powershell
$env:IMAGE_PROVIDER="comfyui"
$env:LOCAL_ENABLE_COMFYUI="1"
$env:LOCAL_COMFYUI_URL="http://127.0.0.1:8188"
$env:LOCAL_COMFYUI_TIMEOUT_SECONDS="900"
$env:IMAGE_GENERATION_DEFAULT_BATCH_SIZE="4"
$env:IMAGE_GENERATION_MAX_RETRIES="2"
$env:IMAGE_QUALITY_MIN_SCORE="0.75"
```

Start ComfyUI with GPU when CUDA PyTorch is installed:

```powershell
cd "C:\Users\Lenovo\Desktop\人工智能与决策\Rl\PPO\ComfyUI"
D:\Miniconda\python.exe main.py --listen 127.0.0.1 --port 8188
```

Do not add `--cpu` when `torch.cuda.is_available()` is `True`.

## Enable OpenAI Image Generation

```powershell
$env:IMAGE_PROVIDER="openai"
$env:OPENAI_API_KEY="sk-..."
$env:IMAGE_MODEL="gpt-image-2"
$env:IMAGE_OUTPUT_DIR="outputs/generated_images"
$env:IMAGE_DEFAULT_SIZE="1024x1024"
$env:IMAGE_DEFAULT_QUALITY="auto"
$env:IMAGE_DEFAULT_FORMAT="png"
$env:IMAGE_DEFAULT_BACKGROUND="auto"
```

`IMAGE_PUBLIC_BASE_URL` is optional. When it is empty, images are served by the backend at:

```text
/generated-images/<filename>
```

The backend only serves files by basename from the configured output directory, so the URL cannot be used to read arbitrary local files.

## Tool: `generate_image_advanced`

Main Agent tool for image requests.

Parameters:

```json
{
  "prompt": "required text prompt",
  "style": "optional style",
  "preset": "optional preset name",
  "size": "optional widthxheight",
  "batch_size": 1,
  "quality_mode": "fast | balanced | high",
  "allow_retry": true,
  "use_highres_fix": false,
  "notes": "optional extra instructions"
}
```

Return shape:

```json
{
  "type": "image_result",
  "final_image": {
    "id": "uuid",
    "url": "/generated-images/quality_xxx.png",
    "score": 0.85,
    "width": 768,
    "height": 512
  },
  "images": [
    {
      "id": "uuid",
      "url": "/generated-images/quality_xxx.png",
      "score": 0.85
    }
  ],
  "generation_plan": {
    "positive_prompt": "...",
    "negative_prompt": "...",
    "preset": "automotive_photorealistic",
    "parameters": {}
  },
  "quality_report": {},
  "retries": []
}
```

Implementation:

- `backend/image_quality.py`: prompt rewriting, presets, scoring, retry controller.
- `backend/comfyui_workflow.py`: ComfyUI workflow patching and runner.
- `backend/tool_registry.py`: tool registration.
- `backend/agent_loop.py`: image-intent routing.

## Tool: `generate_image`

Legacy/simple path and OpenAI provider path.

Parameters:

```json
{
  "prompt": "required text prompt",
  "size": "1024x1024 | 1024x1536 | 1536x1024 | auto",
  "quality": "low | medium | high | auto",
  "format": "png | jpeg | webp",
  "background": "transparent | opaque | auto",
  "n": 1,
  "style_notes": "optional composition, lighting, color, style details",
  "user_visible_prompt": "optional original user prompt"
}
```

Return shape:

```json
{
  "type": "image_result",
  "images": [
    {
      "id": "uuid",
      "url": "/generated-images/image_20260513_xxx.png",
      "path": "server local path",
      "mime_type": "image/png",
      "size": "1024x1024",
      "quality": "auto",
      "format": "png",
      "prompt": "final prompt",
      "created_at": "ISO timestamp"
    }
  ],
  "provider": "openai",
  "model": "gpt-image-2"
}
```

## Frontend Display

The chat renderer supports Markdown image syntax in assistant messages:

```markdown
![generated image](/generated-images/image_xxx.png)
```

The side-panel image generator also displays the first returned image and links to the full image.

In development mode, Vite proxies both:

```text
/api -> http://127.0.0.1:8765
/generated-images -> http://127.0.0.1:8765
```

So inline chat images can load directly from `/generated-images/...`.

## Current Limits

- Text-to-image is implemented.
- Advanced Agent image requests use `generate_image_advanced`.
- Image editing is reserved through `ImageGenerationService.edit_image`, but it currently raises `NotImplementedError`.
- No API key is stored in code or returned to the client.
- `n` is limited to 4.
- Filenames are UUID based and never use the user prompt.

## Common Errors

- `OPENAI_API_KEY_MISSING`: set `OPENAI_API_KEY`.
- `IMAGE_INVALID_SIZE`: use `1024x1024`, `1024x1536`, `1536x1024`, or `auto`.
- `IMAGE_INVALID_FORMAT`: use `png`, `jpeg`, or `webp`.
- `OPENAI_IMAGE_RATE_LIMIT`: wait or check account limits.
- `IMAGE_EMPTY_RESPONSE`: provider returned no `b64_json`.

## Provider Switching

Set:

```powershell
$env:IMAGE_PROVIDER="openai"
```

The external API shape stays the same where possible.
