# Image Quality Pipeline

This project now has an advanced ComfyUI image generation path for Agent image requests.

The goal is to avoid sending the raw user prompt directly to ComfyUI. The pipeline rewrites the request into a structured `ImageGenerationPlan`, adds a negative prompt, generates multiple candidates, scores them, retries when the best candidate is below the quality gate, and returns the best image to the chat UI.

After the backend consolidation, the quality pipeline is intentionally concentrated in:

- `backend/image_quality.py`: plan types, presets, negative prompts, rewriter, critic, quality controller, and `generate_image_advanced`.
- `backend/comfyui_workflow.py`: ComfyUI API workflow builder/patcher/runner.
- `backend/tool_registry.py`: Agent tool registration.
- `backend/agent_loop.py`: direct image-intent routing to the advanced tool.

## Environment Variables

```powershell
$env:IMAGE_PROVIDER="comfyui"
$env:LOCAL_COMFYUI_URL="http://127.0.0.1:8188"
$env:LOCAL_COMFYUI_TIMEOUT_SECONDS="900"
$env:IMAGE_GENERATION_DEFAULT_BATCH_SIZE="4"
$env:IMAGE_GENERATION_MAX_RETRIES="2"
$env:IMAGE_QUALITY_MIN_SCORE="0.75"
$env:IMAGE_OUTPUT_DIR="./outputs/generated_images"
```

`LOCAL_COMFYUI_TIMEOUT_SECONDS` should be long enough for the selected model and GPU. CPU generation may need several minutes; GPU generation is strongly recommended.

## Tool: `generate_image_advanced`

Parameters:

```json
{
  "prompt": "string",
  "style": "string optional",
  "preset": "string optional",
  "size": "string optional",
  "batch_size": "number optional",
  "quality_mode": "fast | balanced | high",
  "allow_retry": "boolean optional",
  "use_highres_fix": "boolean optional",
  "reference_image_id": "string optional",
  "notes": "string optional"
}
```

Return shape:

```json
{
  "type": "image_result",
  "final_image": {
    "id": "string",
    "url": "/generated-images/quality_xxx.png",
    "score": 0.85,
    "width": 768,
    "height": 512
  },
  "candidates": [
    {
      "id": "string",
      "url": "/generated-images/quality_xxx.png",
      "score": 0.85
    }
  ],
  "generation_plan": {
    "positive_prompt": "string",
    "negative_prompt": "string",
    "preset": "automotive_photorealistic",
    "parameters": {}
  },
  "quality_report": {},
  "retries": []
}
```

The response does not expose server absolute paths. The chat renderer displays the returned `/generated-images/...` URL inline.

## Presets

Preset configuration lives in:

```text
backend/image_quality.py
```

Current presets:

- `automotive_photorealistic`
- `product_photography`
- `portrait_photorealistic`
- `cinematic_scene`
- `poster_design`
- `logo_draft`
- `interior_design`
- `general_photorealistic`

Each preset controls default size, batch size, steps, CFG, sampler, scheduler, negative prompt, quality checks, and retry strategy. Unknown preset names fall back to `general_photorealistic`.

## ImageGenerationPlan Example

For:

```text
A photorealistic red Lamborghini-style supercar parked on a wet city street at night
```

The automotive preset rewrites the prompt roughly as:

```text
A single red exotic supercar inspired by Italian high-performance design, one car only, parked alone or shown as the only main vehicle. Low-angle front three-quarter view, closed doors, intact body panels, realistic proportions, detailed headlights, realistic wheels, clean aerodynamic lines. Cinematic lighting, glossy reflections, high-end automotive photography, sharp focus, photorealistic, highly detailed, no people, no text, no watermark, no other vehicles.
```

Negative prompt:

```text
multiple cars, extra vehicles, duplicate car, background car, open doors, broken body, deformed body, malformed wheels, extra wheels, missing wheels, distorted perspective, bad reflections, mirrored vehicle, floating parts, disassembled car, unrealistic car structure, extra headlights, extra mirrors, people, text, watermark, logo
```

## Workflow Configuration

The first version uses a generated ComfyUI API workflow compatible with SD checkpoints:

- `CheckpointLoaderSimple`
- `CLIPTextEncode` positive prompt
- `CLIPTextEncode` negative prompt
- `EmptyLatentImage`
- `KSampler`
- `VAEDecode`
- `SaveImage`

The patcher is in:

```text
backend/comfyui_workflow.py
```

It supports node mapping for future external workflow JSON templates. If you export a workflow from ComfyUI, use the API format and map these node roles:

```json
{
  "positive_prompt": "6",
  "negative_prompt": "7",
  "sampler": "3",
  "checkpoint": "4",
  "latent": "5",
  "save_image": "9"
}
```

## Scoring And Retry

`ImageCritic` currently lives in `backend/image_quality.py` and implements a lightweight rule-based evaluator:

- generated file exists and is non-empty
- image size is usable
- plan includes single-subject constraints
- negative prompt includes text/watermark bans

When the best candidate is below the quality gate, `ImageGenerationQualityController` repairs the plan by adding stronger single-subject and no-duplicate constraints, lowering CFG when useful, increasing steps, and retrying up to `IMAGE_GENERATION_MAX_RETRIES`.

This first version intentionally does not add a heavy VLM dependency. A future VLM critic can plug into `ImageCritic.evaluate_image`.

## High-res Fix

`ImageGenerationPlan` includes a `use_highres_fix` and `highres` configuration. The first version keeps the interface ready, but does not require a second ComfyUI high-res workflow. If high-res workflow JSON is added later, connect it through `ComfyUIWorkflowRunner` and reuse the same artifact structure.

## LoRA And Training Extension

`ImageGenerationPlan` supports a `loras` field, and `TrainingPlan` is reserved for future LoRA training configuration. This version does not train models. To improve a domain such as cars, add a compatible LoRA to ComfyUI and extend the workflow patcher to insert `LoraLoader` nodes.

## Tests

```powershell
python -m unittest tests.test_image_quality_pipeline tests.test_image_generation
```

Tests use fake runners and fake providers. They do not call ComfyUI or OpenAI.

## Current Limitations

- Rule-based scoring cannot truly inspect whether a generated image contains two cars or malformed wheels.
- VLM scoring, object detection, ControlNet, IP-Adapter, mask editing, and LoRA training are not enabled yet.
- High-res fix is represented in the plan but needs a configured ComfyUI high-res workflow before becoming active.
