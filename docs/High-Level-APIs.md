# High-level APIs

## AutoModel

Use the `AutoLigerKernelForCausalLM` when you want a single, high-level API that mirrors Hugging Face's `AutoModelForCausalLM` but with Liger Kernel integrations applied.

| What | Value |
|------|-------|
| API | `liger_kernel.transformers.AutoLigerKernelForCausalLM` |
| Purpose | Extends `transformers.AutoModelForCausalLM` to integrate Liger Kernel implementation details while keeping the familiar HF API surface. |

::: liger_kernel.transformers.AutoLigerKernelForCausalLM
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

!!! Example "Try it Out"
    You can experiment as shown in this example [here](https://github.com/linkedin/Liger-Kernel?tab=readme-ov-file#1-use-autoligerkernelforcausallm).

---

## Patching APIs (per-model helpers)

Use the patching helpers when you want to apply Liger Kernel changes to an existing, specific model architecture (recommended when you need fine-grained control over which kernels/ops are applied).

### How the table is organized
- **Model**: target model family or variant.
- **API**: function to call to patch a loaded model instance.
- **Supported operations**: shorthand of Liger Kernel features enabled by the patch.

| Model family / variant | Patching API | Supported operations (summary) |
|------------------------|--------------|--------------------------------|
| **Llama4** (text & multimodal) | `liger_kernel.transformers.apply_liger_kernel_to_llama4` | RMSNorm, LayerNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| **LLaMA 2 & 3** | `liger_kernel.transformers.apply_liger_kernel_to_llama` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| **LLaMA 3.2-Vision** | `liger_kernel.transformers.apply_liger_kernel_to_mllama` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| **Mistral** | `liger_kernel.transformers.apply_liger_kernel_to_mistral` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| **Mixtral** | `liger_kernel.transformers.apply_liger_kernel_to_mixtral` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| **Gemma1 / Gemma2** | `liger_kernel.transformers.apply_liger_kernel_to_gemma` / `..._to_gemma2` | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| **Gemma3** (text) | `liger_kernel.transformers.apply_liger_kernel_to_gemma3_text` | RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| **Gemma3** (multimodal) | `liger_kernel.transformers.apply_liger_kernel_to_gemma3` | LayerNorm, RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| **Paligemma / Paligemma2** | `liger_kernel.transformers.apply_liger_kernel_to_paligemma` | LayerNorm, RoPE, RMSNorm, GeGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| **Qwen2 / Qwen2.5 / QwQ** | `liger_kernel.transformers.apply_liger_kernel_to_qwen2` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| **Qwen2-VL / QVQ** | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_vl` | RMSNorm, LayerNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| **Qwen2.5-VL** | `liger_kernel.transformers.apply_liger_kernel_to_qwen2_5_vl` | RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| **Qwen3** | `liger_kernel.transformers.apply_liger_kernel_to_qwen3` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| **Qwen3 MoE** | `liger_kernel.transformers.apply_liger_kernel_to_qwen3_moe` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| **Phi3 & Phi3.5** | `liger_kernel.transformers.apply_liger_kernel_to_phi3` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| **Granite 3.0 & 3.1** | `liger_kernel.transformers.apply_liger_kernel_to_granite` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss |
| **OLMo2** | `liger_kernel.transformers.apply_liger_kernel_to_olmo2` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |
| **GLM-4** | `liger_kernel.transformers.apply_liger_kernel_to_glm4` | RoPE, RMSNorm, SwiGLU, CrossEntropyLoss, FusedLinearCrossEntropy |

> Tip: prefer the per-model patch when you need the lowest-level control (for example, changing which norms or activation kernels are swapped in); use the `AutoLigerKernelForCausalLM` for convenience.

---

## Supported operations (legend)

- **RoPE** — Rotary positional embeddings
- **RMSNorm / LayerNorm** — Normalization kernels replaced or fused
- **GeGLU / SwiGLU** — Activation variants supported by the patch
- **CrossEntropyLoss / FusedLinearCrossEntropy** — Loss and fused linear + loss kernels for fast training/inference

---

## Function signatures (inline docs)

Below are the autodoc blocks for each of the patching helpers and the functions that expose the API surface. These will expand into signatures, docstrings, and source (depending on your MkDocs autodoc plugin settings).

::: liger_kernel.transformers.apply_liger_kernel_to_llama4
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

::: liger_kernel.transformers.apply_liger_kernel_to_llama
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

::: liger_kernel.transformers.apply_liger_kernel_to_mllama
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

::: liger_kernel.transformers.apply_liger_kernel_to_mistral
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

::: liger_kernel.transformers.apply_liger_kernel_to_mixtral
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

::: liger_kernel.transformers.apply_liger_kernel_to_gemma
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

::: liger_kernel.transformers.apply_liger_kernel_to_gemma2
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

::: liger_kernel.transformers.apply_liger_kernel_to_gemma3
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

::: liger_kernel.transformers.apply_liger_kernel_to_gemma3_text
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

::: liger_kernel.transformers.apply_liger_kernel_to_paligemma
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

::: liger_kernel.transformers.apply_liger_kernel_to_qwen2
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

::: liger_kernel.transformers.apply_liger_kernel_to_qwen2_vl
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

::: liger_kernel.transformers.apply_liger_kernel_to_qwen3
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

::: liger_kernel.transformers.apply_liger_kernel_to_qwen3_moe
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

::: liger_kernel.transformers.apply_liger_kernel_to_phi3
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

::: liger_kernel.transformers.apply_liger_kernel_to_granite
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

::: liger_kernel.transformers.apply_liger_kernel_to_olmo2
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

::: liger_kernel.transformers.apply_liger_kernel_to_glm4
    options:
      extra:
        show_docstring: true
        show_signature: true
        show_source: true

---
