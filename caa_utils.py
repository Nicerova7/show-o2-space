"""Utilities for the Show-o2 1.5B CAA understanding pilot (BF16 GPU)."""
from contextlib import contextmanager

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from models.misc import get_text_tokenizer

MODEL_ID = "showlab/show-o2-1.5B"


class ShowO2Adapter:
    def __init__(self):
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.autocast_dtype = torch.bfloat16
        self.mask = omni_attn_mask_naive
        self.tokenizer, self.token_ids = get_text_tokenizer(
            "Qwen/Qwen2.5-1.5B-Instruct", return_showo_token_ids=True, llm_name="qwen2_5"
        )
        self.model = Showo2Qwen2_5.from_pretrained(
            MODEL_ID, use_safetensors=False
        ).to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)
        # Dense masks use SDPA rather than the optional compiled FlexAttention path.
        self.model.showo.config.use_cache = False
        vae_path = hf_hub_download("Wan-AI/Wan2.1-T2V-14B", "Wan2.1_VAE.pth")
        self.vae = WanVAE(vae_pth=str(vae_path), dtype=self.dtype, device=self.device)
        self.layers = self.model.showo.model.layers

    def autocast(self):
        return torch.autocast("cuda", dtype=self.autocast_dtype)

    @torch.no_grad()
    def prepare_understanding(self, image_path, question):
        """Return embeddings, additive attention mask, and sequence alignment.

        Match upstream's shortest-side resize, 432 center crop, and [-1, 1]
        normalization. Objects outside the center crop will not be visible.
        """
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(432, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((432, 432)), transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        pixels = transform(image).unsqueeze(0).to(self.device)
        with self.autocast():
            latents = self.vae.sample(pixels.unsqueeze(2)).squeeze(2).to(self.dtype)
            semantic = self.model.image_embedder_und(latents)
            semantic = semantic + self.model.position_embedding(self.model.image_position_ids)
            semantic = self.model.und_trans(semantic)["last_hidden_state"]
            generative = self.model.image_embedder_gen(latents)
            visual = self.model.fusion_proj(torch.cat([semantic, generative], dim=-1))
            encode = lambda text: self.tokenizer(text, add_special_tokens=False)["input_ids"]
            a = ([self.token_ids["bos_id"]]
                 + encode("system\nYou are a helpful assistant.<|im_end|>")
                 + encode("\n<|im_start|>user\n"))
            b = ([self.token_ids["boi_id"], self.token_ids["eoi_id"]]
                 + encode(question) + encode("\n<|im_start|>assistant\n"))
            embed = self.model.showo.model.embed_tokens
            text_a = embed(torch.tensor([a], device=self.device))
            text_b = embed(torch.tensor([b], device=self.device))
            time = self.model.time_embed(torch.tensor([[1.0]], device=self.device), text_a.dtype)
            time = self.model.time_embed_proj(time)
            embeddings = torch.cat([text_a, text_b[:, :1], time, visual, text_b[:, 1:]], dim=1)
        # Preserve the upstream modality offset exactly (see inference_mmu.py).
        positions = torch.tensor([[[len(a) + 2, visual.shape[1] + 1]]], device=self.device)
        mask = self.mask(1, embeddings.shape[1], positions, self.device, inverted=True).to(self.dtype)
        alignment = {
            "sequence_length": embeddings.shape[1],
            "time_position": len(a) + 1,
            "image_span": [len(a) + 2, len(a) + 2 + visual.shape[1]],
            "modality_positions": positions.cpu().tolist(),
            "token_ids": a + [b[0], -1] + [-1] * visual.shape[1] + b[1:],
            "span_convention": "zero-based, end-exclusive; -1 denotes non-text embedding",
        }
        return embeddings.detach(), mask, alignment

    @torch.no_grad()
    def answer(self, embeddings, attention_mask, max_new_tokens=128):
        with self.autocast():
            tokens = self.model.mmu_generate(
                input_embeds=embeddings, attention_mask=attention_mask,
                top_k=1, max_new_tokens=max_new_tokens, eos_token=self.tokenizer.eos_token_id,
            )
        ids = torch.stack(tokens).reshape(-1).cpu().tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=True)


def contrast_tokens(tokenizer, question, positive, negative, pair_index):
    """Alternate A/B labels; keep each pair identical until its answer letter."""
    positive_index = pair_index % 2
    options = [None, None]
    options[positive_index] = positive
    options[1 - positive_index] = negative
    question = f"{question}\n\n(A) {options[0]}\n(B) {options[1]}"
    prefix = tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )
    letters = ["AB"[positive_index], "AB"[1 - positive_index]]
    # No closing parenthesis, EOS, or chat terminator after the answer letter.
    return {
        name: tokenizer(prefix + letter, add_special_tokens=False,
                        return_tensors="pt")["input_ids"]
        for name, letter in zip(["positive", "negative"], letters)
    }


@contextmanager
def capture_answer_letter(layers, layer):
    """Output residual stream of a zero-based decoder block, before final norm."""
    captured = {}

    def save(module, args, output):
        captured["activation"] = output[0][0, -1, :].detach().float().cpu().clone()

    handle = layers[layer].register_forward_hook(save)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def steer_understanding(layers, layer, vector, coefficient, input_length):
    """For upstream's full-sequence, no-KV-cache mmu_generate sampler."""
    start = input_length - 1
    trace = []

    def add(module, args, output):
        hidden = output[0].clone()
        hidden[:, start:, :] += coefficient * vector.to(hidden)
        trace.append({"sequence_length": hidden.shape[1],
                      "first_position": start,
                      "last_position": hidden.shape[1] - 1,
                      "positions": hidden.shape[1] - start})
        return (hidden, *output[1:])

    handle = layers[layer].register_forward_hook(add)
    try:
        yield trace
    finally:
        handle.remove()
