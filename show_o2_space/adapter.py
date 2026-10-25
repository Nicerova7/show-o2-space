"""Small bridge to the original Show-o/show-o2 implementation.

Sequence construction follows upstream inference_mmu.py and inference_t2i.py.
The upstream code remains in a separate, pinned checkout.
"""
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
from models.misc import get_text_tokenizer, prepare_gen_input
import transport

UPSTREAM_REVISION = "45a5a2de01d1ebd10cd5864d29310a76476cdf23"
MODEL_ID = "showlab/show-o2-1.5B"


class ShowO2Adapter:
    def __init__(self, vae_path=None):
        if not torch.cuda.is_available():
            raise RuntimeError("Select a GPU runtime in Colab before loading Show-o2.")
        self.device = torch.device("cuda")
        # T4 does not support BF16: use FP32 weights with FP16 autocast there.
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.transport = transport
        self.prepare_gen_input = prepare_gen_input
        self.mask = omni_attn_mask_naive
        self.tokenizer, self.token_ids = get_text_tokenizer(
            "Qwen/Qwen2.5-1.5B-Instruct", return_showo_token_ids=True, llm_name="qwen2_5"
        )
        self.model = Showo2Qwen2_5.from_pretrained(
            MODEL_ID, use_safetensors=False
        ).to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)
        # Dense masks use SDPA rather than the optional compiled FlexAttention path.
        self.model.showo.config.use_cache = False
        if self.model.config.image_latent_height != 27 or not self.model.config.add_time_embeds:
            raise ValueError("This adapter expects the original 432x432 checkpoint with time embeddings.")
        if vae_path is None:
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
            if hasattr(self.model, "time_embed_proj"):
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

    def forward_understanding(self, embeddings, attention_mask, activation_gradients=False):
        """Differentiable prefill for later experiments; parameters remain frozen.

        Caller chooses a scalar objective from the returned logits and calls
        backward(). Use Capture(..., keep_graph=True) to retain layer gradients.
        """
        with torch.set_grad_enabled(activation_gradients), self.autocast():
            inputs = embeddings.detach().requires_grad_(activation_gradients)
            return self.model.showo(
                inputs_embeds=inputs, attention_mask=attention_mask,
                output_hidden_states=False, use_cache=False,
            )

    @torch.no_grad()
    def answer(self, embeddings, attention_mask, max_new_tokens=128):
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        with self.autocast():
            tokens = self.model.mmu_generate(
                input_embeds=embeddings, attention_mask=attention_mask,
                top_k=1, max_new_tokens=max_new_tokens, eos_token=self.tokenizer.eos_token_id,
            )
        ids = torch.stack(tokens).reshape(-1).cpu().tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @torch.no_grad()
    def generate(self, prompt, steps=50, guidance=5.0, seed=42):
        if steps < 2 or guidance < 0:
            raise ValueError("steps must be >= 2 and guidance must be >= 0")
        torch.manual_seed(seed)
        ids = self.token_ids
        tokens, null_tokens, positions, null_positions = self.prepare_gen_input(
            [prompt], self.tokenizer, 730, ids["bos_id"], ids["eos_id"],
            ids["boi_id"], ids["eoi_id"], self.tokenizer.pad_token_id,
            ids["img_pad_id"], 290, self.device,
        )
        noise = torch.randn((1, 16, 54, 54), device=self.device, dtype=self.dtype)
        if guidance > 0:
            noise = torch.cat([noise, noise])
            tokens = torch.cat([tokens, null_tokens])
            positions = torch.cat([positions, null_positions])
        mask = self.mask(tokens.shape[0], 1024, positions, self.device).to(self.dtype)
        transport = self.transport.create_transport(
            path_type="Linear", prediction="velocity", loss_weight=None,
            train_eps=None, sample_eps=None, snr_type="lognorm", do_shift=True, seq_len=730,
        )
        sample = self.transport.Sampler(transport).sample_ode(
            sampling_method="euler", num_steps=steps, atol=1e-6, rtol=1e-3,
            reverse=False, time_shifting_factor=3.0,
        )
        with self.autocast():
            latents = sample(
                noise, self.model.t2i_generate, text_tokens=tokens, attention_mask=mask,
                modality_positions=positions, output_hidden_states=True,
                max_seq_len=1024, guidance_scale=guidance,
            )[-1][:1]
            pixels = self.vae.batch_decode(latents.unsqueeze(2)).squeeze(2)[0]
        pixels = ((pixels.float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0)
        return Image.fromarray(pixels.round().byte().cpu().numpy())
