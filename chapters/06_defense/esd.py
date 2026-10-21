from pathlib import Path; import torch
from diffusers import StableDiffusionPipeline
device = 'cuda'; torch.manual_seed(2026)
pipe = StableDiffusionPipeline.from_pretrained(
    'stable-diffusion-v1-5/stable-diffusion-v1-5').to(device)
teacher = type(pipe.unet).from_config(pipe.unet.config).to(device)
teacher.load_state_dict(pipe.unet.state_dict()); teacher.requires_grad_(False)
pipe.vae.requires_grad_(False); pipe.text_encoder.requires_grad_(False)
pipe.unet.requires_grad_(False)
params = [p for n,p in pipe.unet.named_parameters() if 'attn2' in n]
for p in params: p.requires_grad_(True)
optim = torch.optim.Adam(params, lr=1e-5)
concept = 'a landscape painting in the style of Van Gogh'
out = Path('results/esd'); out.mkdir(parents=True, exist_ok=True)
def draw(name):
    pipe(concept, generator=torch.Generator(device).manual_seed(2026),
         num_inference_steps=30).images[0].save(out/name)
draw('before.png')
positive, negative = pipe.encode_prompt(concept, device, 1, True)
for step in range(200):
    with torch.no_grad():
        clean = pipe(concept, num_inference_steps=20, output_type='latent').images
        t = torch.randint(1, 1000, (1,), device=device)
        z = pipe.scheduler.add_noise(clean, torch.randn_like(clean), t)
        uc = teacher(z,t,encoder_hidden_states=negative).sample
        cc = teacher(z,t,encoder_hidden_states=positive).sample
        target = uc - (cc-uc)
    loss = (pipe.unet(z,t,encoder_hidden_states=positive).sample-target).square().mean()
    optim.zero_grad(); loss.backward(); optim.step(); print(step, float(loss))
draw('after.png'); pipe.save_pretrained(out/'edited_model')
