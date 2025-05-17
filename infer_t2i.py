from localdit.pipeline_t2i import LocalDiTTxt2ImgPipeline

pipe = LocalDiTTxt2ImgPipeline.from_pretrained("path/to/LocalDiT-model.pt")
pipe = pipe.to("cuda")
prompt = "Nighttime alpine lodge with warm interior lights, Milky Way arch above"
image = pipe(
    prompt=prompt,
    height=576,
    width=1024,
    target_height=1152,
    target_width=2048,
    num_inference_steps=20,
    refinement_steps=20
).images[0]