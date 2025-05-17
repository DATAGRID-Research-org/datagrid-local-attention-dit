from PIL import Image
from localdit.pipeline_i2i import LocalDiTImg2ImgPipeline

init_image:Image = ...
i2i_pipe = LocalDiTImg2ImgPipeline.from_pretrained("path/to/LocalDiT-model.pt")
i2i_pipe = i2i_pipe.to("cuda")
image = i2i_pipe(
    prompt="A snowy mountain landscape with aurora borealis",
    image=init_image,
    strength=0.7,
    num_inference_steps=30
).images[0]