from diffusers import PixArtAlphaPipeline
from diffusers.models.attention import Attention

class LocalDiTImg2ImgPipeline(PixArtAlphaPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)
    