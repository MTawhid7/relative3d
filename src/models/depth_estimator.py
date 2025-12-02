from diffusers import LotusGPipeline
from .base_model import BaseModel

class LotusDepth(BaseModel):
    def _load_model(self):
        # Using the standard Diffusers pipeline for Lotus
        self.pipeline = LotusGPipeline.from_pretrained(
            "jingheya/lotus-depth-g-v2-1-disparity",
            torch_dtype=torch.float16
        ).to(self.device)

    def predict(self, image):
        # Returns 16-bit depth map
        depth = self.pipeline(image).images[0]
        return depth