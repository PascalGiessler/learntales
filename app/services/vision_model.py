import os

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from loguru import logger


class VisualModel:
    def __init__(self, name: str):
        logger.info("initialize visual model")

    def generate_images(
        self, prompts: list[str], iterations: int = 4
    ) -> dict[str, str]:
        raise NotImplementedError()


class NvidiaFoundationVisionModel(VisualModel):

    def __init__(self, api_key: str, model_name: str = "ai-sdxl-lightning"):
        super().__init__("Nvidia Foundation")
        self.model_name = model_name
        os.environ["NVIDIA_API_KEY"] = api_key

    def generate_images(
        self, prompts: list[str], iterations: int = 1
    ) -> dict[str, str]:
        result = {}
        generator = self._build_image_gen(iterations=iterations)
        for p in prompts:
            try:
                logger.debug(f"generating image for {p} using {self.model_name}")
                model_res = generator.invoke(p)
                base64_img = model_res.response_metadata["artifacts"][0]["base64"]
                result[p] = base64_img
            except Exception as e:
                logger.error(f"failed to generate image due to {e}")
        return result

    def _build_image_gen(self, iterations: int = 4, weight: int = 1):
        img_gen = ChatNVIDIA(model=self.model_name)

        def to_sdxl_payload(d):
            if d:
                d = {
                    "text_prompts": [
                        {
                            "text": f"Create a vibrant cartoon-style illustration for a children's book. "
                            f"The scene should include the following: {d.get("messages", [{}])[0].get("content")}."
                            f" Resulting image should be for a children book",
                            "weight": weight,
                        }
                    ],
                    "steps": iterations,
                }
            logger.info(f"resulting sdxl payload: {d}")
            return d

        img_gen.client.payload_fn = to_sdxl_payload
        return img_gen
