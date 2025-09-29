from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

class Generator:
    """
    Generate new content from retrieved document segments using PHI-3 vision model.
    """

    def __init__(self, model_name="tiiuae/phi-3-xx"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_from_text(self, text_segments: list[str], max_length: int = 200) -> str:
        """
        Generate content based on a list of text segments.
        """
        prompt = "\n".join(text_segments)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_length)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def generate_from_images(self, image_paths: list[str], max_length: int = 200) -> str:
        """
        Generate textual description or content based on images.
        PHI-3 vision model can accept images as input.
        """
        images = [Image.open(p).convert("RGB") for p in image_paths if p]
        if not images:
            return ""
        # Assuming model supports .generate_from_images (pseudo-code, adjust as per API)
        outputs = self.model.generate_from_images(images, max_new_tokens=max_length)
        return outputs
