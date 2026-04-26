import replicate
import os
from dotenv import load_dotenv

load_dotenv()

def generate_ai_makeup(source_path):

    output = replicate.run(
        "fofr/face-to-many:a07f252abbbd832009640b27f063ea52d87d7a23a185ca165bec23b5adc8deaf",
        input={
            "image": open(source_path, "rb"),
            "prompt": "dramatic beauty makeup, bold lipstick, smokey eyeshadow, eyeliner, blush, highlighter, professional makeup",
            "num_inference_steps": 28,
            "guidance_scale": 8
        }
    )

    return output[0]