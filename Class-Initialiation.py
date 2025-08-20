from .config import NEGATIVE_PROMPT

class MedicalImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        """Initialize the medical image generator"""
        print("Loading Stable Diffusion model...")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = self.pipe.to(device)
        print(f"Model loaded on {device}")

        # Optimized medical prompts that work well
        self.medical_prompts = {
            "xray": [
                "chest x-ray medical scan, black and white, clinical diagnostic",
                "bone x-ray radiograph, medical imaging, skeletal structure",
                "dental x-ray scan, medical radiography, teeth and jaw",
                "spine x-ray medical image, vertebrae, diagnostic radiology"
            ],

            "mri": [
                "MRI brain scan, medical imaging, grayscale, cross-section",
                "MRI abdominal scan, medical diagnostic image, internal organs",
                "cardiac MRI scan, heart medical imaging, diagnostic",
                "knee MRI medical scan, joint imaging, orthopedic"
            ],

            "ct": [
                "CT scan cross-section, medical imaging, anatomical structure",
                "chest CT scan, lungs medical imaging, diagnostic radiology",
                "abdominal CT scan, medical diagnostic image, organs",
                "head CT scan, brain medical imaging, neurological"
            ],

            "ultrasound": [
                "ultrasound medical scan, grayscale, diagnostic imaging",
                "cardiac ultrasound, heart medical scan, echocardiogram",
                "abdominal ultrasound, medical diagnostic image",
                "prenatal ultrasound scan, medical imaging"
            ],

            "microscopic": [
                "medical microscopy image, cellular structure, pathology",
                "histology medical slide, tissue sample, microscopic",
                "blood cells microscopic view, medical laboratory",
                "cancer cells microscopic image, pathology slide"
            ],

            "anatomical": [
                "anatomical diagram, medical illustration, human body",
                "skeletal system medical diagram, bones anatomy",
                "cardiovascular system anatomy, medical illustration",
                "nervous system anatomical diagram, medical education"
            ]
        }

        self.negative_prompt = NEGATIVE_PROMPT
