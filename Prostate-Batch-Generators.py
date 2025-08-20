    def generate_prostate_specific(self):
        """Generate prostate-specific medical images"""
        prostate_prompts = [
            "prostate MRI scan, medical imaging, pelvic anatomy, diagnostic",
            "prostate ultrasound medical scan, transrectal imaging",
            "prostate biopsy medical image, pathology, microscopic",
            "prostate anatomical diagram, medical illustration, male anatomy",
            "prostate CT scan, pelvic medical imaging, diagnostic radiology"
        ]

        images = []
        for i, prompt in enumerate(prostate_prompts):
            print(f"Generating prostate image {i+1}/{len(prostate_prompts)}")

            with torch.no_grad():
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    num_inference_steps=35,
                    height=512,
                    width=512,
                    guidance_scale=9.0
                ).images[0]

            images.append(image)
            filename = f"prostate_medical_{i+1}.png"
            image.save(filename)
            print(f"Saved: {filename}")

        return images

    def batch_generate(self, categories=None, images_per_category=2):
        """Generate multiple types of medical images"""
        if categories is None:
            categories = list(self.medical_prompts.keys())

        all_images = {}

        for category in categories:
            print(f"\n=== Generating {category.upper()} images ===")
            images = self.generate_medical_image(
                scan_type=category,
                num_images=images_per_category
            )
            all_images[category] = images

        return all_images
