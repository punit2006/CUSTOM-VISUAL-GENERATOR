    def generate_medical_image(self, scan_type="xray", custom_prompt=None,
                               num_images=1, height=512, width=512):
        """
        Generate medical images

        Args:
            scan_type: Type of medical scan (xray, mri, ct, ultrasound, microscopic, anatomical)
            custom_prompt: Custom prompt (overrides scan_type)
            num_images: Number of images to generate
            height, width: Image dimensions
        """

        if custom_prompt:
            prompts = [custom_prompt]
        else:
            if scan_type not in self.medical_prompts:
                print(f"Unknown scan type: {scan_type}")
                print(f"Available types: {list(self.medical_prompts.keys())}")
                return None
            prompts = self.medical_prompts[scan_type][:num_images]

        generated_images = []

        for i, prompt in enumerate(prompts):
            print(f"Generating image {i+1}/{len(prompts)}")
            print(f"Prompt: {prompt}")

            with torch.no_grad():
                image = self.pipe(
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    num_inference_steps=30,
                    height=height,
                    width=width,
                    guidance_scale=8.0
                ).images[0]

            generated_images.append(image)

            # Save image
            filename = f"medical_{scan_type}_{i+1}.png"
            image.save(filename)
            print(f"Saved: {filename}")

        return generated_images
