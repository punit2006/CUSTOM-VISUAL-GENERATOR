from medical_image_generator.generator import MedicalImageGenerator

if __name__ == "__main__":
    # Initialize generator
    generator = MedicalImageGenerator()

    # Option 1: Generate specific type
    print("=== Generating X-ray images ===")
    xray_images = generator.generate_medical_image("xray", num_images=3)

    # Option 2: Generate prostate-specific images
    print("\n=== Generating prostate-specific images ===")
    prostate_images = generator.generate_prostate_specific()

    # Option 3: Generate custom medical image
    print("\n=== Generating custom medical image ===")
    custom_image = generator.generate_medical_image(
        custom_prompt="medical scan of kidney stones, diagnostic imaging, urological"
    )

    # Option 4: Batch generate different types
    print("\n=== Batch generating multiple types ===")
    batch_images = generator.batch_generate(
        categories=["mri", "ct", "ultrasound"],
        images_per_category=2
    )

    print("\nAll images generated successfully!")
    print("Check your current directory for the saved images.")
