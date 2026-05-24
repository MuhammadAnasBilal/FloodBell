from PIL import Image, ImageDraw

def round_corners(image_path, output_path, radius):
    image = Image.open(image_path).convert("RGBA")
    
    # Create a mask with rounded corners
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), image.size], radius, fill=255)
    
    # Apply the mask to the image
    rounded_image = Image.new("RGBA", image.size)
    rounded_image.paste(image, (0, 0), mask=mask)
    
    # Save the result
    rounded_image.save(output_path, "PNG")

if __name__ == "__main__":
    logo_path = "D:\\ML_Prj\\frontend\\public\\logo.png"
    output_path = "D:\\ML_Prj\\frontend\\public\\logo_rounded.png"
    round_corners(logo_path, output_path, 100) # large radius for maximum roundness
    print("Saved rounded logo to", output_path)
