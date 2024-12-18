from PIL import Image
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor

def extend_row_with_variation(row, target_length, variation_rate):
    """
    Extend a binary row with slight variations.
    
    Args:
        row: A binary row as a NumPy array.
        target_length: The desired extended length of the row.
        variation_rate: Probability of flipping a bit in the extended pattern.
    
    Returns:
        A list representing the extended binary row with variations.
    """
    pattern = ''.join(map(str, row.astype(int)))
    if target_length <= len(pattern):
        return list(map(int, pattern[:target_length]))
    
    # Start with the original pattern
    extended_row = list(map(int, pattern))

    # Generate extended part with slight variations
    additional_length = target_length - len(pattern)
    repeated_pattern = (pattern * ((additional_length // len(pattern)) + 1))[:additional_length]
    extended_part = list(map(int, repeated_pattern))

    # Introduce slight variations in the extended part
    for i in range(len(extended_part)):
        if random.random() < variation_rate:
            extended_part[i] = 1 - extended_part[i]  # Flip the bit
    
    return extended_row + extended_part


def process_color(color, image_array, original_height, new_width, variation_rate):
    """
    Process a single color to generate the extended mask for it.
    
    Args:
        color: The RGB color being processed.
        image_array: The original image as a NumPy array.
        original_height: Height of the image.
        new_width: Desired width of the extended output.
        variation_rate: Probability of flipping a bit in the extended pattern.
    
    Returns:
        A tuple of the color and its extended mask as a NumPy array.
    """
    # Create a binary mask for the current color
    mask = np.all(image_array == color, axis=-1).astype(np.uint8)

    # Extend the pattern for each row in the mask
    extended_mask = np.zeros((original_height, new_width), dtype=np.uint8)
    for i, row in enumerate(mask):
        extended_mask[i] = extend_row_with_variation(row, new_width, variation_rate)

    return color, extended_mask


def extend_color_pattern_with_variation(image_path: str, output_path: str, new_width: int, variation_rate: float = 0.01, max_colors: int = 1024):
    """
    Extend the pattern of an image for each unique color, mixing them back together.
    
    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the extended output image.
        new_width (int): The desired width of the output image.
        variation_rate (float): Probability of flipping a bit in the extended pattern (0.0 to 1.0).
        max_colors (int): Maximum number of unique colors to allow in the image.
    """
    # Open the image and convert to RGB
    image = Image.open(image_path).convert("RGB")
    
    # Check the number of unique colors in the image
    image_array = np.array(image)
    unique_colors = np.unique(image_array.reshape(-1, 3), axis=0)
    
    print(f"Found {len(unique_colors)} unique colors.")
    
    # If there are too many colors, quantize the image
    if len(unique_colors) > max_colors:
        print(f"Too many colors ({len(unique_colors)}). Reducing to {max_colors} colors using quantization.")
        # Ensure quantization does not exceed max_colors
        max_colors = min(max_colors, 256)  # Pillow quantize supports a maximum of 256 colors
        image = image.quantize(colors=max_colors, method=0).convert("RGB")
        image_array = np.array(image)  # Update the image array
        unique_colors = np.unique(image_array.reshape(-1, 3), axis=0)  # Recalculate unique colors
        print(f"Quantized to {len(unique_colors)} unique colors.")

    original_height, original_width, _ = image_array.shape

    # Prepare an array for the output image
    extended_image = np.zeros((original_height, new_width, 3), dtype=np.uint8)

    # Process each unique color in parallel
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(
                lambda color: process_color(color, image_array, original_height, new_width, variation_rate),
                unique_colors
            )
        )

    # Combine results back into the final image
    for color, extended_mask in results:
        for i in range(original_height):
            for j in range(new_width):
                if extended_mask[i, j] == 1:
                    extended_image[i, j] = color

    # Convert back to an image and save
    output_image = Image.fromarray(extended_image)
    output_image.save(output_path)
    print(f"Extended image with variations saved to {output_path}")


# Example Usage
input_image_path = "image1.png"  # Input color image path
output_image_path = "outputCol2xz.png"  # Output extended image path
desired_width = 300  # New width of the output image
variation_probability = 0.06  # Probability of flipping each bit (e.g., 6%)
max_colors_allowed = 1024  # Maximum number of unique colors to allow <- sadly right now, this can only perform 256 quantizations despite the indicator due to software limitations despite the math being relatively simple.

extend_color_pattern_with_variation(input_image_path, output_image_path, desired_width, variation_probability, max_colors_allowed)
