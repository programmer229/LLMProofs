import cairosvg
from io import BytesIO
import base64

def is_svg_parsable(svg_text: str) -> bool:
    """
    Check if the SVG text is parsable by cairosvg.
    
    Inputs:
        svg_text (str): The SVG XML-formatted text
    Returns:
        bool: True if the SVG text is parsable, False otherwise.
    """

    try:
        output_png = cairosvg.svg2png(bytestring=svg_text.encode('utf-8'))
        return True
    except Exception as e:
        return False

def convert_svg_to_png(svg_text: str, save_to_png: bool = False, save_path: str = None) -> bytes:
    """
    Convert the SVG text to a PNG image.
    
    Inputs:
        svg_text (str): The SVG XML-formatted text
        save_to_png (bool): Whether to save the PNG image to a file
        save_path (str): The path to save the PNG image to (required if save_to_png is True)
    Returns:
        bytes: The PNG image
    """

    try:
        if save_to_png:
            if save_path is None:
                raise ValueError("save_path must be provided if save_to_png is True")
            output_png = cairosvg.svg2png(bytestring=svg_text.encode('utf-8'), write_to=save_path)
        else:
            output_png = cairosvg.svg2png(bytestring=svg_text.encode('utf-8'))
        return output_png
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")
        return -1

def convert_svg_to_base64(svg_text: str) -> str:
    """
    Convert the SVG text to a base64 image.
    
    Inputs:
        svg_text (str): The SVG XML-formatted text
    Returns:
        str: The base64 image
    """

    try:
        png_image = cairosvg.svg2png(bytestring=svg_text.encode('utf-8'))
        base64_image = base64.b64encode(png_image).decode('utf-8')
        return base64_image
    except Exception as e:
        print(f"Error converting SVG to base64: {e}")
        return -1


if __name__ == "__main__":
    
    # Testing with a toy svg XML
    svg_text = '''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
      <circle cx="50" cy="50" r="40" stroke="black" stroke-width="2" fill="red" />
    </svg>'''

    print("Is SVG parsable:\n", is_svg_parsable(svg_text))
    print("\nConvert SVG to PNG:\n", convert_svg_to_png(svg_text, save_to_png=False))
    print("\nConvert SVG to base64:\n", convert_svg_to_base64(svg_text))