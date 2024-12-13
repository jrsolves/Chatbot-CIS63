import subprocess
import os

def convert_png_to_svg_with_inkscape(png_path, svg_path):
    """
    Convert a PNG file to SVG with color using Inkscape, only if the SVG doesn't already exist.
    """
    if not os.path.exists(svg_path):  # Only convert if SVG doesn't exist
        inkscape_path = r"C:\Program Files\Inkscape\inkscape.exe"  # Hardcoded path to Inkscape
        try:
            # Use Inkscape to convert PNG to SVG, preserving colors
            subprocess.run([inkscape_path, png_path, '--export-type=svg', '--export-filename=' + svg_path], check=True)
            print(f"SVG with color saved at {svg_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error running Inkscape: {e}")
        except FileNotFoundError:
            print("Inkscape not found. Please ensure the path is correct.")
    else:
        print(f"{svg_path} already exists, skipping conversion.")

# Example usage:
convert_png_to_svg_with_inkscape("static/assets/character.png", "static/assets/character.svg")
