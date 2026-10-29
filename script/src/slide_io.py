"""Open scanner slides with OpenSlide and ordinary image TIFFs with ImageSlide."""
from pathlib import Path

import openslide


def open_slide(path):
    path = Path(path).resolve()
    try:
        return openslide.OpenSlide(str(path))
    except openslide.OpenSlideUnsupportedFormatError:
        if path.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            raise
        # Pillow's size protection is retained; large scanner slides need OpenSlide.
        return openslide.ImageSlide(str(path))
