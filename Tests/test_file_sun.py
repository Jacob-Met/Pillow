from __future__ import annotations

import io
import os

import pytest

from PIL import Image, SunImagePlugin, _binary

from .helper import assert_image_equal_tofile, assert_image_similar, hopper

EXTRA_DIR = "Tests/images/sunraster"


def test_sanity() -> None:
    # Arrange
    # Created with ImageMagick: convert hopper.jpg hopper.ras
    test_file = "Tests/images/hopper.ras"

    # Act
    with Image.open(test_file) as im:
        # Assert
        assert im.size == (128, 128)

        assert_image_similar(im, hopper(), 5)  # visually verified

    invalid_file = "Tests/images/flower.jpg"
    with pytest.raises(SyntaxError):
        SunImagePlugin.SunImageFile(invalid_file)


def test_im1() -> None:
    with Image.open("Tests/images/sunraster.im1") as im:
        assert_image_equal_tofile(im, "Tests/images/sunraster.im1.png")


def _sun_header(
    depth: int = 0, file_type: int = 0, palette_length: int = 0
) -> io.BytesIO:
    return io.BytesIO(
        _binary.o32be(0x59A66A95)
        + b"\x00" * 8
        + _binary.o32be(depth)
        + b"\x00" * 4
        + _binary.o32be(file_type)
        + b"\x00" * 4
        + _binary.o32be(palette_length)
    )


def test_unsupported_mode_bit_depth() -> None:
    with pytest.raises(SyntaxError, match="Unsupported Mode/Bit Depth"):
        with SunImagePlugin.SunImageFile(_sun_header()):
            pass


def test_unsupported_color_palette_length() -> None:
    with pytest.raises(SyntaxError, match="Unsupported Color Palette Length"):
        with SunImagePlugin.SunImageFile(_sun_header(depth=1, palette_length=1025)):
            pass


def test_unsupported_palette_type() -> None:
    with pytest.raises(SyntaxError, match="Unsupported Palette Type"):
        with SunImagePlugin.SunImageFile(_sun_header(depth=1, palette_length=1)):
            pass


def test_unsupported_file_type() -> None:
    with pytest.raises(SyntaxError, match="Unsupported Sun Raster file type"):
        with SunImagePlugin.SunImageFile(_sun_header(depth=1, file_type=6)):
            pass


@pytest.mark.parametrize(
    "depth, width, first, second",
    (
        (1, 1, b"\x40", b"\x20"),
        (4, 1, b"\xa0", b"\x50"),
        (8, 3, b"\x10\x20\x30", b"\x40\x50\x60"),
        (24, 1, b"\x10\x20\x30", b"\x40\x50\x60"),
    ),
)
def test_rle_padded_rows(
    depth: int, width: int, first: bytes, second: bytes
) -> None:
    stride = ((width * depth + 15) // 16) * 2
    raw = first.ljust(stride, b"\0") + second.ljust(stride, b"\0")

    def sun_image(file_type: int, data: bytes) -> Image.Image:
        header = b"".join(
            _binary.o32be(value)
            for value in (0x59A66A95, width, 2, depth, len(data), file_type, 0, 0)
        )
        return Image.open(io.BytesIO(header + data))

    with sun_image(1, raw) as original, sun_image(2, raw) as encoded:
        assert encoded.mode == original.mode
        assert list(encoded.getdata()) == list(original.getdata())


def test_rle_run_crosses_padded_row_boundary() -> None:
    header = b"".join(
        _binary.o32be(value)
        for value in (0x59A66A95, 3, 2, 8, 3, 2, 0, 0)
    )
    with Image.open(io.BytesIO(header + b"\x80\x07\x10")) as im:
        assert list(im.getdata()) == [16] * 6


@pytest.mark.skipif(
    not os.path.exists(EXTRA_DIR), reason="Extra image files not installed"
)
def test_rgbx() -> None:
    with open(os.path.join(EXTRA_DIR, "32bpp.ras"), "rb") as fp:
        data = fp.read()

    # Set file type to 3
    data = data[:20] + _binary.o32be(3) + data[24:]

    with Image.open(io.BytesIO(data)) as im:
        r, g, b = im.split()
        im_rgb = Image.merge("RGB", (b, g, r))
        assert_image_equal_tofile(im_rgb, os.path.join(EXTRA_DIR, "32bpp.png"))


@pytest.mark.skipif(
    not os.path.exists(EXTRA_DIR), reason="Extra image files not installed"
)
def test_others() -> None:
    files = (
        os.path.join(EXTRA_DIR, f)
        for f in os.listdir(EXTRA_DIR)
        if os.path.splitext(f)[1] in (".sun", ".SUN", ".ras")
    )
    for path in files:
        with Image.open(path) as im:
            im.load()
            assert isinstance(im, SunImagePlugin.SunImageFile)
            assert_image_equal_tofile(im, f"{os.path.splitext(path)[0]}.png")
