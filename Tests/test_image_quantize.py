from __future__ import annotations

import pytest

from PIL import Image

from .helper import (
    assert_image_similar,
    has_feature_version,
    hopper,
    is_ppc64le,
    skip_unless_feature,
)


def test_sanity() -> None:
    image = hopper()
    converted = image.quantize()
    assert converted.mode == "P"
    assert_image_similar(converted.convert("RGB"), image, 10)

    image = hopper()
    converted = image.quantize(palette=hopper("P"))
    assert converted.mode == "P"
    assert_image_similar(converted.convert("RGB"), image, 60)


@skip_unless_feature("libimagequant")
def test_libimagequant_quantize() -> None:
    image = hopper()
    if is_ppc64le() and not has_feature_version("libimagequant", "4"):
        pytest.skip("Fails with libimagequant earlier than 4.0.0 on ppc64le")
    converted = image.quantize(100, Image.Quantize.LIBIMAGEQUANT)
    assert converted.mode == "P"
    assert_image_similar(converted.convert("RGB"), image, 15)
    colors = converted.getcolors()
    assert colors is not None
    assert len(colors) == 100


def test_octree_quantize() -> None:
    image = hopper()
    converted = image.quantize(100, Image.Quantize.FASTOCTREE)
    assert converted.mode == "P"
    assert_image_similar(converted.convert("RGB"), image, 20)
    colors = converted.getcolors()
    assert colors is not None
    assert len(colors) == 100


def test_rgba_quantize() -> None:
    image = hopper("RGBA")
    with pytest.raises(ValueError):
        image.quantize(method=0)

    assert image.quantize().convert().mode == "RGBA"


@pytest.mark.parametrize("colors", [16, 64, 128, 256])
def test_rgba_fast_octree_alpha_gradients(colors: int) -> None:
    """Quantizing a few RGB colors must not discard most alpha levels."""
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    image = Image.new("RGBA", (128, len(base_colors)))
    image.putdata([(*color, alpha) for color in base_colors for alpha in range(128)])

    quantized = image.quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
    assert quantized.mode == "P"
    assert len(set(quantized.tobytes())) == colors


@pytest.mark.parametrize("distinct", [1, 2, 12, 147, 253, 256])
def test_rgba_fast_octree_exact_colors(distinct: int) -> None:
    # Exact palette mapping must beat coarse octree buckets when all colors fit.
    colors = [(i, i, i, 255) for i in range(distinct)]
    image = Image.new("RGBA", (distinct, 2))
    image.putdata(colors * 2)

    quantized = image.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
    assert quantized.convert("RGBA").tobytes() == image.tobytes()
    assert len(set(quantized.tobytes())) == distinct


def test_rgba_fast_octree_exact_transparency() -> None:
    colors = [
        ((i * 53) % 256, (i * 79) % 256, (i * 101) % 256, i)
        for i in range(256)
    ]
    image = Image.new("RGBA", (256, 1))
    image.putdata(colors)

    quantized = image.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
    assert quantized.convert("RGBA").tobytes() == image.tobytes()


def test_rgba_fast_octree_many_rgb_channel_values() -> None:
    """A handful of colors with varied channels still needs alpha detail."""
    base_colors = [
        ((i * 31 + 1) % 256, (i * 47 + 5) % 256, (i * 59 + 7) % 256) for i in range(8)
    ]
    image = Image.new("RGBA", (128, len(base_colors)))
    image.putdata([(*color, alpha) for color in base_colors for alpha in range(128)])
    quantized = image.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
    assert len(set(quantized.tobytes())) >= 128


def test_rgba_fast_octree_non_power_of_two_colors() -> None:
    """Use nearly all palette entries even with five distinct RGB colors."""
    base_colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]
    image = Image.new("RGBA", (256, len(base_colors)))
    image.putdata([(*color, alpha) for color in base_colors for alpha in range(256)])
    quantized = image.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
    assert len(set(quantized.tobytes())) >= 250


def test_rgba_fast_octree_alpha_accuracy() -> None:
    """An alpha gradient should not have abrupt coarse palette regions."""
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    image = Image.new("RGBA", (256, len(base_colors)))
    image.putdata([(*color, alpha) for color in base_colors for alpha in range(256)])
    quantized = image.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
    result = quantized.convert("RGBA").getchannel("A")
    original = image.getchannel("A")
    error = sum(abs(a - b) for a, b in zip(original.tobytes(), result.tobytes()))
    assert error / (image.width * image.height) <= 2


def test_quantize() -> None:
    with Image.open("Tests/images/caption_6_33_22.png") as image:
        converted = image.convert("RGB")
    converted = converted.quantize()
    assert converted.mode == "P"
    assert_image_similar(converted.convert("RGB"), image, 1)


def test_quantize_no_dither() -> None:
    image = hopper()
    with Image.open("Tests/images/caption_6_33_22.png") as palette:
        palette_p = palette.convert("P")

    converted = image.quantize(dither=Image.Dither.NONE, palette=palette_p)
    assert converted.mode == "P"
    assert converted.palette is not None
    assert palette_p.palette is not None
    assert converted.palette.palette == palette_p.palette.palette


def test_quantize_no_dither2() -> None:
    im = Image.new("RGB", (9, 1))
    im.putdata([(p,) * 3 for p in range(0, 36, 4)])

    palette = Image.new("P", (1, 1))
    data = (0, 0, 0, 32, 32, 32)
    palette.putpalette(data)
    quantized = im.quantize(dither=Image.Dither.NONE, palette=palette)

    assert quantized.palette is not None
    assert tuple(quantized.palette.palette) == data

    px = quantized.load()
    assert px is not None
    for x in range(9):
        assert px[x, 0] == (0 if x < 5 else 1)


def test_quantize_dither_diff() -> None:
    image = hopper()
    with Image.open("Tests/images/caption_6_33_22.png") as palette:
        palette_p = palette.convert("P")

    dither = image.quantize(dither=Image.Dither.FLOYDSTEINBERG, palette=palette_p)
    nodither = image.quantize(dither=Image.Dither.NONE, palette=palette_p)

    assert dither.tobytes() != nodither.tobytes()


@pytest.mark.parametrize(
    "method", (Image.Quantize.MEDIANCUT, Image.Quantize.MAXCOVERAGE)
)
def test_quantize_kmeans(method: Image.Quantize) -> None:
    im = hopper()
    no_kmeans = im.quantize(kmeans=0, method=method)
    kmeans = im.quantize(kmeans=1, method=method)
    assert kmeans.tobytes() != no_kmeans.tobytes()

    with pytest.raises(ValueError):
        im.quantize(kmeans=-1, method=method)


@skip_unless_feature("libimagequant")
def test_resize() -> None:
    im = hopper().resize((100, 100))
    converted = im.quantize(100, Image.Quantize.LIBIMAGEQUANT)
    colors = converted.getcolors()
    assert colors is not None
    assert len(colors) == 100


def test_colors() -> None:
    im = hopper()
    colors = 2
    converted = im.quantize(colors)
    assert converted.palette is not None
    assert len(converted.palette.palette) == colors * len("RGB")


def test_transparent_colors_equal() -> None:
    im = Image.new("RGBA", (1, 2), (0, 0, 0, 0))
    px = im.load()
    assert px is not None
    px[0, 1] = (255, 255, 255, 0)

    converted = im.quantize()
    converted_px = converted.load()
    assert converted_px is not None
    assert converted_px[0, 0] == converted_px[0, 1]


@pytest.mark.parametrize(
    "method, color",
    (
        (Image.Quantize.MEDIANCUT, (0, 0, 0)),
        (Image.Quantize.MAXCOVERAGE, (0, 0, 0)),
        (Image.Quantize.FASTOCTREE, (0, 0, 0)),
        (Image.Quantize.FASTOCTREE, (0, 0, 0, 0)),
    ),
)
def test_palette(method: Image.Quantize, color: tuple[int, ...]) -> None:
    im = Image.new("RGBA" if len(color) == 4 else "RGB", (1, 1), color)

    converted = im.quantize(method=method)
    assert converted.palette is not None
    assert converted.getpixel((0, 0)) == converted.palette.colors[color]


def test_small_palette() -> None:
    # Arrange
    im = hopper()

    colors = (255, 0, 0, 0, 0, 255)
    p = Image.new("P", (1, 1))
    p.putpalette(colors)

    # Act
    im = im.quantize(palette=p)

    # Assert
    quantized_colors = im.getcolors()
    assert quantized_colors is not None
    assert len(quantized_colors) == 2


def test_empty_palette() -> None:
    im = hopper()

    p = Image.new("P", (1, 1))
    assert p.palette is not None
    assert p.palette.palette == b""

    with pytest.raises(ValueError, match="empty palette"):
        im.quantize(palette=p)
