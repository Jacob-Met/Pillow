/* Copyright (c) 2010 Oliver Tonnhofer <olt@bogosoft.com>, Omniscale
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
*/

/*
// This file implements a variation of the octree color quantization algorithm.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include "ImagingUtils.h"
#include "QuantOctree.h"

typedef struct _ColorBucket {
    /* contains palette index when used for look up cube */
    uint32_t count;
    uint64_t r;
    uint64_t g;
    uint64_t b;
    uint64_t a;
} *ColorBucket;

typedef struct _ColorCube {
    unsigned int rBits, gBits, bBits, aBits;
    unsigned int rWidth, gWidth, bWidth, aWidth;
    unsigned int rOffset, gOffset, bOffset, aOffset;

    unsigned long size;
    ColorBucket buckets;
} *ColorCube;

static ColorCube
new_color_cube(int r, int g, int b, int a) {
    ColorCube cube;

    /* malloc check ok, small constant allocation */
    cube = malloc(sizeof(struct _ColorCube));
    if (!cube) {
        return NULL;
    }

    cube->rBits = MAX(r, 0);
    cube->gBits = MAX(g, 0);
    cube->bBits = MAX(b, 0);
    cube->aBits = MAX(a, 0);

    /* overflow check for size multiplication below */
    if (cube->rBits + cube->gBits + cube->bBits + cube->aBits > 31) {
        free(cube);
        return NULL;
    }

    /* the width of the cube for each dimension */
    cube->rWidth = 1 << cube->rBits;
    cube->gWidth = 1 << cube->gBits;
    cube->bWidth = 1 << cube->bBits;
    cube->aWidth = 1 << cube->aBits;

    /* the offsets of each color */

    cube->rOffset = cube->gBits + cube->bBits + cube->aBits;
    cube->gOffset = cube->bBits + cube->aBits;
    cube->bOffset = cube->aBits;
    cube->aOffset = 0;

    /* the number of color buckets */
    cube->size = (unsigned long)cube->rWidth * cube->gWidth *
                 (unsigned long)cube->bWidth * cube->aWidth;
    /* malloc check ok, overflow checked above */
    cube->buckets = calloc(cube->size, sizeof(struct _ColorBucket));

    if (!cube->buckets) {
        free(cube);
        return NULL;
    }
    return cube;
}

static void
free_color_cube(ColorCube cube) {
    if (cube != NULL) {
        free(cube->buckets);
        free(cube);
    }
}

static long
color_bucket_offset_pos(
    const ColorCube cube, unsigned int r, unsigned int g, unsigned int b, unsigned int a
) {
    return r << cube->rOffset | g << cube->gOffset | b << cube->bOffset |
           a << cube->aOffset;
}

static long
color_bucket_offset(const ColorCube cube, const Pixel *p) {
    unsigned int r = p->c.r >> (8 - cube->rBits);
    unsigned int g = p->c.g >> (8 - cube->gBits);
    unsigned int b = p->c.b >> (8 - cube->bBits);
    unsigned int a = p->c.a >> (8 - cube->aBits);
    return color_bucket_offset_pos(cube, r, g, b, a);
}

static ColorBucket
color_bucket_from_cube(const ColorCube cube, const Pixel *p) {
    unsigned int offset = color_bucket_offset(cube, p);
    return &cube->buckets[offset];
}

static void
add_color_to_color_cube(const ColorCube cube, const Pixel *p) {
    ColorBucket bucket = color_bucket_from_cube(cube, p);
    bucket->count += 1;
    bucket->r += p->c.r;
    bucket->g += p->c.g;
    bucket->b += p->c.b;
    bucket->a += p->c.a;
}

static unsigned long
count_used_color_buckets(const ColorCube cube) {
    unsigned long usedBuckets = 0;
    unsigned long i;
    for (i = 0; i < cube->size; i++) {
        if (cube->buckets[i].count > 0) {
            usedBuckets += 1;
        }
    }
    return usedBuckets;
}

static void
avg_color_from_color_bucket(const ColorBucket bucket, Pixel *dst) {
    float count = bucket->count;
    if (count != 0) {
        dst->c.r = CLIP8((int)(bucket->r / count));
        dst->c.g = CLIP8((int)(bucket->g / count));
        dst->c.b = CLIP8((int)(bucket->b / count));
        dst->c.a = CLIP8((int)(bucket->a / count));
    } else {
        dst->c.r = 0;
        dst->c.g = 0;
        dst->c.b = 0;
        dst->c.a = 0;
    }
}

static int
compare_bucket_count(const ColorBucket a, const ColorBucket b) {
    return b->count - a->count;
}

static ColorBucket
create_sorted_color_palette(const ColorCube cube) {
    ColorBucket buckets;
    if (cube->size > LONG_MAX / sizeof(struct _ColorBucket)) {
        return NULL;
    }
    /* malloc check ok, calloc + overflow check above for memcpy */
    buckets = calloc(cube->size, sizeof(struct _ColorBucket));
    if (!buckets) {
        return NULL;
    }
    memcpy(buckets, cube->buckets, sizeof(struct _ColorBucket) * cube->size);

    qsort(
        buckets,
        cube->size,
        sizeof(struct _ColorBucket),
        (int (*)(void const *, void const *))&compare_bucket_count
    );

    return buckets;
}

void
add_bucket_values(ColorBucket src, ColorBucket dst) {
    dst->count += src->count;
    dst->r += src->r;
    dst->g += src->g;
    dst->b += src->b;
    dst->a += src->a;
}

/* expand or shrink a given cube to level */
static ColorCube
copy_color_cube(
    const ColorCube cube,
    unsigned int rBits,
    unsigned int gBits,
    unsigned int bBits,
    unsigned int aBits
) {
    unsigned int r, g, b, a;
    long src_pos, dst_pos;
    unsigned int src_reduce[4] = {0}, dst_reduce[4] = {0};
    unsigned int width[4];
    ColorCube result;

    result = new_color_cube(rBits, gBits, bBits, aBits);
    if (!result) {
        return NULL;
    }

    if (cube->rBits > rBits) {
        dst_reduce[0] = cube->rBits - result->rBits;
        width[0] = cube->rWidth;
    } else {
        src_reduce[0] = result->rBits - cube->rBits;
        width[0] = result->rWidth;
    }
    if (cube->gBits > gBits) {
        dst_reduce[1] = cube->gBits - result->gBits;
        width[1] = cube->gWidth;
    } else {
        src_reduce[1] = result->gBits - cube->gBits;
        width[1] = result->gWidth;
    }
    if (cube->bBits > bBits) {
        dst_reduce[2] = cube->bBits - result->bBits;
        width[2] = cube->bWidth;
    } else {
        src_reduce[2] = result->bBits - cube->bBits;
        width[2] = result->bWidth;
    }
    if (cube->aBits > aBits) {
        dst_reduce[3] = cube->aBits - result->aBits;
        width[3] = cube->aWidth;
    } else {
        src_reduce[3] = result->aBits - cube->aBits;
        width[3] = result->aWidth;
    }

    for (r = 0; r < width[0]; r++) {
        for (g = 0; g < width[1]; g++) {
            for (b = 0; b < width[2]; b++) {
                for (a = 0; a < width[3]; a++) {
                    src_pos = color_bucket_offset_pos(
                        cube,
                        r >> src_reduce[0],
                        g >> src_reduce[1],
                        b >> src_reduce[2],
                        a >> src_reduce[3]
                    );
                    dst_pos = color_bucket_offset_pos(
                        result,
                        r >> dst_reduce[0],
                        g >> dst_reduce[1],
                        b >> dst_reduce[2],
                        a >> dst_reduce[3]
                    );
                    add_bucket_values(
                        &cube->buckets[src_pos], &result->buckets[dst_pos]
                    );
                }
            }
        }
    }
    return result;
}

void
subtract_color_buckets(ColorCube cube, ColorBucket buckets, long nBuckets) {
    ColorBucket minuend, subtrahend;
    long i;
    Pixel p;
    for (i = 0; i < nBuckets; i++) {
        subtrahend = &buckets[i];

        // If the subtrahend contains no buckets, there is nothing to subtract.
        if (subtrahend->count == 0) {
            continue;
        }

        avg_color_from_color_bucket(subtrahend, &p);
        minuend = color_bucket_from_cube(cube, &p);
        minuend->count -= subtrahend->count;
        minuend->r -= subtrahend->r;
        minuend->g -= subtrahend->g;
        minuend->b -= subtrahend->b;
        minuend->a -= subtrahend->a;
    }
}

static void
set_lookup_value(const ColorCube cube, const Pixel *p, long value) {
    ColorBucket bucket = color_bucket_from_cube(cube, p);
    bucket->count = value;
}

uint64_t
lookup_color(const ColorCube cube, const Pixel *p) {
    ColorBucket bucket = color_bucket_from_cube(cube, p);
    return bucket->count;
}

void
add_lookup_buckets(ColorCube cube, ColorBucket palette, long nColors, long offset) {
    long i;
    Pixel p;
    for (i = offset + nColors - 1; i >= offset; i--) {
        avg_color_from_color_bucket(&palette[i], &p);
        set_lookup_value(cube, &p, i);
    }
}

ColorBucket
combined_palette(
    ColorBucket bucketsA,
    unsigned long nBucketsA,
    ColorBucket bucketsB,
    unsigned long nBucketsB
) {
    ColorBucket result;
    if (nBucketsA > LONG_MAX - nBucketsB ||
        (nBucketsA + nBucketsB) > LONG_MAX / sizeof(struct _ColorBucket)) {
        return NULL;
    }
    /* malloc check ok, overflow check above */
    result = calloc(nBucketsA + nBucketsB, sizeof(struct _ColorBucket));
    if (!result) {
        return NULL;
    }
    memcpy(result, bucketsA, sizeof(struct _ColorBucket) * nBucketsA);
    memcpy(&result[nBucketsA], bucketsB, sizeof(struct _ColorBucket) * nBucketsB);
    return result;
}

static Pixel *
create_palette_array(const ColorBucket palette, unsigned int paletteLength) {
    Pixel *paletteArray;
    unsigned int i;

    /* malloc check ok, calloc for overflow */
    paletteArray = calloc(paletteLength, sizeof(Pixel));
    if (!paletteArray) {
        return NULL;
    }

    for (i = 0; i < paletteLength; i++) {
        avg_color_from_color_bucket(&palette[i], &paletteArray[i]);
    }
    return paletteArray;
}

static void
map_image_pixels(
    const Pixel *pixelData,
    uint32_t nPixels,
    const ColorCube lookupCube,
    uint32_t *pixelArray
) {
    long i;
    for (i = 0; i < nPixels; i++) {
        pixelArray[i] = lookup_color(lookupCube, &pixelData[i]);
    }
}

const unsigned int CUBE_LEVELS[8] = {4, 4, 4, 0, 2, 2, 2, 0};
const unsigned int CUBE_LEVELS_ALPHA[8] = {3, 4, 3, 3, 2, 2, 2, 2};

/*
An RGBA image may have only a few distinct RGB colors but hundreds of alpha
levels.  A fixed four-dimensional octree then uses most of its buckets on
unneeded RGB precision, reducing alpha to a few steps.  Such images can be
quantized more accurately by keeping each RGB color and dividing its alpha
range into equally sized buckets.  Other images keep the normal octree path.

Returns 1 if quantized, 0 if this case does not apply, and -1 on allocation
failure.
*/
static int
quantize_sparse_rgba(
    Pixel *pixelData,
    uint32_t nPixels,
    uint32_t nQuantPixels,
    Pixel **palette,
    uint32_t *paletteLength,
    uint32_t **quantizedPixels
) {
    uint32_t rgbKeys[256];
    unsigned int rgbIDs[256];
    unsigned int alphaMin[64], alphaMax[64];
    unsigned char alphaSeen[256] = {0};
    unsigned int rgbCount = 0, alphaCount = 0;
    unsigned int alphaBuckets;
    ColorBucket buckets = NULL;
    Pixel *colors = NULL;
    uint32_t *indices = NULL;
    uint32_t i;

    if (!nPixels || nQuantPixels < 2) {
        return 0;
    }

    for (i = 0; i < 256; i++) {
        rgbKeys[i] = UINT32_MAX;
    }

    for (i = 0; i < nPixels; i++) {
        Pixel *p = &pixelData[i];
        uint32_t rgb = ((uint32_t)p->c.r << 16) | ((uint32_t)p->c.g << 8) | p->c.b;
        unsigned int pos = (rgb * 2654435761U) & 255U;
        unsigned int id;

        while (rgbKeys[pos] != UINT32_MAX && rgbKeys[pos] != rgb) {
            pos = (pos + 1) & 255U;
        }
        if (rgbKeys[pos] == UINT32_MAX) {
            if (rgbCount >= 64 || rgbCount >= nQuantPixels) {
                return 0;
            }
            rgbKeys[pos] = rgb;
            rgbIDs[pos] = rgbCount;
            alphaMin[rgbCount] = 255;
            alphaMax[rgbCount] = 0;
            rgbCount++;
        }
        id = rgbIDs[pos];
        alphaMin[id] = MIN(alphaMin[id], p->c.a);
        alphaMax[id] = MAX(alphaMax[id], p->c.a);

        if (!alphaSeen[p->c.a]) {
            alphaSeen[p->c.a] = 1;
            alphaCount++;
        }
    }
    if (alphaCount <= 8) {
        return 0;
    }

    alphaBuckets = nQuantPixels / rgbCount;

    /* malloc check ok, bounded by nQuantPixels (maximum 256) */
    buckets = calloc(nQuantPixels, sizeof(struct _ColorBucket));
    colors = calloc(nQuantPixels, sizeof(Pixel));
    /* malloc check ok, calloc + overflow protection */
    indices = calloc(nPixels, sizeof(uint32_t));
    if (!buckets || !colors || !indices) {
        free(buckets);
        free(colors);
        free(indices);
        return -1;
    }

    for (i = 0; i < nPixels; i++) {
        Pixel *p = &pixelData[i];
        uint32_t rgb = ((uint32_t)p->c.r << 16) | ((uint32_t)p->c.g << 8) | p->c.b;
        unsigned int pos = (rgb * 2654435761U) & 255U;
        unsigned int id, alpha, entry;
        ColorBucket bucket;

        while (rgbKeys[pos] != rgb) {
            pos = (pos + 1) & 255U;
        }
        id = rgbIDs[pos];
        alpha = (unsigned int)(p->c.a - alphaMin[id]) * alphaBuckets /
                (alphaMax[id] - alphaMin[id] + 1);
        entry = id * alphaBuckets + MIN(alpha, alphaBuckets - 1);
        indices[i] = entry;

        bucket = &buckets[entry];
        bucket->count++;
        bucket->r += p->c.r;
        bucket->g += p->c.g;
        bucket->b += p->c.b;
        bucket->a += p->c.a;
    }

    for (i = 0; i < nQuantPixels; i++) {
        avg_color_from_color_bucket(&buckets[i], &colors[i]);
    }

    free(buckets);
    *palette = colors;
    *paletteLength = nQuantPixels;
    *quantizedPixels = indices;
    return 1;
}

/*
Preserve colors exactly when an RGBA image already fits the requested
palette. The octree's fixed color buckets otherwise merge distinct colors,
even when fewer than 256 colors occur in the entire image.

The 512-slot open-addressed table stores at most 256 colors, so lookups and
output mapping remain linear in the number of pixels.
*/
static int
quantize_exact_rgba(
    Pixel *pixelData,
    uint32_t nPixels,
    uint32_t nQuantPixels,
    Pixel **palette,
    uint32_t *paletteLength,
    uint32_t **quantizedPixels
) {
    uint32_t keys[512];
    unsigned char used[512] = {0};
    unsigned short slots[512];
    Pixel colors[256];
    uint32_t *indices = NULL;
    Pixel *resultPalette = NULL;
    unsigned int count = 0;
    uint32_t i;

    if (!nPixels || !nQuantPixels || nQuantPixels > 256) {
        return 0;
    }

    for (i = 0; i < nPixels; i++) {
        Pixel *p = &pixelData[i];
        uint32_t key = ((uint32_t)p->c.r << 24) |
                       ((uint32_t)p->c.g << 16) |
                       ((uint32_t)p->c.b << 8) | (uint32_t)p->c.a;
        unsigned int pos = (key * 2654435761U) >> 23;
        while (used[pos] && keys[pos] != key) {
            pos = (pos + 1) & 511U;
        }
        if (!used[pos]) {
            if (count >= nQuantPixels) {
                return 0;
            }
            used[pos] = 1;
            keys[pos] = key;
            slots[pos] = count;
            colors[count++] = *p;
        }
    }

    /* calloc protects against oversized images and keeps allocation checks. */
    indices = calloc(nPixels, sizeof(uint32_t));
    resultPalette = calloc(count, sizeof(Pixel));
    if (!indices || !resultPalette) {
        free(indices);
        free(resultPalette);
        return -1;
    }
    memcpy(resultPalette, colors, count * sizeof(Pixel));

    for (i = 0; i < nPixels; i++) {
        Pixel *p = &pixelData[i];
        uint32_t key = ((uint32_t)p->c.r << 24) |
                       ((uint32_t)p->c.g << 16) |
                       ((uint32_t)p->c.b << 8) | (uint32_t)p->c.a;
        unsigned int pos = (key * 2654435761U) >> 23;
        while (keys[pos] != key) {
            pos = (pos + 1) & 511U;
        }
        indices[i] = slots[pos];
    }

    *palette = resultPalette;
    *paletteLength = count;
    *quantizedPixels = indices;
    return 1;
}

int
quantize_octree(
    Pixel *pixelData,
    uint32_t nPixels,
    uint32_t nQuantPixels,
    Pixel **palette,
    uint32_t *paletteLength,
    uint32_t **quantizedPixels,
    int withAlpha
) {
    ColorCube fineCube = NULL;
    ColorCube coarseCube = NULL;
    ColorCube lookupCube = NULL;
    ColorCube coarseLookupCube = NULL;
    ColorBucket paletteBucketsCoarse = NULL;
    ColorBucket paletteBucketsFine = NULL;
    ColorBucket paletteBuckets = NULL;
    uint32_t *qp = NULL;
    long i;
    unsigned long nCoarseColors, nFineColors, nAlreadySubtracted;
    const unsigned int *cubeBits;
    int sparse;

    if (withAlpha) {
        sparse = quantize_exact_rgba(
            pixelData, nPixels, nQuantPixels, palette, paletteLength, quantizedPixels
        );
        if (sparse) {
            return sparse > 0;
        }
        sparse = quantize_sparse_rgba(
            pixelData, nPixels, nQuantPixels, palette, paletteLength, quantizedPixels
        );
        if (sparse) {
            return sparse > 0;
        }
    }

    if (withAlpha) {
        cubeBits = CUBE_LEVELS_ALPHA;
    } else {
        cubeBits = CUBE_LEVELS;
    }

    /*
    Create two color cubes, one fine grained with 8x16x8=1024
    colors buckets and a coarse with 4x4x4=64 color buckets.
    The coarse one guarantees that there are color buckets available for
    the whole color range (assuming nQuantPixels > 64).

    For a quantization to 256 colors all 64 coarse colors will be used
    plus the 192 most used color buckets from the fine color cube.
    The average of all colors within one bucket is used as the actual
    color for that bucket.

     For images with alpha the cubes gets a forth dimension,
     8x16x8x8 and 4x4x4x4.
    */

    /* create fine cube */
    fineCube = new_color_cube(cubeBits[0], cubeBits[1], cubeBits[2], cubeBits[3]);
    if (!fineCube) {
        goto error;
    }
    for (i = 0; i < nPixels; i++) {
        add_color_to_color_cube(fineCube, &pixelData[i]);
    }

    /* create coarse cube */
    coarseCube =
        copy_color_cube(fineCube, cubeBits[4], cubeBits[5], cubeBits[6], cubeBits[7]);
    if (!coarseCube) {
        goto error;
    }
    nCoarseColors = count_used_color_buckets(coarseCube);

    /* limit to nQuantPixels */
    if (nCoarseColors > nQuantPixels) {
        nCoarseColors = nQuantPixels;
    }

    /* how many space do we have in our palette for fine colors? */
    nFineColors = nQuantPixels - nCoarseColors;

    /* create fine color palette */
    paletteBucketsFine = create_sorted_color_palette(fineCube);
    if (!paletteBucketsFine) {
        goto error;
    }

    /* remove the used fine colors from the coarse cube */
    subtract_color_buckets(coarseCube, paletteBucketsFine, nFineColors);

    /* did the subtraction cleared one or more coarse bucket? */
    while (nCoarseColors > count_used_color_buckets(coarseCube)) {
        /* then we can use the free buckets for fine colors */
        nAlreadySubtracted = nFineColors;
        nCoarseColors = count_used_color_buckets(coarseCube);
        nFineColors = nQuantPixels - nCoarseColors;
        subtract_color_buckets(
            coarseCube,
            &paletteBucketsFine[nAlreadySubtracted],
            nFineColors - nAlreadySubtracted
        );
    }

    /* create our palette buckets with fine and coarse combined */
    paletteBucketsCoarse = create_sorted_color_palette(coarseCube);
    if (!paletteBucketsCoarse) {
        goto error;
    }
    paletteBuckets = combined_palette(
        paletteBucketsCoarse, nCoarseColors, paletteBucketsFine, nFineColors
    );

    free(paletteBucketsFine);
    paletteBucketsFine = NULL;
    free(paletteBucketsCoarse);
    paletteBucketsCoarse = NULL;
    if (!paletteBuckets) {
        goto error;
    }

    /* add all coarse colors to our coarse lookup cube. */
    coarseLookupCube =
        new_color_cube(cubeBits[4], cubeBits[5], cubeBits[6], cubeBits[7]);
    if (!coarseLookupCube) {
        goto error;
    }
    add_lookup_buckets(coarseLookupCube, paletteBuckets, nCoarseColors, 0);

    /* expand coarse cube (64) to larger fine cube (4k). the value of each
       coarse bucket is then present in the according 64 fine buckets. */
    lookupCube = copy_color_cube(
        coarseLookupCube, cubeBits[0], cubeBits[1], cubeBits[2], cubeBits[3]
    );
    if (!lookupCube) {
        goto error;
    }

    /* add fine colors to the lookup cube */
    add_lookup_buckets(lookupCube, paletteBuckets, nFineColors, nCoarseColors);

    /* create result pixels and map palette indices */
    /* malloc check ok, calloc for overflow */
    qp = calloc(nPixels, sizeof(Pixel));
    if (!qp) {
        goto error;
    }
    map_image_pixels(pixelData, nPixels, lookupCube, qp);

    /* convert palette buckets to RGB pixel palette */
    *palette = create_palette_array(paletteBuckets, nQuantPixels);
    if (!(*palette)) {
        goto error;
    }

    *quantizedPixels = qp;
    *paletteLength = nQuantPixels;

    free_color_cube(coarseCube);
    free_color_cube(fineCube);
    free_color_cube(lookupCube);
    free_color_cube(coarseLookupCube);
    free(paletteBuckets);
    return 1;

error:
    /* everything is initialized to NULL
       so we are safe to call free */
    free(qp);
    free_color_cube(lookupCube);
    free_color_cube(coarseLookupCube);
    free(paletteBuckets);
    free(paletteBucketsCoarse);
    free(paletteBucketsFine);
    free_color_cube(coarseCube);
    free_color_cube(fineCube);
    return 0;
}
