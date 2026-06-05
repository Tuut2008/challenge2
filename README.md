# Centerline Extraction

This repository extracts a stroke-oriented centerline from binary shape PNGs and exports the result as SVG paths.

## Overview

- `centerline_extractor.py` is the main workflow for processing all PNG samples in `challenge_sample`.
- `skeleton.py` contains the centerline extraction algorithm and SVG export logic.
- `evaluate.py` compares generated SVG results against expected SVGs in `challenge_sample_results_expected`.

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the main extraction pipeline:

```bash
python3 centerline_extractor.py
```

3. Evaluate generated results against expected SVGs:

```bash
python3 evaluate.py --expected
```

Or with custom folders:

```bash
python3 evaluate.py --results challenge_sample_results --expected challenge_sample_results_expected
```

## Algorithm and Pipeline

The extraction pipeline in `skeleton.py` follows these stages:

1. **Load binary mask**
   - Read the input PNG in grayscale.
   - Threshold and invert it so black strokes become a foreground mask.

2. **Skeleton extraction**
   - Default uses a medial-axis skeletonization from `skimage.morphology.medial_axis`.
   - The resulting skeleton is cleaned with a distance-transform-based spur removal pass.
   - A flux-based centerline extractor exists as an alternative method.

3. **Graph tracing**
   - Convert skeleton pixels into a graph.
   - Trace connected edges into ordered branches between endpoints and junctions.

4. **Branch splitting**
   - Split long branches at strong geometric bends so corners and stroke turns behave like separate human strokes.

5. **Branch cleaning**
   - Simplify nearly straight branches to lines.
   - Smooth and simplify curved branches using Ramer-Douglas-Peucker and local smoothing.

6. **Endpoint snapping and merging**
   - Snap nearby branch endpoints to shared junctions.
   - Merge segments that are aligned or form smooth continuation.

7. **Corner handling**
   - Detect connected two-stroke corners and build small connecting bridges.
   - Use endpoint tangent lines to locate the correct meeting point.
   - Anchor corner geometry back onto the raw skeleton when possible.

8. **Junction smoothing**
   - Smooth three-way junctions while preserving original centerline topology.

9. **Final cleanup**
   - Prune redundant short strokes and simplify final paths.
   - Deduplicate repeated paths.

10. **Export to SVG**
    - Write final path data using `M`, `L`, and quadratic `Q` segments.
    - Export with a fixed stroke style for visualization.

## File Structure

- `centerline_extractor.py` - batch workflow for challenge samples.
- `evaluate.py` - compare generated SVG output to expected reference SVGs.
- `skeleton.py` - core extraction algorithm and SVG export.
- `challenge_sample/` - input PNG samples.
- `challenge_sample_results/` - generated SVG outputs.
- `challenge_sample_results_expected/` - expected SVG reference files.

## Notes

- The script `centerline_extractor.py` is the preferred entry point.
- `skeleton.py` contains utility functions and can be imported directly if you want to run individual conversions.
- The evaluation currently compares path count, total length, and endpoint proximity.
