"""
demo.py
-------
Command-line entry point.

    python demo.py path/to/video.mp4
    python demo.py path/to/video.mp4 --weights finetuned_head.pt --stride 2

Prints the Visual Output JSON that would be handed to the Fusion Engine:
    { visual_score, spatial_cnn_score, blinks_detected, frames_analyzed, diagnostics }
"""

import argparse
import json

from visual_analyzer import VisualConsistencyAnalyzer, FusionWeights
from verdict import print_verdict


def main():
    parser = argparse.ArgumentParser(description="Feature 1: Visual Consistency Analyzer")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--weights", default=None, help="Path to fine-tuned spatial CNN weights")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap on frames analyzed")
    parser.add_argument("--w-spatial", type=float, default=0.5)
    parser.add_argument("--w-behavioral", type=float, default=0.25)
    parser.add_argument("--w-jitter", type=float, default=0.25)
    parser.add_argument("--json-only", action="store_true", help="Print raw JSON only, skip the verdict summary")
    args = parser.parse_args()

    weights = FusionWeights(spatial=args.w_spatial, behavioral=args.w_behavioral, jitter=args.w_jitter)

    with VisualConsistencyAnalyzer(
        spatial_weights_path=args.weights,
        fusion_weights=weights,
        frame_stride=args.stride,
        max_frames=args.max_frames,
    ) as analyzer:
        output = analyzer.analyze(args.video_path)

    if args.json_only:
        print(json.dumps(output, indent=2))
    else:
        print_verdict(output)
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
