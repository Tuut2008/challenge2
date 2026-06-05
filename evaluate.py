#!/usr/bin/env python3
import argparse
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

NUMBER_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")
COMMAND_RE = re.compile(r"[MmLlQqHhVvZz]")


def parse_path_d(d):
    tokens = COMMAND_RE.split(d)
    commands = COMMAND_RE.findall(d)
    path = []
    if not commands:
        return path

    current = (0.0, 0.0)
    idx = 0
    for cmd, token in zip(commands, tokens[1:]):
        values = [float(x) for x in NUMBER_RE.findall(token)]
        if cmd == 'M' or cmd == 'L':
            for i in range(0, len(values), 2):
                p = (values[i], values[i + 1])
                path.append((cmd, p))
                current = p
        elif cmd == 'Q':
            for i in range(0, len(values), 4):
                control = (values[i], values[i + 1])
                end = (values[i + 2], values[i + 3])
                path.append((cmd, control, end))
                current = end
        elif cmd == 'H':
            for x in values:
                p = (x, current[1])
                path.append(('L', p))
                current = p
        elif cmd == 'V':
            for y in values:
                p = (current[0], y)
                path.append(('L', p))
                current = p
        elif cmd == 'Z' or cmd == 'z':
            path.append(('Z', None))
        else:
            # unsupported command, ignore
            continue
        idx += 1
    return path


def evaluate_path_length(path):
    length = 0.0
    start = None
    last = None
    for segment in path:
        if segment[0] == 'M':
            last = segment[1]
            if start is None:
                start = last
        elif segment[0] == 'L':
            if last is not None:
                length += math.dist(last, segment[1])
            last = segment[1]
            if start is None:
                start = last
        elif segment[0] == 'Q':
            if last is not None:
                control, end = segment[1], segment[2]
                length += quadratic_bezier_length(last, control, end)
            last = segment[2]
            if start is None:
                start = last
        elif segment[0] == 'Z':
            if last is not None and start is not None:
                length += math.dist(last, start)
            last = start
    return length


def quadratic_bezier_length(p0, p1, p2, samples=24):
    total = 0.0
    prev = p0
    for i in range(1, samples + 1):
        t = i / samples
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        current = (x, y)
        total += math.dist(prev, current)
        prev = current
    return total


def parse_svg(filename):
    try:
        tree = ET.parse(filename)
    except ET.ParseError as exc:
        raise ValueError(f"Error parsing {filename}: {exc}")
    root = tree.getroot()
    ns = {'svg': root.tag[root.tag.find('{') + 1:root.tag.find('}')] if '}' in root.tag else ''}
    paths = []
    for path in root.findall('.//{*}path'):
        d = path.get('d')
        if not d:
            continue
        segments = parse_path_d(d)
        if segments:
            paths.append(segments)
    return paths


def path_endpoints(path):
    points = []
    for segment in path:
        if segment[0] in ('M', 'L'):
            points.append(segment[1])
        elif segment[0] == 'Q':
            points.append(segment[2])
    if not points:
        return None, None
    return points[0], points[-1]


def compare_paths(expected_paths, result_paths):
    exp_lens = [evaluate_path_length(p) for p in expected_paths]
    res_lens = [evaluate_path_length(p) for p in result_paths]
    exp_total = sum(exp_lens)
    res_total = sum(res_lens)
    exp_count = len(exp_lens)
    res_count = len(res_lens)

    exp_ends = [path_endpoints(p) for p in expected_paths]
    res_ends = [path_endpoints(p) for p in result_paths]

    endpoint_distances = []
    for re, ri in res_ends:
        if re is None or ri is None:
            continue
        best = min(math.dist(re, ee) + math.dist(ri, ei) for ee, ei in exp_ends)
        endpoint_distances.append(best)

    avg_endpoint_error = sum(endpoint_distances) / len(endpoint_distances) if endpoint_distances else float('inf')

    return {
        'expected_path_count': exp_count,
        'result_path_count': res_count,
        'expected_total_length': exp_total,
        'result_total_length': res_total,
        'path_count_delta': res_count - exp_count,
        'length_relative': res_total / exp_total if exp_total else float('inf'),
        'avg_endpoint_error': avg_endpoint_error,
        'expected_paths': exp_lens,
        'result_paths': res_lens,
    }


def format_diff(value, expected):
    if expected == 0:
        return f"{value:.2f}"
    return f"{value:.2f} ({(value-expected):+.2f})"


def main():
    parser = argparse.ArgumentParser(description='Evaluate generated SVG centerlines against expected SVGs.')
    parser.add_argument('--results', nargs='?', const='challenge_sample_results', default='challenge_sample_results', help='Directory containing generated SVG results')
    parser.add_argument('--expected', nargs='?', const='challenge_sample_results_expected', default='challenge_sample_results_expected', help='Directory containing expected SVG files')
    parser.add_argument('--verbose', action='store_true', help='Print per-file details')
    args = parser.parse_args()

    if not os.path.isdir(args.results):
        print(f'Results directory not found: {args.results}', file=sys.stderr)
        return 1
    if not os.path.isdir(args.expected):
        print(f'Expected directory not found: {args.expected}', file=sys.stderr)
        return 1

    files = sorted(f for f in os.listdir(args.expected) if f.lower().endswith('.svg'))
    if not files:
        print('No expected SVG files found.', file=sys.stderr)
        return 1

    summary = []
    for name in files:
        expected_file = os.path.join(args.expected, name)
        result_file = os.path.join(args.results, name)
        if not os.path.isfile(result_file):
            print(f'Missing result for {name}', file=sys.stderr)
            continue

        expected_paths = parse_svg(expected_file)
        result_paths = parse_svg(result_file)
        metrics = compare_paths(expected_paths, result_paths)
        summary.append((name, metrics))

        if args.verbose:
            print(f'--- {name} ---')
            print(f'  expected paths: {metrics["expected_path_count"]}')
            print(f'  result paths:   {metrics["result_path_count"]}')
            print(f'  total length: expected={metrics["expected_total_length"]:.1f}, result={metrics["result_total_length"]:.1f}')
            print(f'  length ratio:   {metrics["length_relative"]:.3f}')
            print(f'  avg endpoint error: {metrics["avg_endpoint_error"]:.2f}')
            print()

    if not summary:
        print('No matching output files found to evaluate.', file=sys.stderr)
        return 1

    print('Evaluation summary')
    print('---------------')
    print('Name'.ljust(25) + 'paths exp/res'.ljust(15) + 'len ratio'.ljust(12) + 'avg endpoint error')
    for name, metrics in summary:
        print(
            f'{name.ljust(25)}'
            + f'{metrics["expected_path_count"]}/{metrics["result_path_count"]}'.ljust(15)
            + f'{metrics["length_relative"]:.3f}'.ljust(12)
            + f'{metrics["avg_endpoint_error"]:.2f}'
        )

    print('\nNotes:')
    print(' - path count mismatches may indicate wrong topology or extra spurs.')
    print(' - length ratio close to 1.0 indicates similar centerline total length.')
    print(' - endpoint error is a heuristic for path correspondence; lower is better.')

    return 0


if __name__ == '__main__':
    sys.exit(main())
