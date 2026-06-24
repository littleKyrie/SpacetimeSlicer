import argparse
import json
import sys
from pathlib import Path

from build_spacetime_slicer import (
    build_parser as build_slicer_parser,
    main as run_spacetime_slicer,
    normalize_cli_frame_args,
)
from utils.reorganize_frame_images import (
    ReorganizationError,
    has_reorganized_frame_structure,
    parse_args as parse_reorganize_args,
    reorganize_directory,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / 'configs' / 'spacetime_slicer_batch.json'
REORGANIZE_OVERRIDE_OPTIONS = {
    'pre_frame_count': '--pre-frame-count',
    'camera_count': '--camera-count',
    'original_dir_name': '--original-dir-name',
    'normalized_dir_name': '--normalized-dir-name',
    'image_ext': '--image-ext',
    'dry_run': '--dry-run',
}


def load_batch_config(parser, path):
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open('r', encoding='utf-8') as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        parser.error(f'batch config file not found: {config_path}')
    except json.JSONDecodeError as exc:
        parser.error(
            f'invalid JSON in batch config {config_path}: '
            f'line {exc.lineno}, column {exc.colno}: {exc.msg}'
        )

    allowed = {'reorganize_config', 'slicer_config', 'output_dir'}
    if not isinstance(config, dict):
        parser.error(f'batch config file must contain a JSON object: {config_path}')
    unknown = sorted(set(config) - allowed)
    if unknown:
        parser.error(f'unknown batch config option(s): {", ".join(unknown)}')

    for key in ('reorganize_config', 'slicer_config'):
        value = config.get(key)
        if value is not None:
            value_path = Path(value).expanduser()
            if not value_path.is_absolute():
                value_path = config_path.parent / value_path
            config[key] = str(value_path.resolve())
    return config


def build_parser(config_defaults=None):
    parser = argparse.ArgumentParser(
        description=(
            'Reuse an existing frame layout or reorganize source images, then '
            'generate a spacetime-slicer video. '
            'Unrecognized options are forwarded to build_spacetime_slicer.py.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH), help='Batch JSON config.')
    parser.add_argument(
        '-s',
        '--input_dir',
        dest='source_dir',
        required=True,
        help='Original source-image directory and slicer input directory.',
    )
    parser.add_argument(
        '--output_dir',
        help='Final slicer output root; defaults to ./results/<source-directory-name>.',
    )
    parser.add_argument('--reorganize-config', help='JSON config for image reorganization.')
    parser.add_argument('--slicer-config', help='JSON config for spacetime slicing.')
    parser.add_argument('--pre-frame-count', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--camera-count', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--original-dir-name', default=argparse.SUPPRESS)
    parser.add_argument('--normalized-dir-name', default=argparse.SUPPRESS)
    parser.add_argument('--image-ext', default=argparse.SUPPRESS)
    parser.add_argument(
        '--dry-run',
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help='Preview reorganization and skip slicing.',
    )
    if config_defaults:
        parser.set_defaults(**config_defaults)
    return parser


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    config_args, _ = config_parser.parse_known_args(argv)

    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    config_defaults = load_batch_config(bootstrap_parser, config_args.config)
    parser = build_parser(config_defaults)
    args, slicer_args = parser.parse_known_args(argv)

    source_path = Path(args.source_dir).expanduser()
    if args.output_dir is None:
        args.output_dir = str(Path.cwd() / 'results' / source_path.name)
    if args.reorganize_config is None:
        parser.error('reorganize_config must be set in the batch config or command line')
    if args.slicer_config is None:
        parser.error('slicer_config must be set in the batch config or command line')
    return args, slicer_args


def build_reorganize_argv(args):
    argv = [
        '--config', args.reorganize_config,
        '--input_dir', args.source_dir,
    ]
    for dest, option in REORGANIZE_OVERRIDE_OPTIONS.items():
        if not hasattr(args, dest):
            continue
        value = getattr(args, dest)
        if dest == 'dry_run':
            argv.append('--dry-run' if value else '--no-dry-run')
        else:
            argv.extend([option, str(value)])
    return argv


def build_slicer_argv(args, slicer_args):
    return [
        '--config', args.slicer_config,
        '--input_dir', args.source_dir,
        '--output_dir', args.output_dir,
        *slicer_args,
    ]


def run_pipeline(
    args,
    slicer_args,
    reorganize_func=reorganize_directory,
    slicer_main=run_spacetime_slicer,
    structure_checker=has_reorganized_frame_structure,
):
    reorganize_args = parse_reorganize_args(build_reorganize_argv(args))
    slicer_argv = build_slicer_argv(args, slicer_args)
    normalize_cli_frame_args(build_slicer_parser().parse_args(slicer_argv))

    print(f'Input directory: {args.source_dir}')
    print(f'Output directory: {args.output_dir}')
    is_reorganized = structure_checker(
        reorganize_args.input_dir,
        pre_frame_count=reorganize_args.pre_frame_count,
        camera_count=reorganize_args.camera_count,
        image_ext=reorganize_args.image_ext,
    )

    if is_reorganized:
        print('Step 1/2: input data already matches the frame layout; preprocessing skipped.')
    else:
        print('Step 1/2: reorganizing source images...')
        operations = reorganize_func(
            reorganize_args.input_dir,
            pre_frame_count=reorganize_args.pre_frame_count,
            camera_count=reorganize_args.camera_count,
            original_dir_name=reorganize_args.original_dir_name,
            normalized_dir_name=reorganize_args.normalized_dir_name,
            image_ext=reorganize_args.image_ext,
            dry_run=reorganize_args.dry_run,
        )
        action = 'Planned' if reorganize_args.dry_run else 'Completed'
        print(f'{action} {len(operations)} copy operations.')

    if reorganize_args.dry_run:
        print('Dry run complete; slicing was skipped.')
        return 0

    print('Step 2/2: generating spacetime slices...')
    return slicer_main(slicer_argv)


def main(argv=None):
    args, slicer_args = parse_args(argv)
    try:
        return run_pipeline(args, slicer_args)
    except ReorganizationError as exc:
        print(f'Reorganization failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
