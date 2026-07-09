import argparse
import json
import re
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


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / 'configs' / 'spacetime_slicer_batch.json'
DEFAULT_DATA_ROOT = REPO_ROOT / 'data'
IGNORED_DATASET_DIRS = {'Slicer', 'Slicers', 'results', '__pycache__'}
DATASET_TIME_PATTERN = re.compile(r'.*?(\d{4})-(\d{2})-(\d{2})-(\d{6})$')
REORGANIZE_OVERRIDE_OPTIONS = {
    'pre_frame_count': '--pre-frame-count',
    'camera_count': '--camera-count',
    'original_dir_name': '--original-dir-name',
    'normalized_dir_name': '--normalized-dir-name',
    'image_ext': '--image-ext',
    'dry_run': '--dry-run',
}


def resolve_root_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_config_path(value, config_path):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


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

    allowed = {'reorganize_config', 'slicer_config', 'data_root', 'output_dir'}
    if not isinstance(config, dict):
        parser.error(f'batch config file must contain a JSON object: {config_path}')
    unknown = sorted(set(config) - allowed)
    if unknown:
        parser.error(f'unknown batch config option(s): {", ".join(unknown)}')

    for key in ('reorganize_config', 'slicer_config'):
        value = config.get(key)
        if value is not None:
            config[key] = str(resolve_config_path(value, config_path))
    if config.get('data_root') is not None:
        config['data_root'] = str(resolve_root_path(config['data_root']))
    return config


def load_slicer_path_defaults(parser, path):
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open('r', encoding='utf-8') as config_file:
            config = json.load(config_file)
    except FileNotFoundError:
        parser.error(f'slicer config file not found: {config_path}')
    except json.JSONDecodeError as exc:
        parser.error(
            f'invalid JSON in slicer config {config_path}: '
            f'line {exc.lineno}, column {exc.colno}: {exc.msg}'
        )

    if not isinstance(config, dict):
        parser.error(f'slicer config file must contain a JSON object: {config_path}')

    defaults = {}
    for key in ('input_dir', 'output_dir'):
        value = config.get(key)
        if value not in (None, ''):
            defaults[key] = str(resolve_root_path(value))
    return defaults


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
        help='Single source-image directory and slicer input directory.',
    )
    parser.add_argument(
        '--sub_dir',
        help='Batch directory under data_root, such as 0630.',
    )
    parser.add_argument(
        '--data_root',
        help='Root containing batch directories; relative paths are resolved under the repo root.',
    )
    parser.add_argument(
        '--datasets',
        nargs='*',
        help='One or more dataset directory names under data_root/sub_dir.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-run even if the dataset output slicer.mp4 already exists.',
    )
    parser.add_argument(
        '--output_dir',
        help='Final slicer output root for a single selected dataset.',
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


def argv_has_option(argv, *options):
    tokens = sys.argv[1:] if argv is None else list(argv)
    for token in tokens:
        for option in options:
            if token == option or token.startswith(f'{option}='):
                return True
    return False


def dataset_output_dir(input_dir):
    return input_dir.parent / 'Slicers' / input_dir.name


def output_already_exists(output_dir):
    video_path = output_dir / 'slicer.mp4'
    return video_path.is_file() and video_path.stat().st_size > 0


def dataset_sort_key(path):
    match = DATASET_TIME_PATTERN.fullmatch(path.name)
    if match:
        year, month, day, time_value = match.groups()
        return (0, f'{year}{month}{day}{time_value}', path.name)
    return (1, path.name)


def discover_datasets(data_root, sub_dir):
    batch_dir = data_root / sub_dir
    if not batch_dir.exists():
        raise FileNotFoundError(f'batch directory does not exist: {batch_dir}')
    return sorted(
        (
            path for path in batch_dir.iterdir()
            if path.is_dir()
            and path.name not in IGNORED_DATASET_DIRS
            and not path.name.startswith('.')
        ),
        key=dataset_sort_key,
    )


def resolve_input_sub_dir(input_dir, explicit_sub_dir=None):
    if explicit_sub_dir:
        return explicit_sub_dir
    return input_dir.parent.name


def resolve_dataset_candidates(parser, args):
    data_root = Path(args.data_root).expanduser() if args.data_root else DEFAULT_DATA_ROOT
    if not data_root.is_absolute():
        data_root = resolve_root_path(data_root)
    else:
        data_root = data_root.resolve()
    args.data_root = str(data_root)

    single_input_mode = bool(args.source_dir) and (
        args.source_dir_explicit or not args.sub_dir
    )
    if single_input_mode:
        input_dir = Path(args.source_dir).expanduser()
        if not input_dir.is_absolute():
            input_dir = (REPO_ROOT / input_dir).resolve()
        else:
            input_dir = input_dir.resolve()
        args.sub_dir = resolve_input_sub_dir(input_dir, args.sub_dir)
        if args.output_dir is None or not args.output_dir_explicit:
            args.output_dir = str(dataset_output_dir(input_dir))
        args.datasets_to_process = [str(input_dir)]
        return

    if not args.sub_dir:
        parser.error('provide --sub_dir, --input_dir, or input_dir in the slicer config')
    if not args.output_dir_explicit:
        args.output_dir = None

    if args.datasets is not None:
        candidates = [data_root / args.sub_dir / name for name in args.datasets]
    else:
        try:
            discovered = discover_datasets(data_root, args.sub_dir)
        except FileNotFoundError as exc:
            parser.error(str(exc))
        if args.force:
            candidates = discovered
        else:
            candidates = [
                path for path in discovered
                if not output_already_exists(dataset_output_dir(path))
            ]

    selected = []
    skipped = []
    for candidate in candidates:
        output_dir = dataset_output_dir(candidate)
        if not args.force and output_already_exists(output_dir):
            skipped.append(str(candidate))
            continue
        selected.append(str(candidate.resolve()))

    args.datasets_to_process = selected
    args.skipped_datasets = skipped


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH))
    config_args, _ = config_parser.parse_known_args(argv)

    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    config_defaults = load_batch_config(bootstrap_parser, config_args.config)
    if config_defaults.get('slicer_config'):
        slicer_defaults = load_slicer_path_defaults(bootstrap_parser, config_defaults['slicer_config'])
        if config_defaults.get('output_dir') is None and slicer_defaults.get('output_dir'):
            config_defaults['output_dir'] = slicer_defaults['output_dir']
        if config_defaults.get('source_dir') is None and slicer_defaults.get('input_dir'):
            config_defaults['source_dir'] = slicer_defaults['input_dir']
    parser = build_parser(config_defaults)
    args, slicer_args = parser.parse_known_args(argv)
    args.source_dir_explicit = argv_has_option(argv, '-s', '--input_dir')
    args.output_dir_explicit = argv_has_option(argv, '--output_dir')

    if args.reorganize_config is None:
        parser.error('reorganize_config must be set in the batch config or command line')
    if args.slicer_config is None:
        parser.error('slicer_config must be set in the batch config or command line')
    resolve_dataset_candidates(parser, args)
    return args, slicer_args


def build_reorganize_argv(args, source_dir):
    argv = [
        '--config', args.reorganize_config,
        '--input_dir', source_dir,
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


def build_slicer_argv(args, slicer_args, source_dir, output_dir):
    return [
        '--config', args.slicer_config,
        '--input_dir', source_dir,
        '--output_dir', output_dir,
        *slicer_args,
    ]


def run_single_dataset(
    args,
    slicer_args,
    source_dir,
    output_dir,
    reorganize_func=reorganize_directory,
    slicer_main=run_spacetime_slicer,
    structure_checker=has_reorganized_frame_structure,
):
    reorganize_args = parse_reorganize_args(build_reorganize_argv(args, source_dir))
    slicer_argv = build_slicer_argv(args, slicer_args, source_dir, output_dir)
    normalize_cli_frame_args(build_slicer_parser().parse_args(slicer_argv))

    print(f'Input directory: {source_dir}')
    print(f'Output directory: {output_dir}')
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


def run_pipeline(
    args,
    slicer_args,
    reorganize_func=reorganize_directory,
    slicer_main=run_spacetime_slicer,
    structure_checker=has_reorganized_frame_structure,
):
    datasets = getattr(args, 'datasets_to_process', None) or []
    if not datasets:
        skipped = getattr(args, 'skipped_datasets', [])
        if skipped:
            print(f'All specified datasets already processed ({len(skipped)} skipped).')
        else:
            print('All datasets already processed.')
        return 0

    print(f'Will process {len(datasets)} dataset(s).')
    for source_dir in datasets:
        source_path = Path(source_dir)
        output_dir = args.output_dir
        if output_dir is None or len(datasets) > 1:
            output_dir = str(dataset_output_dir(source_path))
        result = run_single_dataset(
            args,
            slicer_args,
            str(source_path),
            output_dir,
            reorganize_func=reorganize_func,
            slicer_main=slicer_main,
            structure_checker=structure_checker,
        )
        if result:
            return result
    return 0


def main(argv=None):
    args, slicer_args = parse_args(argv)
    try:
        return run_pipeline(args, slicer_args)
    except ReorganizationError as exc:
        print(f'Reorganization failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
