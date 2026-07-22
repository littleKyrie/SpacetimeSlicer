import argparse
import json
import re
import sys
from dataclasses import dataclass
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
DATASET_TIMESTAMP_PATTERN = re.compile(
    r'^(?P<prefix>QP.*)-(?P<timestamp>\d{4}-\d{2}-\d{2}-\d{6})$'
)
REORGANIZE_OVERRIDE_OPTIONS = {
    'pre_frame_count': '--pre-frame-count',
    'camera_count': '--camera-count',
    'original_dir_name': '--original-dir-name',
    'normalized_dir_name': '--normalized-dir-name',
    'image_ext': '--image-ext',
    'dry_run': '--dry-run',
}


@dataclass(frozen=True)
class BatchPaths:
    input_root: Path
    output_root: Path
    absolute_mode: bool


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
        raw_data_root = Path(config['data_root']).expanduser()
        config['data_root_is_absolute'] = raw_data_root.is_absolute()
        config['data_root'] = str(resolve_root_path(raw_data_root))
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
        help=(
            'Batch date directory, such as 0717. With an absolute data_root, '
            'this replaces data_root parent date directory.'
        ),
    )
    parser.add_argument(
        '--data_root',
        help=(
            'Relative batch root under the repo, or an absolute input template '
            'such as Y:/0717/关键帧.'
        ),
    )
    parser.add_argument(
        '--datasets',
        nargs='*',
        help='One or more dataset directory names under the resolved batch input root.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-run even if the dataset-named output MP4 already exists.',
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


def dataset_output_dir(output_root, input_dir):
    return output_root / input_dir.name


def dataset_video_path(output_dir):
    output_dir = Path(output_dir)
    return output_dir / f'{output_dir.name}.mp4'


def output_already_exists(output_dir):
    video_path = dataset_video_path(output_dir)
    return video_path.is_file() and video_path.stat().st_size > 0


def dataset_sort_key(path):
    match = DATASET_TIME_PATTERN.fullmatch(path.name)
    if match:
        year, month, day, time_value = match.groups()
        return (0, f'{year}{month}{day}{time_value}', path.name)
    return (1, path.name)


def parse_dataset_pre_frame_count(dataset_name):
    """Return the optional frame count encoded before a QP dataset timestamp."""
    match = DATASET_TIMESTAMP_PATTERN.fullmatch(dataset_name)
    if match is None:
        return None

    _, separator, frame_text = match.group('prefix').rpartition('_')
    if not separator:
        return None
    if not frame_text.isdecimal():
        raise ValueError(
            'dataset frame count before the timestamp must be a positive integer: '
            f'{dataset_name}'
        )

    frame_count = int(frame_text)
    if frame_count < 1:
        raise ValueError(
            'dataset frame count before the timestamp must be at least 1: '
            f'{dataset_name}'
        )
    return frame_count


def discover_datasets(input_root, qp_only=False):
    if not input_root.exists():
        raise FileNotFoundError(f'batch directory does not exist: {input_root}')
    if not input_root.is_dir():
        raise NotADirectoryError(f'batch path is not a directory: {input_root}')
    return sorted(
        (
            path for path in input_root.iterdir()
            if path.is_dir()
            and path.name not in IGNORED_DATASET_DIRS
            and not path.name.startswith('.')
            and (not qp_only or path.name.startswith('QP'))
        ),
        key=dataset_sort_key,
    )


def validate_sub_dir(parser, sub_dir):
    if (
        not sub_dir
        or sub_dir in {'.', '..'}
        or '/' in sub_dir
        or '\\' in sub_dir
        or Path(sub_dir).is_absolute()
        or Path(sub_dir).name != sub_dir
    ):
        parser.error('--sub_dir must be a single directory name without path separators')


def resolve_batch_paths(data_root, sub_dir, absolute_mode):
    if not absolute_mode:
        input_root = data_root / sub_dir
        return BatchPaths(input_root, input_root / 'Slicers', False)

    date_template_dir = data_root.parent
    if not data_root.name or not date_template_dir.name:
        raise ValueError(
            'absolute data_root must have the form '
            '<fixed-prefix>/<date>/<input-directory>'
        )
    date_dir = date_template_dir.parent / sub_dir
    return BatchPaths(
        input_root=date_dir / data_root.name,
        output_root=date_dir / '风暴时刻输出',
        absolute_mode=True,
    )


def validate_explicit_dataset(parser, input_root, name, qp_only):
    name_path = Path(name)
    if name_path.is_absolute() or name_path.name != name or '/' in name or '\\' in name:
        parser.error(f'dataset name must be a single directory name: {name}')
    if qp_only and not name.startswith('QP'):
        parser.error(f'dataset name must start with QP in absolute data_root mode: {name}')
    candidate = input_root / name
    if qp_only and (not candidate.exists() or not candidate.is_dir()):
        parser.error(f'dataset directory does not exist: {candidate}')
    return candidate


def resolve_input_sub_dir(input_dir, explicit_sub_dir=None):
    if explicit_sub_dir:
        return explicit_sub_dir
    return input_dir.parent.name


def resolve_dataset_candidates(parser, args):
    data_root = Path(args.data_root).expanduser() if args.data_root else DEFAULT_DATA_ROOT
    data_root_is_absolute = getattr(args, 'data_root_is_absolute', False)
    if args.data_root_explicit:
        data_root_is_absolute = data_root.is_absolute()
    if not data_root.is_absolute():
        data_root = resolve_root_path(data_root)
    else:
        data_root = data_root.resolve()
    args.data_root = str(data_root)
    args.data_root_is_absolute = data_root_is_absolute

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
            args.output_dir = str(input_dir.parent / 'Slicers' / input_dir.name)
        args.datasets_to_process = [str(input_dir)]
        args.output_dirs_to_process = [args.output_dir]
        try:
            args.dataset_pre_frame_counts = [
                parse_dataset_pre_frame_count(input_dir.name)
            ]
        except ValueError as exc:
            parser.error(str(exc))
        args.batch_input_root = None
        args.batch_output_root = None
        return

    if not args.sub_dir:
        parser.error('provide --sub_dir, --input_dir, or input_dir in the slicer config')
    validate_sub_dir(parser, args.sub_dir)
    if not args.output_dir_explicit:
        args.output_dir = None

    try:
        batch_paths = resolve_batch_paths(
            data_root,
            args.sub_dir,
            data_root_is_absolute,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.batch_input_root = str(batch_paths.input_root)
    args.batch_output_root = str(batch_paths.output_root)
    if not batch_paths.input_root.exists():
        parser.error(f'batch directory does not exist: {batch_paths.input_root}')
    if not batch_paths.input_root.is_dir():
        parser.error(f'batch path is not a directory: {batch_paths.input_root}')

    if args.datasets is not None:
        candidates = [
            validate_explicit_dataset(
                parser,
                batch_paths.input_root,
                name,
                batch_paths.absolute_mode,
            )
            for name in args.datasets
        ]
    else:
        try:
            discovered = discover_datasets(
                batch_paths.input_root,
                qp_only=batch_paths.absolute_mode,
            )
        except (FileNotFoundError, NotADirectoryError) as exc:
            parser.error(str(exc))
        candidates = discovered

    selected = []
    selected_outputs = []
    selected_pre_frame_counts = []
    for candidate in candidates:
        if args.output_dir_explicit and len(candidates) == 1:
            output_dir = Path(args.output_dir).expanduser()
        else:
            output_dir = dataset_output_dir(batch_paths.output_root, candidate)
        try:
            pre_frame_count = parse_dataset_pre_frame_count(candidate.name)
        except ValueError as exc:
            parser.error(str(exc))
        selected.append(str(candidate.resolve()))
        selected_outputs.append(str(output_dir))
        selected_pre_frame_counts.append(pre_frame_count)

    args.datasets_to_process = selected
    args.output_dirs_to_process = selected_outputs
    args.dataset_pre_frame_counts = selected_pre_frame_counts


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
    args.data_root_explicit = argv_has_option(argv, '--data_root')
    args.pre_frame_count_explicit = argv_has_option(argv, '--pre-frame-count')
    args.start_frame_explicit = argv_has_option(slicer_args, '--start_frame')
    args.freeze_frame_explicit = argv_has_option(slicer_args, '--freeze_frame')

    explicit_frame_parser = argparse.ArgumentParser(add_help=False)
    explicit_frame_parser.add_argument('--start_frame', type=int)
    explicit_frame_parser.add_argument('--freeze_frame', type=int)
    explicit_frame_args, _ = explicit_frame_parser.parse_known_args(slicer_args)
    args.explicit_start_frame = explicit_frame_args.start_frame
    args.explicit_freeze_frame = explicit_frame_args.freeze_frame

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


def build_slicer_argv(
    args,
    slicer_args,
    source_dir,
    output_dir,
    source_sequence_dir=None,
):
    argv = [
        '--config', args.slicer_config,
        '--input_dir', source_dir,
        '--output_dir', output_dir,
    ]
    if source_sequence_dir is not None:
        argv.extend(['--source_sequence_dir', str(source_sequence_dir)])
    argv.extend(slicer_args)
    return argv


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
    source_sequence_dir = (
        Path(source_dir) / reorganize_args.normalized_dir_name
    )
    slicer_argv = build_slicer_argv(
        args,
        slicer_args,
        source_dir,
        output_dir,
        source_sequence_dir=source_sequence_dir,
    )
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
    batch_input_root = getattr(args, 'batch_input_root', None)
    batch_output_root = getattr(args, 'batch_output_root', None)
    if batch_input_root is not None:
        mode = 'absolute' if args.data_root_is_absolute else 'relative'
        print(f'Data-root mode: {mode}')
        print(f'Input root: {batch_input_root}')
        print(f'Output root: {batch_output_root}')

    if not datasets:
        print('No datasets to process.')
        return 0

    output_dirs = getattr(args, 'output_dirs_to_process', None) or []
    dataset_pre_frame_counts = getattr(args, 'dataset_pre_frame_counts', None) or []
    print(f'Will process {len(datasets)} dataset(s).')
    for index, source_dir in enumerate(datasets):
        source_path = Path(source_dir)
        if index < len(output_dirs):
            output_dir = output_dirs[index]
        elif batch_output_root is not None:
            output_dir = str(dataset_output_dir(Path(batch_output_root), source_path))
        else:
            output_dir = args.output_dir
        pre_frame_count = (
            dataset_pre_frame_counts[index]
            if index < len(dataset_pre_frame_counts)
            else parse_dataset_pre_frame_count(source_path.name)
        )
        dataset_args = argparse.Namespace(**vars(args))
        dataset_slicer_args = list(slicer_args)
        if pre_frame_count is not None:
            manual_pre_frame_count = (
                args.pre_frame_count
                if args.pre_frame_count_explicit
                else None
            )
            manual_freeze_frame = (
                args.explicit_freeze_frame
                if args.freeze_frame_explicit
                else None
            )
            if (
                manual_pre_frame_count is not None
                and manual_freeze_frame is not None
                and manual_pre_frame_count != manual_freeze_frame
            ):
                raise ReorganizationError(
                    'explicit --pre-frame-count and --freeze_frame must match '
                    f'for dataset {source_path.name}: '
                    f'{manual_pre_frame_count} != {manual_freeze_frame}'
                )

            effective_frame_count = (
                manual_pre_frame_count
                if manual_pre_frame_count is not None
                else manual_freeze_frame
                if manual_freeze_frame is not None
                else pre_frame_count
            )
            if not args.pre_frame_count_explicit:
                dataset_args.pre_frame_count = effective_frame_count
            if not args.start_frame_explicit:
                dataset_slicer_args.extend(['--start_frame', '1'])
            if not args.freeze_frame_explicit:
                dataset_slicer_args.extend([
                    '--freeze_frame', str(effective_frame_count)
                ])
            effective_start_frame = (
                args.explicit_start_frame
                if args.start_frame_explicit
                else 1
            )
            print(
                f'Dataset frame metadata: encoded={pre_frame_count}; effective '
                f'pre_frame_count={effective_frame_count}, '
                f'start_frame={effective_start_frame}, '
                f'freeze_frame={effective_frame_count}'
            )
        result = run_single_dataset(
            dataset_args,
            dataset_slicer_args,
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
