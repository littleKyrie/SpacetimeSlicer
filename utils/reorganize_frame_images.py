import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
NUMBER_PATTERN = re.compile(r'(\d+)')
DEFAULT_ORIGINAL_DIR_NAME = '原始图片'
DEFAULT_NORMALIZED_DIR_NAME = '重命名数据'


class ReorganizationError(ValueError):
    pass


@dataclass(frozen=True)
class NumberedImage:
    sequence_id: int
    path: Path


@dataclass(frozen=True)
class CopyOperation:
    source: Path
    destination: Path


def extract_last_number(path):
    matches = NUMBER_PATTERN.findall(path.stem)
    if not matches:
        raise ReorganizationError(f'Image filename has no numeric id: {path.name}')
    return int(matches[-1])


def normalize_extension(value):
    if not value:
        raise ReorganizationError('image extension cannot be empty')
    if not value.startswith('.'):
        value = f'.{value}'
    return value.lower()


def discover_images(input_dir):
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ReorganizationError(f'Input directory does not exist: {input_path}')

    numbered_images = []
    seen_ids = {}
    for path in input_path.iterdir():
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        sequence_id = extract_last_number(path)
        if sequence_id in seen_ids:
            first = seen_ids[sequence_id].name
            raise ReorganizationError(
                f'Duplicate numeric id {sequence_id}: {first}, {path.name}'
            )
        seen_ids[sequence_id] = path
        numbered_images.append(NumberedImage(sequence_id, path))

    if not numbered_images:
        raise ReorganizationError(f'No supported image files found in: {input_path}')

    return sorted(numbered_images, key=lambda image: image.sequence_id)


def make_unique_operations(operations):
    unique_by_destination = {}
    ordered_operations = []

    for operation in operations:
        existing = unique_by_destination.get(operation.destination)
        if existing is None:
            unique_by_destination[operation.destination] = operation
            ordered_operations.append(operation)
            continue

        if existing.source != operation.source:
            raise ReorganizationError(
                'Conflicting copy targets: '
                f'{existing.source} and {operation.source} both write {operation.destination}'
            )

    return ordered_operations


def build_copy_plan(
    input_dir,
    images,
    pre_frame_count=125,
    camera_count=90,
    original_dir_name=DEFAULT_ORIGINAL_DIR_NAME,
    normalized_dir_name=DEFAULT_NORMALIZED_DIR_NAME,
    image_ext='.jpg',
):
    if pre_frame_count < 1:
        raise ReorganizationError('pre_frame_count must be at least 1')
    if camera_count < 1:
        raise ReorganizationError('camera_count must be at least 1')

    required_count = pre_frame_count + camera_count + 1
    if len(images) < required_count:
        raise ReorganizationError(
            f'Not enough images: need at least {required_count}, found {len(images)}'
        )

    input_path = Path(input_dir)
    image_ext = normalize_extension(image_ext)
    original_dir = input_path / original_dir_name
    normalized_dir = input_path / normalized_dir_name

    normalized_paths = [
        normalized_dir / f'{index:03d}{image_ext}' for index in range(len(images))
    ]
    operations = []

    for image in images:
        operations.append(CopyOperation(image.path, original_dir / image.path.name))

    for image, normalized_path in zip(images, normalized_paths):
        operations.append(CopyOperation(image.path, normalized_path))

    for source_index in range(pre_frame_count):
        frame_dir = input_path / f'{source_index + 1:04d}'
        operations.append(
            CopyOperation(normalized_paths[source_index], frame_dir / f'001{image_ext}')
        )

    effect_frame_dir = input_path / f'{pre_frame_count:04d}'
    effect_start_index = pre_frame_count - 1
    effect_end_index = pre_frame_count + camera_count
    for camera_index, source_index in enumerate(
        range(effect_start_index, effect_end_index), start=1
    ):
        operations.append(
            CopyOperation(
                normalized_paths[source_index],
                effect_frame_dir / f'{camera_index:03d}{image_ext}',
            )
        )

    following_start_index = pre_frame_count + camera_count
    for source_index in range(following_start_index, len(images)):
        frame_number = pre_frame_count + 1 + source_index - following_start_index
        frame_dir = input_path / f'{frame_number:04d}'
        operations.append(
            CopyOperation(normalized_paths[source_index], frame_dir / f'001{image_ext}')
        )

    return make_unique_operations(operations)


def execute_copy_plan(operations, root_original_paths=None, dry_run=False):
    root_original_paths = root_original_paths or []
    if dry_run:
        for operation in operations:
            print(f'COPY {operation.source} -> {operation.destination}')
        for path in root_original_paths:
            print(f'DELETE {path}')
        return

    existing_destinations = [
        operation.destination for operation in operations if operation.destination.exists()
    ]
    if existing_destinations:
        preview = ', '.join(str(path) for path in existing_destinations[:5])
        if len(existing_destinations) > 5:
            preview = f'{preview}, ...'
        raise ReorganizationError(f'Target file already exists: {preview}')

    for operation in operations:
        operation.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(operation.source, operation.destination)

    for path in root_original_paths:
        path.unlink()


def reorganize_directory(
    input_dir,
    pre_frame_count=125,
    camera_count=90,
    original_dir_name=DEFAULT_ORIGINAL_DIR_NAME,
    normalized_dir_name=DEFAULT_NORMALIZED_DIR_NAME,
    image_ext='.jpg',
    dry_run=False,
):
    images = discover_images(input_dir)
    operations = build_copy_plan(
        input_dir,
        images,
        pre_frame_count=pre_frame_count,
        camera_count=camera_count,
        original_dir_name=original_dir_name,
        normalized_dir_name=normalized_dir_name,
        image_ext=image_ext,
    )
    execute_copy_plan(
        operations,
        root_original_paths=[image.path for image in images],
        dry_run=dry_run,
    )
    return operations


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Reorganize numbered source images into frame/camera directories.'
    )
    parser.add_argument(
        'input_dir_positional',
        nargs='?',
        help='Directory containing source image files.',
    )
    parser.add_argument(
        '--input_dir',
        dest='input_dir_option',
        metavar='INPUT_DIR',
        help='Directory containing source image files.',
    )
    parser.add_argument(
        '--pre-frame-count',
        type=int,
        default=125,
        help='Number of ordinary frames before the effect frame. Default: 125.',
    )
    parser.add_argument(
        '--camera-count',
        type=int,
        default=90,
        help='Effect camera parameter. The effect frame writes camera_count + 1 files.',
    )
    parser.add_argument(
        '--original-dir-name',
        default=DEFAULT_ORIGINAL_DIR_NAME,
        help='Subdirectory name used to keep a copy of original files.',
    )
    parser.add_argument(
        '--normalized-dir-name',
        default=DEFAULT_NORMALIZED_DIR_NAME,
        help='Subdirectory name used for 000.jpg, 001.jpg, ... files.',
    )
    parser.add_argument(
        '--image-ext',
        default='.jpg',
        help='Output filename extension. Files are copied without re-encoding.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print planned copy operations without creating directories or files.',
    )
    args = parser.parse_args(argv)
    args.input_dir = args.input_dir_option or args.input_dir_positional
    if args.input_dir is None:
        parser.error('input directory is required: pass input_dir or --input_dir')
    del args.input_dir_option
    del args.input_dir_positional
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        operations = reorganize_directory(
            args.input_dir,
            pre_frame_count=args.pre_frame_count,
            camera_count=args.camera_count,
            original_dir_name=args.original_dir_name,
            normalized_dir_name=args.normalized_dir_name,
            image_ext=args.image_ext,
            dry_run=args.dry_run,
        )
    except ReorganizationError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1

    action = 'Planned' if args.dry_run else 'Completed'
    print(f'{action} {len(operations)} copy operations.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
