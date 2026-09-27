"""Frame provenance for the slicer's existing output schedule (one-based IDs)."""


def output_frame_mapping(parameters):
    p = parameters
    mapping = []
    start, freeze, end = p['source_start_frame'], p['source_freeze_frame'], p['source_end_frame']
    slice_end = p['source_slice_end_frame']
    camera_ids = p['camera_ids']

    def add(phase, **details):
        mapping.append({'output_frame_id': len(mapping) + 1, 'phase': phase, **details})

    def repeat(phase, source_frame, camera_id, factor):
        for _ in range(factor):
            add(phase, source_frame_ids=[source_frame], camera_ids=[camera_id])

    for source in range(1, start):
        repeat('head', source, camera_ids[0], p['stretch_head'])
    for source in range(start, slice_end + 1):
        if source > start:
            for step in range(1, p['stretch_ghost']):
                add('slices', source_frame_ids=[source - 1, source], camera_ids=[camera_ids[0]],
                    interpolation_ratio=step / p['stretch_ghost'])
        add('slices', source_frame_ids=[source], camera_ids=[camera_ids[0]])

    background_source = {'freeze': freeze, 'start': start}.get(p['background_mode'])
    for offset in range(p['recovery_transition_frames']):
        add('recovery_transition', transition_step=offset + 1)
    for offset in range(p['resolved_fade_duration_frames'] * p['stretch_fade']):
        add('recovery', recovery_step=offset + 1,
            background_source_frame_id=background_source, camera_ids=[camera_ids[0]])

    for i, camera in enumerate(camera_ids):
        if p['freeze_interp_mode'] == 'repeat':
            repeat('freeze_orbit', freeze, camera, p['stretch_freeze'])
        else:
            if i:
                for step in range(1, p['stretch_freeze']):
                    add('freeze_orbit', source_frame_ids=[freeze],
                        camera_ids=[camera_ids[i - 1], camera],
                        interpolation_ratio=step / p['stretch_freeze'])
            add('freeze_orbit', source_frame_ids=[freeze], camera_ids=[camera])
    for source in range(freeze + 1, end + 1):
        repeat('tail', source, p['tail_camera_id'], p['stretch_tail'])
    return mapping
