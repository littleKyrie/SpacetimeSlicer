"""Opt-in edge strategies and compact, continuous matting caches."""
import numpy as np


EDGE_COMPOSITE_STRATEGIES = (
    'legacy', 'diagnostic_no_protection', 'soft_alpha',
    'soft_smoothstep', 'foreground_only', 'soft_foreground',
)
SOFT_EDGE_STRATEGIES = ('soft_alpha', 'soft_smoothstep', 'soft_foreground')
FOREGROUND_EDGE_STRATEGIES = ('foreground_only', 'soft_foreground')


def validate_edge_strategy(name, effect_base_mode, protect_dilate,
                           supports_foreground=False):
    if name not in EDGE_COMPOSITE_STRATEGIES:
        raise ValueError(f'Unknown edge_composite_strategy: {name}')
    if name != 'legacy' and effect_base_mode != 'source':
        raise ValueError(f'edge_composite_strategy={name} requires effect_base_mode=source')
    if name in SOFT_EDGE_STRATEGIES and protect_dilate != 0:
        raise ValueError(f'edge_composite_strategy={name} requires live_subject_protect_dilate=0')
    if name in FOREGROUND_EDGE_STRATEGIES and not supports_foreground:
        raise ValueError(f'edge_composite_strategy={name} requires a foreground-capable model (RVM)')


def cache_ghost(frame, alpha, opacity, matting=None, soft=False,
                use_foreground=False):
    """Keep legacy byte caches; pack new float16 mattes/colors into their ROI.

    ``alpha`` remains a full byte mask for the existing recovery geometry.
    Continuous alpha is normalized; foreground colors are straight BGR [0,255].
    Float16 is a cache format only: accumulation and interpolation use float32.
    """
    ghost = {'frame': None if soft and use_foreground else frame.copy(),
             'alpha': alpha.copy(), 'opacity': opacity}
    if not soft:
        if use_foreground:
            ghost['frame'] = np.clip(matting.foreground * 255, 0, 255).astype(np.uint8)
        return ghost

    continuous_alpha = matting.alpha
    support = continuous_alpha > 0
    ys = np.flatnonzero(np.any(support, axis=1))
    xs = np.flatnonzero(np.any(support, axis=0))
    if len(xs):
        x0, y0 = max(0, int(xs.min()) - 2), max(0, int(ys.min()) - 2)
        x1 = min(alpha.shape[1], int(xs.max()) + 3)
        y1 = min(alpha.shape[0], int(ys.max()) + 3)
    else:
        x0 = y0 = x1 = y1 = 0
    ghost['matte_roi'] = (x0, y0, x1, y1)
    ghost['continuous_alpha'] = continuous_alpha[y0:y1, x0:x1].astype(np.float16)
    if use_foreground:
        ghost['frame_shape'] = frame.shape
        ghost['foreground_roi'] = (
            matting.foreground[y0:y1, x0:x1] * 255
        ).astype(np.float16)
        # No additional full-resolution source/foreground color cache.
        ghost['frame'] = None
    return ghost


def ghost_alpha(ghost):
    if 'continuous_alpha' not in ghost:
        return ghost['alpha'].astype(np.float32) / 255
    alpha = np.zeros(ghost['alpha'].shape, dtype=np.float32)
    x0, y0, x1, y1 = ghost['matte_roi']
    alpha[y0:y1, x0:x1] = ghost['continuous_alpha']
    return alpha


def ghost_frame(ghost):
    if ghost.get('frame') is not None:
        return ghost['frame']
    frame = np.zeros(ghost['frame_shape'], dtype=np.float32)
    x0, y0, x1, y1 = ghost['matte_roi']
    frame[y0:y1, x0:x1] = ghost['foreground_roi']
    return frame


def compose_soft_stack(background, layers, protection=None, return_stack=False):
    """Source-over premultiplied layers, then apply live protection ONCE."""
    color = np.zeros(background.shape, dtype=np.float32)
    alpha = np.zeros(background.shape[:2], dtype=np.float32)
    for ghost, opacity in layers:
        matte = ghost_alpha(ghost)
        layer_alpha = matte * opacity
        premultiplied = ghost.get('premultiplied')
        if premultiplied is None:
            premultiplied = ghost_frame(ghost).astype(np.float32) * matte[:, :, None]
        color *= (1 - layer_alpha[:, :, None])
        color += premultiplied * opacity
        alpha *= 1 - layer_alpha
        alpha += layer_alpha
    visibility = 1.0 if protection is None else 1 - protection
    if isinstance(visibility, np.ndarray):
        visibility = visibility[:, :, None]
    effective_alpha = alpha[:, :, None] * visibility
    output = color * visibility + background.astype(np.float32) * (1 - effective_alpha)
    output = np.rint(np.clip(output, 0, 255)).astype(np.uint8)
    return (output, color, alpha) if return_stack else output
