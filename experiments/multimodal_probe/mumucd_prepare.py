"""Stage the MUMUCD Tirana optical/SAR probe (run in ovcd-ape-env on Slurm)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from PIL import Image

from cd_agent.paths import dataset_path
from cd_agent.verify import EpisodeInput

SCENE = 1536
TILE = 512
GSD_M = 10
TRACK = 'multimodal_probe'
RUN_NAME = '20261123_mumucd_tirana_probe'
S2_BANDS = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12')
S1_BANDS = ('VV', 'VH', 'angle')
T1_CHANNELS = ('B4 (red)', 'B3 (green)', 'B2 (blue)')
T2_CHANNELS = ('VV', 'VH', 'VV - VH (linear backscatter difference before stretching)')
PERCENTILES = (2, 98)
FILES = {
    't1': ('before-s2', 'sr'), 't2': ('after-s1', 'S1_data'), 'gt': ('cd-binary', 'change'),
    'dw_before': ('before-dw', 'lcc'), 'dw_after': ('after-dw', 'lcc'),
}
INSTRUCTION = (
    f'T1 is a co-registered {GSD_M} m Sentinel-2 optical true-color image; R, G, B are '
    f'{", ".join(T1_CHANNELS)} respectively. T2 is a {GSD_M} m Sentinel-1 SAR pseudo-RGB image; '
    f'R, G, B are {", ".join(T2_CHANNELS)} respectively. Each channel was independently '
    'stretched to uint8 using its scene-wide 2nd and 98th percentiles. '
    'Find significant land-cover changes between T1 and T2, in either direction and without '
    'restricting to one class. Do not treat differences caused only by the sensors, SAR speckle, '
    'season or misregistration as changes. Optical segmentation tools are intended for optical '
    'imagery (T1); T2 shows radar backscatter, not optical colors. The Python tool can compare '
    'spatial patterns and compute channel statistics on images["t1"] and images["t2"]; raw '
    'pixel-by-pixel color differences across modalities are not direct evidence of change.'
)


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def tiles():
    for row in range(SCENE // TILE):
        for col in range(SCENE // TILE):
            y, x = row * TILE, col * TILE
            yield f'tirana-r{row}c{col}', (x, y, x + TILE, y + TILE)


def stretch_rgb(rgb):
    result = np.zeros(rgb.shape, dtype=np.uint8)
    details = []
    for channel in range(3):
        band = rgb[..., channel]
        valid = np.isfinite(band)
        if not valid.any():
            raise ValueError(f'No valid pixels in channel {channel}')
        low, high = np.percentile(band[valid], PERCENTILES)
        if high <= low:
            raise ValueError(f'Collapsed 2–98% stretch in channel {channel}: {low}, {high}')
        result[..., channel][valid] = np.rint(np.clip((band[valid] - low) / (high - low), 0, 1) * 255)
        details.append(dict(low=float(low), high=float(high), invalid_pixels=int((~valid).sum())))
    return result, details


def value_counts(values):
    levels, counts = np.unique(values, return_counts=True)
    return {str(int(level)): int(count) for level, count in zip(levels, counts)}


def decode_gt(raw, nodata):
    valid = np.ones(raw.shape, bool) if nodata is None else raw != nodata
    if not valid.any() or not np.isin(raw[valid], (0, 1)).all():
        raise ValueError(f'Unexpected GT values: {value_counts(raw)} (nodata={nodata})')
    return valid & (raw == 1), valid


def read_sources(src):
    import rasterio as rio
    from rasterio.windows import Window

    arrays, metadata, grid = {}, {}, {}
    with warnings.catch_warnings(), rio.Env(GDAL_NETCDF_BOTTOMUP='NO'):
        # The files supply explicit 2D lat/lon rather than a GDAL affine/CRS.
        warnings.filterwarnings('ignore', category=rio.errors.NotGeoreferencedWarning)
        for key, (suffix, variable) in FILES.items():
            path = src / f'tirana-{suffix}.nc'
            for axis in ('lat', 'lon'):
                with rio.open(f'NETCDF:"{path}":{axis}') as ds:
                    coords = ds.read(1, masked=True).filled(np.nan)
                if coords.shape != (SCENE, SCENE) or not np.isfinite(coords).all():
                    raise ValueError(f'{path.name}: invalid {axis} grid')
                if axis not in grid:
                    grid[axis] = coords
                elif not np.allclose(coords, grid[axis], atol=1e-8, rtol=0):
                    raise ValueError(f'{path.name}: {axis} grid differs from T1')
            with rio.open(f'NETCDF:"{path}":{variable}') as ds:
                tags = ds.tags()
                if key in ('t1', 't2'):
                    s2 = key == 't1'
                    axis = 'cw' if s2 else 'dim'
                    expected = S2_BANDS if s2 else S1_BANDS
                    bands = tuple(b.strip() for b in tags.get(f'{axis}#description', '').split(','))
                    # NetCDF array order (nj, ni, channel): GDAL exposes nj as raster
                    # band, ni as raster row, channel as raster column. Keep native row order.
                    if (bands != expected or (ds.count, *ds.shape) != (SCENE, SCENE, len(expected))
                            or tags.get('NETCDF_DIM_EXTRA') != '{nj}'
                            or any(ds.tags(i).get('NETCDF_DIM_nj') != str(i) for i in (1, SCENE))
                            or ds.dtypes[0] != 'float32'):
                        raise ValueError(f'{path.name}: unexpected axes or channel order: {bands}')
                    if s2:
                        # Read B2, B3, B4 together; reverse to RGB.
                        rgb = ds.read(window=Window(1, 0, 3, SCENE), masked=True).filled(np.nan)[..., ::-1]
                    else:
                        vv, vh = np.moveaxis(ds.read(window=Window(0, 0, 2, SCENE), masked=True).filled(np.nan), -1, 0)
                        rgb = np.stack((vv, vh, vv - vh), axis=-1)
                    arrays[key], stretch = stretch_rgb(rgb)
                    dimensions = dict(nj=SCENE, ni=SCENE, **{axis: len(bands)})
                    extra = dict(native_band_order=bands, png_rgb=T1_CHANNELS if s2 else T2_CHANNELS,
                                 stretch_percentiles=PERCENTILES, stretch_scope='whole scene', stretch_rgb=stretch)
                else:
                    if (ds.count, *ds.shape) != (1, SCENE, SCENE) or ds.dtypes[0] != 'int32':
                        raise ValueError(f'{path.name}: expected a single int32 label grid')
                    arrays[key] = ds.read(1)
                    dimensions = dict(nj=SCENE, ni=SCENE)
                    extra = dict(value_description=ds.tags(1).get('description'))
                metadata[key] = dict(file=path.name, variable=variable, dimensions=dimensions,
                                     dtype=ds.dtypes[0], nodata=ds.nodata,
                                     source=tags.get('NC_GLOBAL#source_origin'),
                                     acquisition=tags.get('NC_GLOBAL#source_file'), **extra)
    return arrays, metadata


def prepare(src, out):
    arrays, metadata = read_sources(src)
    gt, valid = decode_gt(arrays['gt'], metadata['gt']['nodata'])
    if not gt.any():
        raise ValueError('Official binary GT is empty across the entire scene')
    dw1, dw2 = arrays['dw_before'], arrays['dw_after']
    dw_valid = np.isin(dw1, range(9)) & np.isin(dw2, range(9))  # class 9 is no data
    dw_changed = dw_valid & (dw1 != dw2)
    inputs, offline = out / 'inputs', out / 'offline'
    inputs.mkdir(parents=True, exist_ok=False)
    offline.mkdir(exist_ok=False)
    public, audit = [], []
    for case, (x0, y0, x1, y1) in tiles():
        region = np.s_[y0:y1, x0:x1]
        folder = inputs / case
        folder.mkdir()
        for key in ('t1', 't2'):
            Image.fromarray(arrays[key][region]).save(folder / f'{key}.png')
        record = dict(id=case, image_t1=f'{case}/t1.png', image_t2=f'{case}/t2.png', instruction=INSTRUCTION)
        EpisodeInput.from_mapping(record)
        public.append(record)

        gt_file = offline / f'{case}-gt.png'
        Image.fromarray(gt[region].astype(np.uint8) * 255).save(gt_file)
        invalid = ~valid[region]
        ignore_path = None
        if invalid.any():
            ignore = offline / f'{case}-ignore.png'
            Image.fromarray(invalid.astype(np.uint8) * 255).save(ignore)
            ignore_path = str(ignore.relative_to(out))
        audit.append(dict(
            id=case, window_xyxy=[x0, y0, x1, y1], gt_mask=str(gt_file.relative_to(out)),
            gt_ignore_mask=ignore_path, gt_raw_counts=value_counts(arrays['gt'][region]),
            gt_positive_pixels=int(gt[region].sum()), gt_nodata_pixels=int(invalid.sum()),
            dw_comparable_pixels=int(dw_valid[region].sum()), dw_changed_pixels=int(dw_changed[region].sum()),
        ))
    audit_data = dict(scene='tirana', gsd_m=GSD_M, shape=[SCENE, SCENE], tile_shape=[TILE, TILE],
                      sources=metadata, gt_raw_counts=value_counts(arrays['gt']),
                      gt_nodata_pixels=int((~valid).sum()), gt_positive_tiles=sum(row['gt_positive_pixels'] > 0 for row in audit),
                      dw_note='Raw Dynamic World classes 0–8 only; this difference is not the official filtered GT.',
                      dw_comparable_pixels=int(dw_valid.sum()), dw_changed_pixels=int(dw_changed.sum()), cases=audit)
    write_json(out / 'sample_audit.json', audit_data)
    write_json(inputs / 'cases.json', public)
    print('Prepared 9 tiles. Official GT counts:', audit_data['gt_raw_counts'],
          'positive tiles:', audit_data['gt_positive_tiles'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=dataset_path('MUMUCD', 'tirana'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.out)


if __name__ == '__main__':
    main()
