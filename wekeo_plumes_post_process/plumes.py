#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:23:12 2026

@author: spipien

Détection de panaches CO à partir de S5P-PCA griddé
(Traitement morphologique image binaire)
"""

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.colors import ListedColormap

import warnings
from pathlib import Path
import cv2 as cv
from scipy.ndimage import generate_binary_structure, label
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore", category=DeprecationWarning)

from wekeo_plumes_post_process import config


# Résoudre les chaînes (ex: 360→358→354)
def resolve_chain(merge_map):
    """
    Resolve transitive merge chains to their ultimate target label.

    Given a mapping where each key should be replaced by its value,
    follows chains until a stable root label is reached.
    For example, {354: 350, 358: 354, 360: 358} resolves to
    {354: 350, 358: 350, 360: 350}.

    Parameters
    ----------
    merge_map : dict
        Mapping of {old_label: new_label} pairs representing direct merges.

    Returns
    -------
    dict
        Mapping of {old_label: root_label} where root_label is the final
        target after following the full chain.
    """
    resolved = {}
    for key in merge_map:
        target = key
        while target in merge_map:
            target = merge_map[target]
        resolved[key] = target
    return resolved


def map_plumes(list_lon, list_lat, zone, titre, date, output=None,
               ll_lon=-180, ur_lon=180, ll_lat=-90, ur_lat=90, point_size=2):
    """
    Generate a global map of detected plumes coloured by label.

    Each element of list_lon / list_lat corresponds to one plume or group
    and is drawn with a distinct colour from the tab20/tab20b/tab20c palettes.
    The figure is displayed in notebook. If output is provided, it is also saved as PNG.

    Parameters
    ----------
    list_lon : list of array-like
        Longitude arrays, one per plume/group.
    list_lat : list of array-like
        Latitude arrays, one per plume/group (same order as list_lon).
    zone : str
        Label for the geographic zone shown in the title and file name (e.g. "GLOBAL").
    titre : str
        Plot type key. Accepted values: "plumes", "small_groups", "discarded",
        "plumes_and_small". Controls the title string and output file name.
    date : str
        Date string used in the title and file name (e.g. "20210814").
    output : str, optional
        Base path/prefix for the output PNG file (directory + file stem).
        If None, plot is only displayed, not saved.
    ll_lon : float, optional
        Left longitude of the map extent. Default -180.
    ur_lon : float, optional
        Right longitude of the map extent. Default 180.
    ll_lat : float, optional
        Lower latitude of the map extent. Default -90.
    ur_lat : float, optional
        Upper latitude of the map extent. Default 90.
    point_size : float, optional
        Marker size passed to `scatter`. Default 2.
    """

    dlon, dlat = 30, 20
    xticks = np.arange(-180, 180.1, dlon)
    yticks = np.arange(-90, 90.1, dlat)

    fig = plt.figure(figsize=(16, 8))
    crs_latlon = ccrs.PlateCarree()
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))

    ax.coastlines()
    ax.add_feature(cf.BORDERS)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                      linewidth=1, linestyle=":", color="k", alpha=0.8)
    gl.xlocator = mticker.FixedLocator(xticks)
    gl.ylocator = mticker.FixedLocator(yticks)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_extent((ll_lon, ur_lon, ll_lat, ur_lat), crs=crs_latlon)

    colors_combined = (list(plt.cm.get_cmap("tab20").colors)
                       + list(plt.cm.get_cmap("tab20b").colors)
                       + list(plt.cm.get_cmap("tab20c").colors))
    num_features = len(list_lon)
    cmap = ListedColormap(colors_combined[:max(num_features, 1)])

    for i in range(num_features):
        color = cmap(i / max(1, num_features - 1))
        ax.scatter(list_lon[i], list_lat[i], s=point_size, color=color,
                   transform=ccrs.PlateCarree(), label=f"{i+1} : {len(list_lon[i])}")

    titles = {
        "plumes":            f"S5P FOV {zone} {date} — {num_features} detected plumes",
        "small_groups":      f"S5P FOV {zone} {date} — {num_features} small groups",
        "discarded":         f"S5P FOV {zone} {date} — pixels removed",
        "plumes_and_small":  f"S5P FOV {zone} {date} — panaches + small groups",
    }
    title_str = titles.get(titre, f"S5P FOV {zone} {date}")

    ax.legend(loc="upper left", fontsize=16, markerscale=2,
              bbox_to_anchor=(1.02, 1), ncol=2)

    if output is not None:
        filenames = {
            "plumes":            f"{output}_plumes_{zone}_{date}",
            "small_groups":      f"{output}_small_groups_{zone}_{date}",
            "discarded":         f"{output}_discarded_{zone}_{date}",
            "plumes_and_small":  f"{output}_plumes_and_small_{zone}_{date}",
        }
        filename = filenames.get(titre, f"{output}_FIG_Map_{zone}_{date}")
        pngfile = filename + '.png'
        print(pngfile)
        fig.savefig(pngfile, bbox_inches="tight", pad_inches=0.01)
    
    plt.show()


def region_props_numpy(labels, label_id):
    """
    Compute geometric properties of a labelled region using NumPy.

    Calculates the centroid, bounding box, and principal-axis lengths
    (major/minor) and orientation via the second-order central moments
    (covariance matrix of pixel coordinates).

    Parameters
    ----------
    labels : ndarray of int, shape (H, W)
        2-D label array where each connected region has a unique integer id
        and the background is 0.
    label_id : int
        The label value of the region to analyse.

    Returns
    -------
    dict or None
        None if the region contains no pixels, otherwise a dict with keys:

        - ``n_pixels``        : int   — number of pixels in the region.
        - ``row_centroid``    : float — mean row index.
        - ``col_centroid``    : float — mean column index.
        - ``major_axis_px``   : float — length of the major axis in pixels
                                        (4 * sqrt(largest eigenvalue)).
        - ``minor_axis_px``   : float — length of the minor axis in pixels.
        - ``orientation_rad`` : float — angle of the major axis w.r.t. the
                                        horizontal, in radians ∈ (-π/2, π/2].
        - ``bbox``            : tuple (r0, c0, r1, c1) — bounding box
                                        (min row, min col, max row, max col).
    """
    rows, cols = np.where(labels == label_id)
    n = len(rows)

    if n == 0:
        return None

    row_c = float(np.mean(rows))
    col_c = float(np.mean(cols))
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    if n < 2:
        return {
            "n_pixels"       : n,
            "row_centroid"   : row_c,
            "col_centroid"   : col_c,
            "major_axis_px"  : 1.0,
            "minor_axis_px"  : 1.0,
            "orientation_rad": 0.0,
            "bbox"           : (r0, c0, r1, c1),
        }

    rows_c = rows - row_c
    cols_c = cols - col_c
    mu20 = float(np.sum(rows_c ** 2)) / n
    mu02 = float(np.sum(cols_c ** 2)) / n
    mu11 = float(np.sum(rows_c * cols_c)) / n

    cov = np.array([[mu20, mu11], [mu11, mu02]])
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues  = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    major_axis = 4.0 * np.sqrt(max(eigenvalues[0], 0.0))
    minor_axis = 4.0 * np.sqrt(max(eigenvalues[1], 0.0))

    drow, dcol = eigenvectors[:, 0]
    orientation_rad = float(np.arctan2(-drow, dcol))
    if orientation_rad > np.pi / 2:
        orientation_rad -= np.pi
    elif orientation_rad < -np.pi / 2:
        orientation_rad += np.pi

    return {
        "n_pixels"       : n,
        "row_centroid"   : row_c,
        "col_centroid"   : col_c,
        "major_axis_px"  : float(major_axis),
        "minor_axis_px"  : float(minor_axis),
        "orientation_rad": orientation_rad,
        "bbox"           : (r0, c0, r1, c1),
    }


def compute_plume_attributes(labels, lat_grid, lon_grid):
    """
    Build a DataFrame of per-plume attributes from a labelled grid.

    For every non-zero label in `labels`, the function calls
    `region_props_numpy` and converts the pixel-space centroid to
    geographic coordinates using linear interpolation on the lat/lon grids.

    Parameters
    ----------
    labels : ndarray of int, shape (H, W)
        2-D label array (0 = background, positive integers = plume ids).
    lat_grid : array-like, shape (H,)
        Latitude values corresponding to each row of `labels`.
    lon_grid : array-like, shape (W,)
        Longitude values corresponding to each column of `labels`.

    Returns
    -------
    pd.DataFrame
        One row per labelled region with columns:
        ``label``, ``n_pixels``, ``lat_centroid``, ``lon_centroid``.
    """
    height, width = labels.shape
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    records = []
    for lbl in unique_labels:
        p = region_props_numpy(labels, lbl)
        if p is None:
            continue
        lat_c = float(np.interp(p["row_centroid"], np.arange(height), lat_grid))
        lon_c = float(np.interp(p["col_centroid"], np.arange(width),  lon_grid))
        records.append({
            "label"       : int(lbl),
            "n_pixels"    : p["n_pixels"],
            "lat_centroid": lat_c,
            "lon_centroid": lon_c
        })

    return pd.DataFrame(records)


def save_outputs(labels, lat_grid, lon_grid, image2, dir_out, date_str):
    """
    Save the labelled plume grid to a NetCDF file.

    The output dataset contains a single variable ``plume_labels`` on a
    (latitude, longitude) grid.  Label conventions:

    - 0   : background (no detection).
    - 1-99: main plumes (ordered by size after filtering).
    - ≥100: small groups that were not attached to a large plume.

    Parameters
    ----------
    labels : ndarray of int32, shape (H, W)
        Final label array produced by the detection pipeline.
    lat_grid : array-like, shape (H,)
        Latitude coordinate values.
    lon_grid : array-like, shape (W,)
        Longitude coordinate values.
    image2 : ndarray
        Extended binary image (unused in the current implementation,
        kept for API compatibility).
    dir_out : str
        Output directory path (must exist or be created before calling).
    date_str : str
        Date string appended to the output file name (e.g. "20210814").
    """
    ds_out = xr.Dataset({
        "plume_labels": xr.DataArray(
            labels.astype(np.int32),
            coords={"latitude": lat_grid, "longitude": lon_grid},
            dims=("latitude", "longitude"),
            attrs={"long_name": "Plume label (0=background, 1-99=plumes, >=100=small groups)", "units": "1"}
        ),
    })
    nc_file = dir_out + f"plume_labels_{date_str}.nc"
    ds_out.to_netcdf(nc_file)





def apply_plume_detection(s5p_pca_l3: Path | xr.Dataset) -> xr.Dataset:
    """
    Detect CO plumes from gridded S5P-PCA data using morphological image processing.

    Parameters
    ----------
    s5p_pca_l3 : Path | xr.Dataset
        Either a Path to the S5P-PCA L3 NetCDF file, or an already-loaded xarray Dataset.

    Returns
    -------
    xr.Dataset
        Input dataset with new variable 'plume_labels' added.
        Label convention:
        - 0: background (no detection)
        - 1-99: main plumes
        - >=100: small groups not attached to large plumes
    """
    # ======================
    # PARAMETRES / CONSTANTS
    # ======================
    MIN_PLUME_SIZE = 70
    KERNEL_STR = "elliptique"
    STRUCTURE_STR = "structure_8"
    ELLIPT_SIZE = 15
    SCORE_CO_SMALL = 0.4    # seuil pour groupes >= 2 px (CO_1 ET CO_2)
    SCORE_CO_HIGH = 0.90    # seuil "extrême" pour pixels isolés (1 pixel)
    MAX_DIST_PX = 20        # ~2.2° à 0.11°/px
    SMALL_GROUP_OFFSET = 100  # numérotation des small groups
    VAR_NAME = "nb_detect_mean"

    # Load NetCDF file if Path provided, otherwise use Dataset directly
    if isinstance(s5p_pca_l3, xr.Dataset):
        ds = s5p_pca_l3
        s5p_pca_l3_path = None  # No path available
    else:
        s5p_pca_l3_path = Path(s5p_pca_l3)
        ds = xr.open_dataset(s5p_pca_l3_path)

    # Infer date from dataset attributes or filename
    if 'time_coverage_start' in ds.attrs:
        date_str = ds.attrs['time_coverage_start'][:10].replace('-', '')
    elif 'date' in ds.attrs:
        date_str = str(ds.attrs['date']).replace('-', '')
    else:
        # Extract from filename: s5p_pca_l3_2021-08-14_...
        if s5p_pca_l3_path is not None:
            import re
            match = re.search(r'(\d{4})-(\d{2})-(\d{2})', s5p_pca_l3_path.name)
            if match:
                date_str = ''.join(match.groups())
            else:
                raise ValueError("Cannot infer date from dataset attributes or filename")
        else:
            raise ValueError("Cannot infer date from dataset attributes and no filename available")

    print('****************')
    print('** ' + date_str + ' **')
    print('****************')

    # Setup output directory
    dir_out = config.output_dir / date_str
    dir_out.mkdir(parents=True, exist_ok=True)
    dir_out = str(dir_out) + '/'

    # Load data from dataset
    lat_grid = ds["latitude"].values
    lon_grid = ds["longitude"].values
    count    = ds[VAR_NAME].values
    score_CO   = ds["mean_score_CO"].values
    score_CO_1 = ds["score_CO_1_mean"].values
    score_CO_2 = ds["score_CO_2_mean"].values
    score_CO_3 = ds["score_CO_3_mean"].values

    height, width = count.shape
    print('Grille originale : ' + str(height) + ' x ' + str(width))

    # ===================================================
    # IMAGE BINAIRE
    # ===================================================

    if VAR_NAME == 'nb_detect_mean':
        image = (~np.isnan(count)).astype(np.uint8)
    elif VAR_NAME == 'nb_detect_count':
        image = (count > 0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown VAR_NAME: {VAR_NAME}")

    # ============================
    # TRAITEMENT MORPHOLOGIQUE
    # ============================

    image2 = np.concatenate((image, image[:, :1]), axis=1)
    image_extended = np.concatenate((image2, image2, image2), axis=1)

    kernel_map = {
        "diagonal"   : np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=np.uint8),
        "croix"      : cv.getStructuringElement(cv.MORPH_CROSS,   (25, 25)),
        "elliptique" : cv.getStructuringElement(cv.MORPH_ELLIPSE, (ELLIPT_SIZE, ELLIPT_SIZE)),
        "rect_5_5"   : np.ones((5,  5), np.uint8),
        "carre_3_3"  : np.ones((3,  3), np.uint8),
    }
    kernel = kernel_map.get(KERNEL_STR, kernel_map["elliptique"])

    image_closed = cv.morphologyEx(image_extended, cv.MORPH_CLOSE, kernel)

    # =================
    # LABELLISATION
    # =================

    structure_4 = generate_binary_structure(2, 1)
    structure_8 = generate_binary_structure(2, 2)
    structure   = structure_8 if STRUCTURE_STR == 'structure_8' else structure_4

    labels_extended, _ = label(image_closed, structure=structure)

    # /!\ Dans le cas idéal, étendre l'image à gauche et à droite permet de ne pas 
    # "couper" un panache qui traverse 180° en longitude.
    # Mais en pratique, il est possible qu'un panache : 
        # 1) traverse les 180° 
        # 2) ET soient coupé en deux par quelques pixels vides
    # => on suppose que la largeur max de la coupure est d'environ 10 pixels.
    # Cette partie permet donc de fusionner des panaches qui auraient été coupés
    # en deux, au voisinage de 180° lon, par des colonnes / pixels vides.

    # Exemple pour 1 latitude donnée :
    #| bande gauche | bande centrale | bande droite  |
    #0000000000000000000000000001111111111000000000000
    # --> cas "idéal" du panache labelisé 1

    #| bande gauche | bande centrale | bande droite  |
    #0000000000000000000000000001111002222000000000000
    # --> cas particulier traité ici : les deux "bouts" ont été labelisés indépendamment, 
    # à cause des 2 pixels à 0 entre les deux,
    # mais ils sont très proches => on les fusionne.

    # Fusion des labels à cheval sur ±180° lon
    w_ext = width + 1 # +1 = 1 pixel de recouvrement pour comparer les bords de chaque image
    labels_left_check  = labels_extended[:, :w_ext]
    labels_mid_check   = labels_extended[:, w_ext:2*w_ext]
    labels_right_check = labels_extended[:, 2*w_ext:3*w_ext]

    merge_map = {}

    # on regarde de part et d'autre de -180° sur 1 largeur de 10 pixels
    # On suppose que 10 pixels est la largeur maximale au-delà de laquelle un panache
    # peut avoir été "coupé". Au-delà on considère qu'il s'agit de 2 panaches distincts.

    for col in range(min(10, width)): # lon
        for r in range(height): # lat
            lbl_r = labels_mid_check[r, -(col+1)] # dernières colonnes de la bande centrale
            lbl_l = labels_left_check[r, col] # premières colonnes de la bande gauche
            if lbl_r == 0 or lbl_l == 0:
                continue
            if lbl_r != lbl_l:
                a, b = min(lbl_r, lbl_l), max(lbl_r, lbl_l)
                merge_map[b] = a # on garde le plus petit label comme réf. et le plus grand = remplacé par le petit

    # on regarde de part et d'autre de +180° sur 1 largeur de 10 pixels

    for col in range(min(10, width)):
        for r in range(height):
            lbl_m = labels_mid_check[r, col]
            lbl_r = labels_right_check[r, col]
            if lbl_m == 0 or lbl_r == 0:
                continue
            if lbl_m != lbl_r:
                a, b = min(lbl_m, lbl_r), max(lbl_m, lbl_r)
                merge_map[b] = a

    merge_map_resolved = resolve_chain(merge_map)
    print('Panaches autour de ±180° fusionnés : ' + str(len(set(merge_map_resolved.values()))))

    labels_extended_merged = labels_extended.copy()
    for old, new in merge_map_resolved.items():
        labels_extended_merged[labels_extended_merged == old] = new

    # =================================
    # FILTRAGE PAR TAILLE + SCORE CO
    # =================================

    labels_mid_tmp = labels_extended_merged[:, w_ext:2*w_ext][:, :width]
    # nombre de labels / panaches détectés
    unique_lbl_ext, counts_ext = np.unique(labels_mid_tmp, return_counts=True)
    mask_fg        = unique_lbl_ext != 0
    unique_lbl_ext = unique_lbl_ext[mask_fg]
    counts_ext     = counts_ext[mask_fg] # combien de pixel est associé à chaque label / panache

    # => 2 catégories : les panaches et les petits groupes
    large_labels_ext = unique_lbl_ext[counts_ext >= MIN_PLUME_SIZE]
    small_labels     = unique_lbl_ext[counts_ext < MIN_PLUME_SIZE]

    # === DIAGNOSTIC PETITS GROUPES ===
    # (tests pour optimiser les critères de sélection)
    records_small = []
    for lbl in small_labels:
        mask_lbl = (labels_mid_tmp == lbl)
        if not mask_lbl.any():
            continue
        rows, cols = np.where(mask_lbl)
        cols_mod = cols % width

        def safe_mean(arr):
            v = arr[rows, cols_mod]
            v = v[~np.isnan(v)]
            return float(np.mean(v)) if len(v) > 0 else np.nan

        n_px = int(mask_lbl.sum())
        records_small.append({
            "label"     : int(lbl),
            "n_pixels"  : n_px,
            "mean_CO"   : safe_mean(score_CO),
            "mean_CO_1" : safe_mean(score_CO_1),
            "mean_CO_2" : safe_mean(score_CO_2),
            "mean_CO_3" : safe_mean(score_CO_3),
            "max_CO_1"  : float(np.nanmax(score_CO_1[rows, cols_mod])),
            "max_CO_2"  : float(np.nanmax(score_CO_2[rows, cols_mod])),
            "lat_mean"  : float(np.mean(lat_grid[rows])),
            "lon_mean"  : float(np.mean(lon_grid[cols_mod])),
        })

    if records_small:
        df_small = pd.DataFrame(records_small).sort_values("mean_CO_1", ascending=False)
        csv_diag = dir_out + f"diagnostic_small_plumes_{date_str}.csv"
        df_small.to_csv(csv_diag, index=False)
        print(f"Diagnostic sauvegardé : {csv_diag}")
        print(df_small[["label","n_pixels","mean_CO_1","mean_CO_2","mean_CO_3","lat_mean","lon_mean"]].to_string())
    else:
        print("Aucun petit groupe à diagnostiquer")
        df_small = pd.DataFrame()

    # Décision par petit groupe : kept_for_score ou supprimé
    kept_for_score   = set()   # petits groupes à garder comme small groups indépendants
    remove_labels_ext = []     # à supprimer définitivement (score insuffisant)

    for lbl in small_labels:
        mask_lbl = (labels_mid_tmp == lbl)
        if not mask_lbl.any():
            remove_labels_ext.append(lbl)
            continue

        rows, cols = np.where(mask_lbl)
        cols_mod = cols % width

        def safe_mean(arr):
            v = arr[rows, cols_mod]
            v = v[~np.isnan(v)]
            return float(np.mean(v)) if len(v) > 0 else 0.0

        mean_score_CO_1 = safe_mean(score_CO_1)
        mean_score_CO_2 = safe_mean(score_CO_2)
        n_px = int(mask_lbl.sum())

        keep = False
        if n_px == 1:
            if (mean_score_CO_1 >= SCORE_CO_HIGH) or (mean_score_CO_2 >= SCORE_CO_HIGH):
                keep = True
        elif (mean_score_CO_1 >= SCORE_CO_SMALL) and (mean_score_CO_2 >= SCORE_CO_SMALL):
            keep = True

        if keep:
            kept_for_score.add(int(lbl))
            print(f"  Petits groupes label={lbl} récupérés via score "
                  f"(CO_1={mean_score_CO_1:.3f}, CO_2={mean_score_CO_2:.3f}, n={n_px}px)")
        else:
            remove_labels_ext.append(lbl)
            print(f"  Petits groupes label={lbl} supprimés "
                  f"(CO_1={mean_score_CO_1:.3f}, CO_2={mean_score_CO_2:.3f}, n={n_px}px)")

    # Appliquer le filtrage (on garde grands panaches + kept_for_score)
    labels_extended_filtered = labels_extended_merged.copy()
    labels_extended_filtered[np.isin(labels_extended_filtered, remove_labels_ext)] = 0

    # Extraire la portion centrale
    labels_mid = labels_extended_filtered[:, w_ext:2*w_ext][:, :width]

    # Re-numérotation provisoire (grands en 1..N, petits repêchés ensuite) pour pouvoir
    # distinguer facilement les gros panaches des small groups (utile pour l'étape suivante)
    unique_lbl_all = np.unique(labels_mid)
    unique_lbl_all = unique_lbl_all[unique_lbl_all != 0]

    large_kept = [lbl for lbl in unique_lbl_all if lbl not in kept_for_score]
    small_kept  = [lbl for lbl in unique_lbl_all if lbl in kept_for_score]

    labels_provisional = np.zeros_like(labels_mid)
    old_to_new = {}

    # Grands panaches : 1 .. N_plumes
    for new_id, old_id in enumerate(large_kept, start=1):
        labels_provisional[labels_mid == old_id] = new_id
        old_to_new[old_id] = new_id

    # Petits repêchés : SMALL_GROUP_OFFSET .. SMALL_GROUP_OFFSET + M - 1
    for sg_id, old_id in enumerate(small_kept, start=SMALL_GROUP_OFFSET):
        labels_provisional[labels_mid == old_id] = sg_id
        old_to_new[old_id] = sg_id

    num_plumes_init    = len(large_kept)
    num_sg_before_att  = len(small_kept)
    print(f'\n--> Grands panaches : {num_plumes_init}')
    print(f'--> Petits groupes récupérés (avant rattachement) : {num_sg_before_att}')
    print(f'    labels small groups : {SMALL_GROUP_OFFSET} – {SMALL_GROUP_OFFSET + num_sg_before_att - 1}')

    # ======================================================
    # RATTACHEMENT DES PETITS GROUPES AUX GRANDS PANACHES
    # ======================================================
    # Tous les petits groupes retenus dont la
    # distance min au panache le plus proche <= max_dist_px sont absorbés.
    # Ceux trop loin restent comme small groups indépendants.
    # Le groupe rattaché prend le label du grand panache (numérotation 1..N).

    # Masques sur labels_provisional
    plume_labels_ids = np.arange(1, num_plumes_init + 1)
    sg_labels_ids    = np.arange(SMALL_GROUP_OFFSET,
                                  SMALL_GROUP_OFFSET + num_sg_before_att)

    labels_after_att = labels_provisional.copy()

    # Coordonnées de tous les pixels des grands panaches
    mask_large = np.isin(labels_provisional, plume_labels_ids)
    rows_large, cols_large = np.where(mask_large)
    coords_large = np.stack([rows_large, cols_large], axis=1)  # (N_px, 2)

    attached_sgs = set()   # small group labels absorbés
    remaining_sgs = set()  # small group labels conservés comme indépendants

    for sg_lbl in sg_labels_ids:
        mask_sg = (labels_provisional == sg_lbl)
        if not mask_sg.any():
            continue

        rows_s, cols_s = np.where(mask_sg)
        coords_sg = np.stack([rows_s, cols_s], axis=1)

        if len(coords_large) == 0:
            # Aucun grand panache : conserver comme small group indépendant
            remaining_sgs.add(sg_lbl)
            print(f"  SG label={sg_lbl} conservé (aucun grand panache disponible)")
            continue

        dists   = cdist(coords_sg, coords_large)
        min_dist = dists.min()

        if min_dist <= MAX_DIST_PX:
            idx_large     = np.unravel_index(dists.argmin(), dists.shape)[1]
            nearest_label = labels_provisional[rows_large[idx_large], cols_large[idx_large]]
            labels_after_att[mask_sg] = nearest_label
            attached_sgs.add(sg_lbl)
            print(f"  SG label={sg_lbl} ({mask_sg.sum()}px) : rattaché au panache {nearest_label} "
                  f"(dist={min_dist:.1f}px)")
        else:
            remaining_sgs.add(sg_lbl)
            print(f"  SG label={sg_lbl} conservé comme indépendant (dist min={min_dist:.1f}px > {MAX_DIST_PX}px)")

    print(f'\n--> Small groups rattachés : {len(attached_sgs)}')
    print(f'--> Small groups indépendants : {len(remaining_sgs)}')

    # =======================================================
    # RE-NUMÉROTATION FINALE
    # Grands panaches : 1..N (inchangés, mais re-numérotation
    #   consécutive au cas où certains labels ne sont plus présents)
    # Small groups indépendants : SMALL_GROUP_OFFSET..
    # =======================================================

    # Grands panaches (labels < SMALL_GROUP_OFFSET)
    unique_plumes_final = np.unique(labels_after_att)
    unique_plumes_final = unique_plumes_final[
        (unique_plumes_final != 0) & (unique_plumes_final < SMALL_GROUP_OFFSET)
    ]

    # Small groups restants (labels >= SMALL_GROUP_OFFSET)
    unique_sg_final = np.unique(labels_after_att)
    unique_sg_final = unique_sg_final[unique_sg_final >= SMALL_GROUP_OFFSET]

    labels_final = np.zeros_like(labels_after_att)

    # Grands panaches : 1, 2, 3 ...
    for new_id, old_id in enumerate(unique_plumes_final, start=1):
        labels_final[labels_after_att == old_id] = new_id

    # Small groups : SMALL_GROUP_OFFSET, SMALL_GROUP_OFFSET+1 ... (donc 100, 101 ...)
    for sg_id, old_id in enumerate(unique_sg_final, start=SMALL_GROUP_OFFSET):
        labels_final[labels_after_att == old_id] = sg_id

    num_plumes_final = len(unique_plumes_final)
    num_sg_final     = len(unique_sg_final)

    print(f'\n==============================')
    print(f'--> Panaches finaux   : {num_plumes_final}  (labels 1 – {num_plumes_final})')
    print(f'--> Small groups finaux : {num_sg_final}  (labels {SMALL_GROUP_OFFSET} – {SMALL_GROUP_OFFSET + num_sg_final - 1})')
    print(f'==============================')

    # ===================
    # SAUVEGARDE NETCDF
    # ===================

    save_outputs(labels_final, lat_grid, lon_grid, image2, dir_out, date_str)

    # ======================================
    # AJOUTER LES LABELS AU DATASET
    # ======================================
    ds_out = ds.copy()
    ds_out['plume_labels'] = xr.DataArray(
        labels_final.astype(np.int32),
        coords={"latitude": lat_grid, "longitude": lon_grid},
        dims=("latitude", "longitude"),
        attrs={
            "long_name": "Plume label (0=background, 1-99=plumes, >=100=small groups)",
            "units": "1",
            "num_plumes": num_plumes_final,
            "num_small_groups": num_sg_final,
            "date_str": date_str
        }
    )

    return ds_out