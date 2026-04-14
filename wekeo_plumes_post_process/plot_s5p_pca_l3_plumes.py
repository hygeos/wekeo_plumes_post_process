"""
Plotting functions for S5P-PCA L3 plume detection results.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path

from wekeo_plumes_post_process.plumes import map_plumes


def plot_binary_image(image, figures_dir: str | Path | None = None, filename="binary_img_original.png"):
    """
    Plot the original binary detection image.
    
    Parameters
    ----------
    image : ndarray
        Binary image (height x width).
    figures_dir : str | Path | None, optional
        Output directory path. If None, plot is only displayed, not saved.
    filename : str, optional
        Output filename (default: "binary_img_original.png").
    """
    fig = plt.figure(figsize=(16, 8))
    plt.imshow(image, origin="lower")
    plt.title("Original binary image")
    plt.xlabel("Longitude (index)")
    plt.ylabel("Latitude (index)")
    if figures_dir is not None:
        fig.savefig(str(figures_dir) + "/" + filename, bbox_inches="tight", pad_inches=0.1)
    plt.show()


def plot_colored_labels(
    labels_final,
    height,
    width,
    date_str,
    kernel_str,
    structure_str,
    min_plume_size,
    small_group_offset,
    figures_dir: str | Path | None = None
):
    """
    Plot a color-coded image of detected plumes and small groups.
    
    Parameters
    ----------
    labels_final : ndarray
        Final label array (height x width).
    height : int
        Image height.
    width : int
        Image width.
    date_str : str
        Date string (YYYYMMDD).
    kernel_str : str
        Kernel type name.
    structure_str : str
        Structure type name.
    min_plume_size : int
        Minimum plume size threshold.
    small_group_offset : int
        Label offset for small groups.
    figures_dir : str | Path | None, optional
        Output directory path. If None, plot is only displayed, not saved.
    """
    # Count plumes and small groups
    unique_all = np.unique(labels_final)
    unique_all = unique_all[unique_all != 0]
    
    plumes = unique_all[unique_all < small_group_offset]
    small_groups = unique_all[unique_all >= small_group_offset]
    
    num_plumes_final = len(plumes)
    num_sg_final = len(small_groups)
    
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)
    all_labels_ids = list(range(1, num_plumes_final + 1)) + \
                     list(range(small_group_offset, small_group_offset + num_sg_final))
    num_all = len(all_labels_ids)
    cmap_img = plt.cm.get_cmap("tab20", num_all + 1)
    for i, lbl in enumerate(all_labels_ids):
        colored_image[labels_final == lbl] = (np.array(cmap_img(i)[:3]) * 255).astype(np.uint8)

    fig = plt.figure(figsize=(16, 8))
    plt.imshow(colored_image, origin="lower")
    plt.title(f"Detected plumes — {date_str} — kernel={kernel_str} — min_size={min_plume_size}")
    plt.xlabel("Longitude (index)")
    plt.ylabel("Latitude (index)")
    if figures_dir is not None:
        fig.savefig(str(figures_dir) + f"/binary_img_colored_{structure_str}_min_size_{min_plume_size}.png",
                    bbox_inches="tight", pad_inches=0.1)
    plt.show()


def plot_plume_maps(
    labels_final,
    lat_grid,
    lon_grid,
    image,
    date_str,
    var_name,
    ellipt_size,
    min_plume_size,
    small_group_offset,
    figures_dir: str | Path | None = None
):
    """
    Generate all plume maps (plumes, small groups, discarded, combined).
    
    Parameters
    ----------
    labels_final : ndarray
        Final label array (height x width).
    lat_grid : ndarray
        Latitude grid (1D).
    lon_grid : ndarray
        Longitude grid (1D).
    image : ndarray
        Original binary image.
    date_str : str
        Date string (YYYYMMDD).
    var_name : str
        Variable name used for detection.
    ellipt_size : int
        Elliptical kernel size.
    min_plume_size : int
        Minimum plume size.
    small_group_offset : int
        Label offset for small groups.
    figures_dir : str | Path | None, optional
        Output directory path. If None, plots are only displayed, not saved.
    """
    lon_2d, lat_2d = np.meshgrid(lon_grid, lat_grid)
    
    # Count plumes and small groups
    unique_all = np.unique(labels_final)
    unique_all = unique_all[unique_all != 0]
    
    plumes = unique_all[unique_all < small_group_offset]
    small_groups = unique_all[unique_all >= small_group_offset]
    
    num_plumes_final = len(plumes)
    num_sg_final = len(small_groups)
    
    # --- Panaches (labels 1..N) ---
    list_plumes_lat, list_plumes_lon = [], []
    for lbl in range(1, num_plumes_final + 1):
        mask = labels_final == lbl
        list_plumes_lat.append(lat_2d[mask])
        list_plumes_lon.append(lon_2d[mask])

    # --- Small groups (labels >= small_group_offset) ---
    list_sg_lat, list_sg_lon = [], []
    for lbl in range(small_group_offset, small_group_offset + num_sg_final):
        mask = labels_final == lbl
        list_sg_lat.append(lat_2d[mask])
        list_sg_lon.append(lon_2d[mask])

    # --- Discarded : pixels dans image=1 mais absents de labels_final ---
    mask_detected = image.astype(bool)
    mask_kept     = (labels_final > 0)
    mask_discarded = mask_detected & ~mask_kept
    list_disc_lat = [lat_2d[mask_discarded]]
    list_disc_lon = [lon_2d[mask_discarded]]

    # --- Combiné panaches + small groups ---
    list_all_lat = list_plumes_lat + list_sg_lat
    list_all_lon = list_plumes_lon + list_sg_lon

    base_out = str(figures_dir) + f"/plumes_{var_name}_ellipt_{ellipt_size}_min_size_{min_plume_size}" if figures_dir else None

    # ============================
    # CARTE 1 : Panaches seuls
    # ============================
    print(f"Plumes : {len(list_plumes_lat)}")
    map_plumes(list_lon=list_plumes_lon, list_lat=list_plumes_lat,
               zone="GLOBAL", titre="plumes", date=date_str,
               output=base_out)

    # ============================
    # CARTE 2 : Small groups seuls
    # ============================
    print(f"Small groups : {len(list_sg_lat)}")
    if list_sg_lat:
        map_plumes(list_lon=list_sg_lon, list_lat=list_sg_lat,
                   zone="GLOBAL", titre="small_groups", date=date_str,
                   output=base_out)
    else:
        print("  (aucun small group indépendant — carte non générée)")

    # ============================
    # CARTE 3 : Discarded
    # ============================
    print(f"Discarded pixels : {mask_discarded.sum()}")
    if mask_discarded.any():
        map_plumes(list_lon=list_disc_lon, list_lat=list_disc_lat,
                   zone="GLOBAL", titre="discarded", date=date_str,
                   output=base_out)
    else:
        print("  (aucun pixel écarté — carte non générée)")

    # ============================
    # CARTE 4 : Panaches + Small groups
    # ============================
    print(f"Plumes + small groups : {len(list_all_lat)}")
    if list_all_lat:
        map_plumes(list_lon=list_all_lon, list_lat=list_all_lat,
                   zone="GLOBAL", titre="plumes_and_small", date=date_str,
                   output=base_out)


def plot_plume_detection_results(
    ds: xr.Dataset,
    figures_dir: str | Path | None = None,
    var_name: str = "mean_nb_detect",
    kernel_str: str = "elliptique",
    structure_str: str = "structure_8",
    ellipt_size: int = 15,
    min_plume_size: int = 70,
    small_group_offset: int = 100,
    prefix: str = ""
):
    """
    Generate all plots from a dataset with plume detection results.
    
    This convenience function extracts all needed data from the Dataset
    output of apply_plume_detection() and generates all standard plots.
    Plots are displayed in notebook. If figures_dir is provided, they are also saved.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset output from apply_plume_detection() containing 'plume_labels' variable.
    figures_dir : str | Path | None, optional
        Output directory for saving plots. If None, plots are only displayed, not saved.
    var_name : str, optional
        Variable name used for detection (default: "mean_nb_detect").
    kernel_str : str, optional
        Kernel type name (default: "elliptique").
    structure_str : str, optional
        Structure type name (default: "structure_8").
    ellipt_size : int, optional
        Elliptical kernel size (default: 15).
    min_plume_size : int, optional
        Minimum plume size threshold (default: 70).
    small_group_offset : int, optional
        Label offset for small groups (default: 100).
    """
    # Extract arrays from dataset
    labels_final = ds[prefix + 'plume_labels'].values
    lat_grid = ds['latitude'].values
    lon_grid = ds['longitude'].values
    
    # Extract metadata from attributes
    date_str = ds[prefix + 'plume_labels'].attrs.get('date_str', 'unknown')
    
    # Override parameters from attributes if available
    if 'num_plumes' in ds[prefix + 'plume_labels'].attrs:
        # Attributes are available, use them for verification
        pass
    
    # Create binary image from variable if available
    if var_name in ds:
        count = ds[var_name].values
        if var_name == 'mean_nb_detect':
            image = (~np.isnan(count)).astype(np.uint8)
        elif var_name == 'nb_detect_count':
            image = (count > 0).astype(np.uint8)
        else:
            # Fallback: create from labels
            image = (labels_final > 0).astype(np.uint8)
    else:
        # No source variable, create from labels
        image = (labels_final > 0).astype(np.uint8)
    
    height, width = labels_final.shape
    
    # Generate all plots
    plot_binary_image(image, figures_dir)
    
    plot_colored_labels(
        labels_final, height, width, date_str,
        kernel_str, structure_str, min_plume_size,
        small_group_offset, figures_dir
    )
    
    plot_plume_maps(
        labels_final, lat_grid, lon_grid, image,
        date_str, var_name, ellipt_size, min_plume_size,
        small_group_offset, figures_dir
    )
