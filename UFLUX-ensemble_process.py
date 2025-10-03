# -*- coding: utf-8 -*-
"""
UFLUX-emsemble products processing and plotting program for journal publication
Author: Songyan Zhu
Email: Songyan.Zhu@soton.ac.uk
Note: code is gegeranted by SZ but comments are producced with help of GenAI
"""

# --- Standard Library Imports ---
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm # Progress bar for loops

# --- Scientific and Plotting Imports ---
import xarray as xr # Used for map/grid data (Figures 4, 5, 7, 8)
import geopandas as gpd # Used for plotting geographical boundaries (world maps)
from scipy import stats # Used for linear regression (linregress)
from scipy.stats import gaussian_kde # Used for Kernel Density Estimation (KDE) scatter plots
from scipy.optimize import curve_fit # Used for non-linear curve fitting (Figure 10/11)

# --- Matplotlib and Custom Plotting Imports ---
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec # Used for fine control over subplot layout (GridSpec)
from matplotlib.lines import Line2D # Used to create custom legend handles

# --- Custom/Third-Party Library Imports  ---
# Assuming these custom modules and their functions are correctly installed/accessible
from scitbx.sciplt import setup_canvas, upper_legend, unify_xylim # Canvas setup and legend utilities
from scitbx.sciplt import * # Imports remaining scitbx.sciplt utilities (e.g., roundit, add_text, nature_colors)
from sciml.regress2 import regress2 # Used for Reduced Major Axis (RMA) regression

# --- Project Setup ---
# Define the root directory for accessing input and output files
project_directory = Path('<YOUR PROJECT DIRECTORY>')

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# --- Define functions ---

# Define the Kernel Density Estimate (KDE) plot function (used in Figures 2, 7, 9)
def kde_scatter(ax, dfp, x_name, y_name, frac = 0.1):
    """Generates a scatter plot colored by data density using Gaussian KDE."""
    # Sample a fraction of the data for faster plotting
    dfp = dfp[[x_name, y_name]].dropna().sample(frac = frac)
    x = dfp[x_name]
    y = dfp[y_name]

    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last (on top)
    idx = z.argsort()
    # Apply sorting to the sampled data (must use .iloc or convert back to numpy/series)
    x, y, z = x.iloc[idx], y.iloc[idx], z[idx]

    ax.scatter(x, y, c=z, s=50, cmap = 'RdYlBu_r')

    # The 1:1 line plotting is commented out in the original code, but I'll leave the call here.
    # xl = np.linspace(np.floor(x.min()), np.ceil(x.max()), 100)
    # ax.plot(xl, xl, ls = '-.', color = 'k')

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

## Plot Figure 2: EC vs. UFLUX Time Series Scatter Plots

# Dictionary to store validation time series data (EC vs. UFLUX) for each flux
# Data structure: 5 fluxes, each a time series of EC vs. UFLUX at validation sites.
ddp = {}
for flux in ['GPP', 'Reco', 'NEE', 'H', 'LE']:
    # Load validation time series data for the current flux from a parquet file
    ddp[flux] = pd.read_parquet(project_directory.joinpath(f'4_analysis/UFLUX_val_ts_{flux}.parquet'))

# Setup the main figure and GridSpec for a 5-panel layout (3 top, 2 bottom)
fig = plt.figure(figsize = (12, 8))
gs = gridspec.GridSpec(8, 12, figure = fig)
# Adjust spacing between subplots (using the original values)
gs.update(wspace = 36, hspace = 2)
# Define subplot locations
ax1 = plt.subplot(gs[0:4, :4])      # GPP (Top-Left)
ax2 = plt.subplot(gs[0:4, 4:8])    # Reco (Top-Middle)
ax3 = plt.subplot(gs[0:4, 8:])     # NEE (Top-Right)
ax4 = plt.subplot(gs[4:, 2:6])     # H (Bottom-Left-Center)
ax5 = plt.subplot(gs[4:, 6:10])    # LE (Bottom-Right-Center)

axes = [ax1, ax2, ax3, ax4, ax5]

# Plot each flux comparison (GPP, Reco, NEE, H, LE)
for cnt, flux in tqdm(enumerate(['GPP', 'Reco', 'NEE', 'H', 'LE'])):
    dfp = ddp[flux]
    ax = axes[cnt]
    x = dfp['EC']    # EC Tower measurement (Reference/Truth)
    y = dfp['UFLUX'] # UFLUX prediction

    # Calculate Ordinary Least Squares (OLS) linear correlation statistics
    slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)

    # Plot the scatter plot using Kernel Density Estimate (KDE) for coloring
    # This visualizes data density, with higher density points colored similarly.
    kde_scatter(ax, dfp, 'EC', 'UFLUX')

    # Unify x and y axis limits based on the data range to create square plots
    vmin, vmax = unify_xylim(ax)
    # For unidirectional fluxes (GPP, Reco, H, LE), ensure the minimum limit is 0
    if flux in ['GPP', 'Reco', 'H', 'LE']: vmin = 0

    # Plot the 1:1 line (line of perfect agreement)
    ax.plot(np.linspace(vmin, vmax, 100), np.linspace(vmin, vmax, 100), ls = '-.', color = 'k')

    # Add text box with subplot label, flux name, and regression statistics
    ax.text(
        0.05, 0.75,
        # Format: (a), Flux Name, Regression Equation, R-squared value
        f'({chr(97 + cnt)}) \n{flux} (UFLUX)\ny={roundit(slope, 2)}x+{roundit(intercept, 2)}\n$R^2:$ {roundit(rvalue**2)}',
        transform = ax.transAxes, # Coordinates relative to the axes (0 to 1)
        bbox=dict(
            facecolor='none',  # background color (transparent)
            edgecolor='none',  # no border
            boxstyle='round,pad=0.3', # rounded corners
            alpha=1
        )
    )
    # Apply unified limits and labels
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel('EC')
    ax.set_ylabel('UFLUX')


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

## Plot Figure 3: Ensemble Performance Heatmap (R² and MRAE)

# Load UFLUX model validation metrics (R², RMSE, MRAE, etc.) across all ensembles
dfm = pd.read_csv(project_directory.joinpath("2_validation_metrics/UFLUX-ensemble_validation.csv"))

# Define the explicit display order for the ensemble products (Y-axis)
product_order = [
    'GOME-2-SIF-ERA5', 'GOSAT-755-SIF-ERA5', 'GOSAT-772-SIF-ERA5', 'OCO-2-CSIF-ERA5', # SIF products
    'AVHRR-EVI2-ERA5', 'AVHRR-NDVI-ERA5', 'AVHRR-NIRv-ERA5', # AVHRR products
    'MODIS-NIRv-CFSV2', 'MODIS-NIRv-ERA5-NT', # Alternative climate/no-trend inputs
    'MODIS-EVI2-ERA5', 'MODIS-NDVI-ERA5', 'MODIS-NIRv-ERA5', # MODIS products
]

# Define the explicit display order for the flux types (X-axis)
flux_order = ['GPP', 'Reco', 'NEE', 'H', 'LE']

# Pivot the metrics data from long to wide format for R² and MRAE
# The index is the ensemble 'LABEL', and columns are the 'FLUX' types.
dfm_r2 = dfm.pivot(index = 'LABEL', columns = 'FLUX', values = 'r2').dropna().loc[product_order, flux_order]
dfm_mrae = dfm.pivot(index = 'LABEL', columns = 'FLUX', values = 'MRAE').dropna().loc[product_order, flux_order]

# Setup grid coordinates for plotting scatter points
xs = np.arange(dfm_r2.shape[1]) # X-coordinates (Fluxes)
ys = np.arange(dfm_r2.shape[0]) # Y-coordinates (Products)
xs_, ys_ = np.meshgrid(xs, ys) # Create a 2D grid of coordinates

# Prepare a flat DataFrame of plotting points with their R² and MRAE values
dfp = []
for x, y in zip(xs_.ravel(), ys_.ravel()):
    # [X-coordinate, Y-coordinate, R2 value, MRAE value]
    dfp.append([x, y, dfm_r2.iloc[y, x], dfm_mrae.iloc[y, x]])
dfp = pd.DataFrame(dfp, columns = ['X', 'Y', 'R2', 'MRAE'])

# Setup figure and axis
fig, ax = setup_canvas(1, 1, figsize = (4, 10))
im = ax.scatter(
    dfp['X'], dfp['Y'],
    c = dfp['R2'], # Color of the dot is mapped to R² (performance)
    # Size of the dot is mapped to MRAE (error). Scaled by 1000 for visibility.
    s = dfp['MRAE'] * 1000,
    edgecolor = 'k',
    # Use a sequential, colorblind-friendly colormap for R²
    cmap = 'Blues'
)

# Adjust ticks and labels
ax.set_xticks(xs)
ax.set_xticklabels(dfm_r2.columns) # Flux names as X-axis labels

ax.set_yticks(ys)
ax.set_yticklabels(dfm_r2.index) # Ensemble names as Y-axis labels

# Set limits with padding
ax.set_xlim(xs[0] - 1, xs[-1] + 1)
ax.set_ylim(ys[0] - 1, ys[-1] + 1)

# Add color bar for R²
pos = ax.get_position()
# Create space for the color bar to the right of the main plot
cbar_ax = fig.add_axes([pos.x1, pos.y0, 0.05, pos.y1 - pos.y0])
cbar = fig.colorbar(im, cax=cbar_ax, orientation = 'vertical', label = r'R$^2$') # Using LaTeX for R^2

# Plot R² | MRAE values as text above each dot
for x, y in zip(xs_.ravel(), ys_.ravel()):
    r2 = roundit(dfm_r2.iloc[y, x], 1)
    mrae = roundit(dfm_mrae.iloc[y, x], 1)
    # Text format: R2|MRAE
    text = f'{r2}|{mrae}'
    # Position text slightly above and to the left of the dot
    ax.text(x - 0.47, y + 0.45, text, fontsize = 10)

# Create 'phantom' scatter plots just to generate the legend handles for dot size (MRAE quantiles)
for qt_ in [0.1, 0.25, 0.5, 0.75, 0.99]:
    # Size is determined by the MRAE quantile value, scaled by 1000
    ax.scatter([], [],
        color = 'gray',
        edgecolor = 'k',
        alpha = 0.3,
        s = (dfp['MRAE'].quantile(qt_) - dfp['MRAE'].min()) / (dfp['MRAE'].max() - dfp['MRAE'].min()) * 1000,
        label = f"{roundit(dfp['MRAE'].quantile(qt_), 1)}"
    )

# Set up legend for MRAE dot sizes
upper_legend(ax, 0.5, 1.08)
# Add a static text label for the MRAE legend
add_text(ax, -0.18, 1.054, 'MRAE:', fontsize = 10)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

## Plot Figure 4: Global Multi-Year Average Flux Maps

# Load UFLUX-ensemble multiyear average maps for GPP, Reco, NEE, H, and LE
ncp = xr.open_dataset(project_directory.joinpath('4_analysis/UFLUX_five_fluxes_maps.nc'))

# Load the world shape file for plotting land borders
world = gpd.read_file(project_directory.joinpath('4_analysis/world.geojson'))

# Setup canvas for 5 rows, 1 column (vertical arrangement of maps)
fig, axes = setup_canvas(5, 1, figsize = (8, 15))

# Plot each flux map using xarray's plotting capabilities
ax = axes[0]
# GPP map
ncp['GPP'].plot(ax = ax, cmap = 'YlGn', vmin = 0, vmax = 10, cbar_kwargs={'shrink': 0.8, 'label': r'GPP ($gC \ m^{-2} \ d^{-1}$)'})
world.plot(ax = ax, color = 'none', edgecolor = 'k') # Overlay world borders

ax = axes[1]
# Reco map
ncp['Reco'].plot(ax = ax, cmap = 'YlOrBr', vmin = 0, vmax = 10, cbar_kwargs={'shrink': 0.8, 'label': r'Reco ($gC \ m^{-2} \ d^{-1}$)'})
world.plot(ax = ax, color = 'none', edgecolor = 'k')

ax = axes[2]
# NEE map (diverging colormap)
ncp['NEE'].plot(ax = ax, cmap = 'PRGn_r', vmin = -5, vmax = 5, cbar_kwargs={'shrink': 0.8, 'label': r'NEE ($gC \ m^{-2} \ d^{-1}$)'})
world.plot(ax = ax, color = 'none', edgecolor = 'k')

ax = axes[3]
# H map (Sensible Heat)
ncp['H'].plot(ax = ax, cmap = 'hot_r', vmin = 0, vmax = 100, cbar_kwargs={'shrink': 0.8, 'label': r'H ($W \ m^{-2}$)'})
world.plot(ax = ax, color = 'none', edgecolor = 'k')

ax = axes[4]
# LE map (Latent Heat)
ncp['LE'].plot(ax = ax, cmap = 'Blues', vmin = 0, vmax = 100, cbar_kwargs={'shrink': 0.8, 'label': r'LE ($W \ m^{-2}$)'})
world.plot(ax = ax, color = 'none', edgecolor = 'k')

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

## Plot Figure 5: Latitudinal Distribution of Fluxes (Ensembles vs. EC)

# Load EC tower latitudinal distribution data (aggregated)
df_ec = pd.read_csv(project_directory.joinpath('4_analysis/EC-latitudinal.csv'), index_col = 0)
# Load UFLUX ensemble latitudinal distribution data (multiple products)
ncp = xr.open_dataset(project_directory.joinpath('4_analysis/UFLUX-latitudinal.nc'))

# Create canvas for 5 subplots
fig = plt.figure(figsize = (12, 8))
gs = gridspec.GridSpec(8, 12, figure = fig)
gs.update(wspace = 36, hspace = 2) # Original spacing

ax1 = plt.subplot(gs[0:4, :4])
ax2 = plt.subplot(gs[0:4, 4:8])
ax3 = plt.subplot(gs[0:4, 8:])
ax4 = plt.subplot(gs[4:, 2:6])
ax5 = plt.subplot(gs[4:, 6:10])

axes = [ax1, ax2, ax3, ax4, ax5]

# Define units for plot labels
flux_units = {
    'NEE': r'NEE $(gC \ m^{-2} \ d^{-1})$',
    'GPP': r'GPP $(gC \ m^{-2} \ d^{-1})$',
    'Reco': r'Reco $(gC \ m^{-2} \ d^{-1})$',
    'H': r'H $(MJ \ m^{-2} \ d^{-1})$', # Energy units
    'LE': r'LE $(MJ \ m^{-2} \ d^{-1})$' # Energy units
}

# Colorblind-friendly palette (12 colors)
colorblind_palette_12 = [
    '#E69F00',  # orange
    '#117733',  # greenish teal
    '#009E73',  # green
    '#DDCC77',  # coral
    '#0072B2',  # blue
    '#D55E00',  # dark orange
    '#CC79A7',  # pink
    '#999999',  # grey
    '#999933',  # olive
    '#44AA99',  # teal
    '#882255',  # wine
    '#AA4499',  # purple
]


# Iterate across fluxes and plot latitudinal profiles in each subplot
for cnt, flux in enumerate(['GPP', 'Reco', 'NEE', 'H', 'LE']):
    ax = axes[cnt]
    
    # Convert xarray data to a wide DataFrame where columns are products ('prod') and index is 'latitude'
    dfp = ncp[flux].to_dataframe().reset_index().pivot(index = 'latitude', columns = 'prod')
    dfp.columns = dfp.columns.droplevel(0) # Drop the flux name level from columns

    # Plot each UFLUX product's latitudinal profile
    for i, c in enumerate(dfp.columns):
        color = colorblind_palette_12[i % len(colorblind_palette_12)]
        # Use dashed line for specific products to differentiate them
        linestyle = '--' if c in ['MODIS-NIRv-ERA5-NT', 'OCO-2-CSIF-ERA5'] else '-'
        ax.plot(dfp[c], dfp.index, label = c, color = color, ls = linestyle)

    # Plot the EC tower aggregation data for comparison (black dashed line with circles)
    ax.plot(df_ec[flux], df_ec.index, '--o', color = 'k', markersize = 5, markerfacecolor = 'w')
    
    # Add subplot label
    ax.text(
        0.05, 0.05,
        f'({chr(97 + cnt)})', transform = ax.transAxes,
        bbox=dict(
            facecolor='white',  # background color
            edgecolor='none',   # no border
            boxstyle='round,pad=0.3', # rounded corners
            alpha=1
        )
    )
    ax.set_ylabel('Latitude')
    ax.set_xlabel(flux_units[flux])

# Add legend for UFLUX products (positioned above the top row)
upper_legend(axes[1], 0.5, 1.3, ncols = 4)

# ------------------------------------------------------------------------------

## Add a new legend (for EC Tower data)

# Create the custom legend handle for the EC tower measurement line
ec_legend = Line2D(
    [0], [0], linestyle='--', marker='o', color='k',
    markerfacecolor='w', markersize=5, label='EC Tower'
)

# Add this legend to one axis (axes[2]) for clarity, positioned outside the plot area
axes[2].legend(handles=[ec_legend], loc='upper right', bbox_to_anchor=(1.1, 1.15), frameon=False, title='')

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

## Plot Figure 6: Global Interannual Variability (IAV) of GPP

# Load Interannual Variability (IAV) data for GPP products (global aggregates)
df_iav = pd.read_csv(project_directory.joinpath('4_analysis/GPP_IAV_products.csv'), index_col = 0)

# Convert the index to datetime objects
df_iav.index = pd.to_datetime(df_iav.index, format = '%Y-%m-%d')


# Colorblind-friendly palette (Paul Tol 7-color palette)
colors_blindfriendly = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']

# Setup canvas for the time series plot
fig, ax = setup_canvas(1, 1, fontsize = 12, labelsize = 12)

# Plot IAV for UFLUX(NT) (No-Trend model)
df_iav['UFLUX(NT)'].plot(ax = ax, style = '-o', markersize = 10, label = 'UFLUX(NT)', color = colors_blindfriendly[1])
# Plot uncertainty band (Standard Deviation)
ax.fill_between(df_iav.index, df_iav['UFLUX(NT)'] - df_iav['UFLUX(NT)_std'], df_iav['UFLUX(NT)'] + df_iav['UFLUX(NT)_std'], color = colors_blindfriendly[1], alpha = 0.3, zorder = 1)

# Plot IAV for UFLUX(WT) (With-Trend model)
df_iav['UFLUX(WT)'].plot(ax = ax, style = '-o', markersize = 10, label = 'UFLUX(WT)', color = colors_blindfriendly[2])
# Plot uncertainty band
ax.fill_between(df_iav.index, df_iav['UFLUX(WT)'] - df_iav['UFLUX(WT)_std'], df_iav['UFLUX(WT)'] + df_iav['UFLUX(WT)_std'], color = colors_blindfriendly[2], alpha = 0.3, zorder = 1)

# Plot IAV for Upscale products
df_iav['Upscale'].plot(ax = ax, style = '-o', markersize = 10, label = 'Upscale', color = colors_blindfriendly[4])
# Plot uncertainty band
ax.fill_between(df_iav.index, df_iav['Upscale'] - df_iav['Upscale_std'], df_iav['Upscale'] + df_iav['Upscale_std'], color = colors_blindfriendly[4], alpha = 0.3, zorder = 1)

# Plot IAV for Trendy model ensemble
df_iav['Trendy'].plot(ax = ax, style = '-o', markersize = 10, label = 'Trendy', color = colors_blindfriendly[6])
# Plot uncertainty band
ax.fill_between(df_iav.index, df_iav['Trendy'] - df_iav['Trendy_std'], df_iav['Trendy'] + df_iav['Trendy_std'], color = colors_blindfriendly[6], alpha = 0.1, zorder = 2)

# Plot EC tower data on a secondary Y-axis (since it's a local/non-global measure)
ax2 = ax.twinx()
df_iav['EC'].plot(ax = ax2, style = '--^', markersize = 10, label = 'EC', color = 'k')
ax2.set_ylabel(r'EC (gC $m^{-2}$ $d^{-1}$)') # Local units

# Adjust x-axis limits with padding
ax.set_xlim(df_iav['UFLUX(NT)'].index[0] - pd.DateOffset(years = 2), df_iav['UFLUX(NT)'].index[-1] + pd.DateOffset(years = 1))

# Add legends for both axes
upper_legend(ax, 0.5, 1.2, ncols = 4)
upper_legend(ax2, 0.5, 1.1, ncols = 4)
ax.set_xlabel('') # Clear x-label since date is implicit on x-axis
ax.set_ylabel(r'Product ($Pg \ C$)') # Global units

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

## Plot Figure 7: High-Resolution (Sentinel-2) vs. Coarse (MODIS) GPP Comparison

# Load MODIS and Sentinel-2 UFLUX GPP maps for Europe
ncp = xr.open_dataset(project_directory.joinpath('4_analysis/MODIS_Sentinel-2_Europe.nc'))
# Load the corresponding data as a spreadsheet for the scatter plot
dfp = pd.read_parquet(project_directory.joinpath('4_analysis/MODIS_Sentinel-2_Europe.parquet'))

# Load world shape file
world = gpd.read_file(project_directory.joinpath('4_analysis/world.geojson'))

# Setup canvas for 2 map subplots side-by-side
fig, axes = setup_canvas(1, 2, figsize = (12, 6), sharex = False, sharey = False, wspace = 0.05)

# --- Map Plotting (Left: MODIS; Right: Sentinel-2) ---
ncp['MODIS'].plot(ax = axes[0], label = 'MODIS', cmap = 'YlGn', vmin = 0, vmax = 10, cbar_kwargs = {'shrink': 0.6, 'label': None})
ncp['Sentinel-2'].plot(ax = axes[1], label = 'Sentinel-2', cmap = 'YlGn', vmin = 0, vmax = 10, cbar_kwargs = {'shrink': 0.6, 'label': None})
world.plot(ax = axes[0], color = 'none', edgecolor = 'k')
world.plot(ax = axes[1], color = 'none', edgecolor = 'k')

# Add source labels to the maps
for ax, label_ in [[axes[0], 'MODIS'], [axes[1], 'Sentinel-2']]:
    ax.text(
        0.99, 0.03, label_,
        horizontalalignment='right',
        transform = ax.transAxes,
        bbox=dict(
            facecolor='none',  # transparent
            edgecolor='none',  # no border
            boxstyle='round,pad=0.3',
            alpha=1
        )
    )

# --- Scatter Plot Creation ---
# Add a new axis on the right of the maps for the scatter plot comparison
pos = axes[1].get_position()
# Coordinates for the new axis: [left, bottom, width, height]
ax3 = fig.add_axes([pos.x1 + 0.1, pos.y0, pos.x1 - pos.x0, pos.y1 - pos.y0])
ax3.tick_params(direction = "in", which = "both")

# Data for scatter plot
x = dfp['MODIS']; y = dfp['Sentinel-2']
# Linear regression stats
slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)

# Plot KDE scatter
kde_scatter(ax3, dfp, 'MODIS', 'Sentinel-2')
vmin, vmax = unify_xylim(ax3)

# Plot the 1:1 line
ax3.plot(np.linspace(vmin, vmax, 100), np.linspace(vmin, vmax, 100), ls = '-.', color = 'k')

# Add regression stats text
ax3.text(
    0.05, 0.8,
    f'y={roundit(slope, 2)}x+{roundit(intercept, 2)}\n$R^2:$ {roundit(rvalue**2)}',
    transform = ax3.transAxes,
    bbox=dict(
        facecolor='none',
        edgecolor='none',
        boxstyle='round,pad=0.3',
        alpha=1
    )
)
# Set limits and labels
ax3.set_xlim(0, 14)
ax3.set_ylim(0, 14)
ax3.set_xlabel('MODIS')
ax3.set_ylabel('Sentinel-2')

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

## Plot Figure 8: Difference Map (FLUXCOM-X vs. UFLUX)

# Load FLUXCOM-X and UFLUX data for GPP, NEE, and ET (as LE)
# Note: The original code uses LE and implies a conversion to ET later (ET (mm/hr) ≈ LE (W/m2) × 0.00147)
ncp = xr.open_dataset(project_directory.joinpath('4_analysis/FLUXCOMX+UFLUX.nc'))


# Select flux name to plot
name = 'GPP' # Flux name (GPP, NEE, or ET)

# Calculate the difference map: FLUXCOM-X minus UFLUX
ncd = ncp.sel(source = 'FLUXCOMX').drop_vars('source') - ncp.sel(source = 'UFLUX').drop_vars('source')

# Calculate the spatial correlation between the two products
# Convert xarray data to DataFrame, pivot by source, and drop NaNs
dft = ncp[name].to_dataframe().reset_index().pivot(columns = ['source'], index = ['latitude', 'longitude'], values = [name]).dropna()
dft.columns = dft.columns.droplevel(0) # Remove the flux name level
# Linear regression
slope, intercept, rvalue, pvalue, stderr = stats.linregress(dft['UFLUX'], dft['FLUXCOMX'])

# --- Difference Map Plotting ---
fig, ax = setup_canvas(1, 1, figsize = (6, 4), fontsize = 10, labelsize = 10)
cbar_kwargs = {'shrink':0.5, 'label': f'$\Delta${name}' + r' ($gC \ m^{-2} \ d^{-1}$)'} # Delta symbol for difference

# Plot the difference map (PuOr_r is a diverging colormap)
im = ncd[name].drop_vars('spatial_ref', errors = 'ignore').plot(
    ax = ax, cbar_kwargs = cbar_kwargs, cmap = 'PuOr_r',
    vmin = -5, vmax = 5
)

# Add the correlation stats text in the bottom right
ax.text(
    0.95, 0.01,
    f'{name}; $R^2:$ {roundit(rvalue**2)}; y={roundit(slope, 2)}x+{roundit(intercept, 2)}',
    transform = ax.transAxes, fontsize = 9,
    horizontalalignment = 'right',
    verticalalignment = 'bottom',
    bbox=dict(
        facecolor='none',  # transparent
        edgecolor='none',  # no border
        boxstyle='round,pad=0.3',
        alpha=0.9
    )
)

# Plot map borders and set labels
ax.set_xlabel(''); ax.set_ylabel('')
world.plot(ax = ax, facecolor = 'none', edgecolor = 'k', linewidth = 0.5)

# --- Add Marginal Plots (Longitudinal and Latitudinal Distributions of the Difference) ---
x0 = ax.get_position().x0
x1 = ax.get_position().x1
y0 = ax.get_position().y0
y1 = ax.get_position().y1

# Add Longitudinal distribution of the difference (below the map)
ax1 = fig.add_axes([x0, y0 - 0.2, x1 - x0, 0.15])
ax1.set_ylabel(name)
ax1.set_xlabel('longitude')
ax1.yaxis.set_label_position("right")
ax1.yaxis.tick_right()
# Plot the mean difference across all latitudes
ax1.plot(ncd[name].longitude, ncd[name].mean(dim = ['latitude']), ls = '--', color = 'k')

# Add Latitudinal distribution of the difference (left of the map)
ax2 = fig.add_axes([x0 - 0.15, y0, 0.1, y1 - y0])
# Plot the mean difference across all longitudes
ax2.plot(ncd[name].mean(dim = ['longitude']), ncd[name].latitude, ls = '--', color = 'k')
ax2.set_xlabel(name)
ax2.set_ylabel('latitude')

# Set consistent limits
ax.set_xlim(-180, 180)
ax1.set_xlim(-180, 180)

ax.set_ylim(-60, 80)
ax2.set_ylim(-60, 80)

# Set limits for the difference plots
ax1.set_ylim(-5, 5)
ax2.set_xlim(-5, 5)

# Add subplot label
ax.text(0.05, 0.1, '(a)', fontsize = 14, transform = ax.transAxes, horizontalalignment = 'left', verticalalignment = 'center')

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

## Plot Figure 9: Closure Metrics (WUE and EBR)

# Load data for UFLUX and EC Water-Use Efficiency (WUE) and Energy Balance Ratio (EBR) at validation sites
wue = pd.read_parquet(project_directory.joinpath('4_analysis/UFLUX_EC_WUE.parquet'))
ebr = pd.read_parquet(project_directory.joinpath('4_analysis/UFLUX_EC_EBR.parquet'))

# Setup canvas for 2 subplots side-by-side
fig, axes = setup_canvas(1, 2, figsize = (10, 4.5), sharex = False, sharey = False, wspace = 0.2, labelsize = 10, fontsize = 10)

# --- Plotting Water Use Efficiency (WUE) ---

x = wue['WUE_EC']       # EC (Truth)
y = wue['WUE_UFLUX']    # UFLUX (Prediction)

ax = axes[0]
kde_scatter(ax, wue, 'WUE_EC', 'WUE_UFLUX', frac = 1) # KDE scatter plot using all data

# Perform Reduced Major Axis (RMA) regression (regress2 is a custom function)
# Note: _need_intercept=False forces the line through the origin, common for ratio comparisons
res = regress2(x.values, y.values, _method_type_2="reduced major axis", _need_intercept=False)

slope = np.round(res['slope'], 2)
intercept = np.round(res['intercept'], 2)
r2 = np.round(res['r'][0] ** 2, 2)

# Set limits and 1:1 line
ax.set_xlim(0, 150)
ax.set_ylim(0, 150)
ax.plot(np.linspace(0, 150, 100), np.linspace(0, 150, 100), ls = '-.', color = 'k') # 1:1 line

# Add regression stats text
ax.text(
    0.05, 0.8,
    f'(a)\ny = {slope}x + {intercept}' + '\n$R^2$: ' + f'{r2}',
    transform = ax.transAxes,
    fontsize = 10,
    bbox=dict(
        facecolor='none',
        edgecolor='none',
        boxstyle='round,pad=0.3',
        alpha=1
    )
)

ax.set_xlabel(r'WUE EC $(gC \ (kg \ H20)^{-1})$')
ax.set_ylabel(r'WUE UFLUX $(gC \ (kg \ H20)^{-1})$')

# --- Plotting Energy Balance Ratio (EBR) ---

ax =axes[1]
x = ebr['EBR_EC']       # EC (Truth)
y = ebr['EBR_UFLUX']    # UFLUX (Prediction)

# Perform RMA regression
res = regress2(x.values, y.values, _method_type_2="reduced major axis", _need_intercept=False)

slope = np.round(res['slope'], 2)
intercept = np.round(res['intercept'], 2)
r2 = np.round(res['r'][0] ** 2, 2)

# Set limits and 1:1 line
ax.set_xlim(0, 2.5)
ax.set_ylim(0, 2.5)
ax.plot(np.linspace(0, 2.5, 100), np.linspace(0, 2.5, 100), ls = '-.', color = 'k') # 1:1 line

# Add regression stats text
ax.text(
    0.05, 0.8,
    f'(b)\ny = {slope}x + {intercept}' + '\n$R^2$: ' + f'{r2}',
    transform = ax.transAxes,
    fontsize = 10,
    bbox=dict(
        facecolor='none',
        edgecolor='none',
        boxstyle='round,pad=0.3',
        alpha=1
    )
)

kde_scatter(ax, ebr, 'EBR_EC', 'EBR_UFLUX', frac = 1)
ax.set_xlabel(r'EC ([H + LE] / [R_{NET} - G])') # Using R_NET - G for net available energy
ax.set_ylabel(r'UFLUX ([H + LE] / [R_{NET} - G])')

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

## Plot Figure 10 & 11: Flux-to-Climate Response Curves

# --- Helper Functions (Custom Curve Fitting) ---

# Define function for polynomial and exponential fitting (used for flux-climate response curves)
def plot_curve_fitting(func_name, ax, x, y, precision = 2, with_text = False, lineshape = '-'):
    """Performs non-linear curve fitting and plots the result, along with formatting the equation text."""
    if func_name not in ['lin', 'exp', 'poly2']: raise Exception('func_name must be `lin`, `exp`, or `poly2`!')
    
    # Define fitting functions
    def func_lin(x, a, b):
        return a * x + b
    def func_poly2(x, a, b, c):
        return a * x**2 + b * x + c
    def func_exp(x, a, b, c):
        return a * np.exp(-b * x) + c

    func_dict = {
        'lin': func_lin, 'poly2': func_poly2, 'exp': func_exp
    }
    
    # Clean data (remove NaNs)
    x = x.copy(); y = y.copy()
    idx = (~x.isna()) & (~y.isna())
    x = x[idx]; y = y[idx]

    func = func_dict[func_name]
    # Perform curve fitting (optimization)
    popt, pcov = curve_fit(func, x, y)
    
    # --- Custom Functions for Fit Metrics (assumed imported/defined) ---
    # Print fitting p-values and r2
    fit_p = get_curve_fit_p_value(func, popt, x, y)
    fit_r2 = get_curve_fit_r2(func, popt, x, y)
    # +++++++++++++++++++++++++++++++++++++++++++++++
    
    # Plot the fitted curve
    x_plot = np.linspace(x.min(), x.max(), 100) # Use smoother x-values for the line
    ax.plot(x_plot, func(x_plot, *popt), lineshape, color = 'k', label = 'Fitted')
    
    # Format equation text using LaTeX and f-strings
    if func_name == 'lin':
        a, b = popt
        a = roundit(a, precision); b = roundit(b, precision)
        sign = '+' if b >= 0 else '-'
        text = fr'y={a}x {sign} {np.abs(b)}'
    elif func_name == 'exp':
        a, b, c = popt
        a = roundit(a, precision); b = roundit(b, precision); c = roundit(c, precision)
        sign = '+' if c >= 0 else '-'
        text = fr'$y = {a} \times e^{{-{b}x}} {sign} {np.abs(c)}$'
    elif func_name == 'poly2':
        a, b, c = popt
        a = roundit(a, precision); b = roundit(b, precision); c = roundit(c, precision)
        sign1 = '+' if b >= 0 else '-'
        sign2 = '+' if c >= 0 else '-'
        text = f'y={a}x^2 {sign1} {np.abs(b)}x {sign2} {np.abs(c)}'
    else:
        raise Exception('func_name must be `lin`, `exp`, or `poly2`!')

    if with_text: add_text(ax, 0.05, 0.05, text, horizontalalignment = 'left')

# Define function calculating site years
def get_n_sites_years(dfa, igbp_code):
    """Calculates the number of unique sites (ID) and total site-years for a given IGBP code."""
    dft = dfa.copy().loc[dfa['IGBP'] == igbp_code, 'ID']

    df_site_year = []
    for g, gp in dft.reset_index().groupby('ID'):
        gp = gp.set_index('datetime')
        # Count unique years
        nyear = len(gp.index.year.drop_duplicates())
        df_site_year.append(nyear)
    df_site_year = np.array(df_site_year)
    n_sites = len(df_site_year)
    n_years = np.sum(df_site_year)
    return n_sites, n_years

# --- Main Plotting Logic for Figures 10/11 ---

# Load flux to climate response data for both EC and UFLUX at validation sites
# Figure 10 uses (PPFD, NEE), Figure 11 uses (TA, Reco)
x_name = 'PPFD' # Independent variable (PPFD or TA)
y_name = 'NEE' # Dependent variable (NEE or Reco)
dfp = pd.read_parquet(project_directory.joinpath(f'4_analysis/{y_name}_{x_name}_response.parquet'))


# Setup canvas for 11 subplots (one for each IGBP land cover type)
fig = plt.figure(figsize = (16, 12))
gs = gridspec.GridSpec(12, 16, figure = fig)
gs.update(wspace = 10, hspace = 2) # Original spacing

# Define 11 subplot locations
ax1 = plt.subplot(gs[0:4, :4]); ax2 = plt.subplot(gs[0:4, 4:8]); ax3 = plt.subplot(gs[0:4, 8:12]); ax4 = plt.subplot(gs[0:4, 12::])
ax5 = plt.subplot(gs[4:8, :4]); ax6 = plt.subplot(gs[4:8, 4:8]); ax7 = plt.subplot(gs[4:8, 8:12]); ax8 = plt.subplot(gs[4:8, 12::])
ax9 = plt.subplot(gs[8:, 2:6]); ax10 = plt.subplot(gs[8:, 6:10]); ax11 = plt.subplot(gs[8:, 10:14])

axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11]

# Iterate across IGBP land cover types
for cnt, igbp_code in enumerate(['CRO', 'CSH', 'DBF', 'EBF', 'ENF', 'GRA', 'MF', 'OSH', 'WET', 'WSA', 'SAV']):
    ax = axes[cnt]
    
    # Calculate site count metadata
    n_sites, n_years = get_n_sites_years(dfp, igbp_code)
    
    # Aggregate data by date and IGBP, then filter for the current IGBP code
    dfpt = dfp[['IGBP', f'EC_{y_name}', f'UFLUX_{y_name}', x_name]].reset_index().dropna()
    dfpt = dfpt.groupby(['datetime', 'IGBP']).mean().reset_index().set_index('datetime')
    dfpt = dfpt[dfpt['IGBP'] == igbp_code].dropna()
    
    # Scatter plot EC and UFLUX data points
    ax.scatter(dfpt[x_name], dfpt[f'EC_{y_name}'], color = nature_colors[0], s = 15, edgecolor = 'gray', alpha = 0.3)
    ax.scatter(dfpt[x_name], dfpt[f'UFLUX_{y_name}'], color = nature_colors[1], s = 15, edgecolor = 'gray', alpha = 0.3)

    # Sort data by x-variable for continuous curve fitting
    dfpt = dfpt.sort_values(by = x_name)
    
    # Plot curve fit for EC data (solid line)
    plot_curve_fitting('poly2', ax, dfpt[x_name], dfpt[f'EC_{y_name}'], precision = 2, with_text = False, lineshape = '-')
    # Plot curve fit for UFLUX data (dashed-dot line)
    plot_curve_fitting('poly2', ax, dfpt[x_name], dfpt[f'UFLUX_{y_name}'], precision = 2, with_text = False, lineshape = '-.')

    # Determine text box location (low for NEE, high for Reco)
    if y_name == 'NEE':
        text_location = (0.05, 0.05)
    elif y_name == 'Reco':
        text_location = (0.05, 0.8)
    else:
        raise Exception('y_name must be `NEE` or `Reco`!')
        
    # Add IGBP code and site count info
    ax.text(
        *text_location,
        f"{igbp_code}\n{n_sites} towers\n{n_years} site-years",
        transform = ax.transAxes,
        bbox=dict(
            facecolor='none',
            edgecolor='none',
            boxstyle='round,pad=0.3',
            alpha=0.9
        )
    )

# Set axis labels only on the center plots to reduce clutter
if (x_name == 'PPFD') and (y_name == 'NEE'):
    axes[4].set_ylabel(r'NEE ($gC \ m^{-2} \ d^{-1}$)')
    axes[9].set_xlabel(r'PPFD ($\mu mol \ Photon \ m^{-2} \ s^{-1}$)')
elif (x_name == 'TA') and (y_name == 'Reco'):
    axes[4].set_ylabel(r'Reco ($gC \ m^{-2} \ d^{-1}$)')
    axes[9].set_xlabel(r'TA ($^\circ C$)')
else:
    raise Exception('x_name and y_name must be `TA` and `Reco` or `PPFD` and `NEE`!')