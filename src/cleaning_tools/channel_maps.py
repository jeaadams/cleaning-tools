"""
Channel map plotting with Keplerian mask and sigma contour overlays.

This module provides tools for creating ALMA channel map visualizations with:
- Keplerian mask contours (from FITS files)
- CARTA-style sigma contours (3σ, 5σ, etc.)
- CARTA-style Gaussian smoothing before contouring
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import bettermoments as bm
from ipywidgets import interact, IntSlider, Checkbox, FloatSlider, VBox, HBox, Layout


class ChannelMapPlotter:
    """
    Channel map plotter with Keplerian mask and sigma contour support.

    This class handles loading ALMA data cubes and creating channel map
    visualizations with optional Keplerian mask contours and sigma-based
    contours (using CARTA-style smoothing).

    Parameters
    ----------
    cube_path : str
        Path to the FITS cube file
    keplerian_mask_path : str, optional
        Path to Keplerian mask FITS file

    Attributes
    ----------
    data : ndarray
        Data cube [nchan, ny, nx]
    velax : ndarray
        Velocity axis in km/s
    header : FITS header
        Header from the cube file
    keplerian_mask : ndarray or None
        Keplerian mask data [nchan, ny, nx]
    rms : float or None
        RMS noise level (calculated from line-free channels)
    """

    def __init__(self, cube_path, keplerian_mask_path=None):
        """Initialize the channel map plotter."""
        self.cube_path = cube_path
        self.keplerian_mask_path = keplerian_mask_path

        # Load cube using bettermoments
        self.data, self.velax = bm.load_cube(cube_path)
        self.header = fits.getheader(cube_path)

        # Load Keplerian mask if provided
        self.keplerian_mask = None
        if keplerian_mask_path:
            self.keplerian_mask = bm.get_user_mask(
                data=self.data,
                user_mask_path=keplerian_mask_path
            )

        # RMS will be calculated when needed
        self.rms = None
        self._smoothed_cube = None
        self._smooth_sigma = None

    def get_wcs_extent(self):
        """
        Get the WCS extent for plotting in arcseconds.

        Returns
        -------
        tuple or None
            (x_max, x_min, y_min, y_max) in arcseconds, or None if WCS not available
        """
        try:
            ra_offset = 3600. * self.header['CDELT1'] * (
                np.arange(self.header['NAXIS1']) - (self.header['CRPIX1'] - 1)
            )
            dec_offset = 3600. * self.header['CDELT2'] * (
                np.arange(self.header['NAXIS2']) - (self.header['CRPIX2'] - 1)
            )
            return (np.max(ra_offset), np.min(ra_offset),
                    np.min(dec_offset), np.max(dec_offset))
        except KeyError:
            return None

    def calculate_rms(self, line_free_channels=None, recalculate=False):
        """
        Calculate RMS noise from line-free channels using sigma clipping.

        This uses the same robust method as CARTA: iterative sigma clipping
        to exclude outliers before computing the standard deviation.

        Parameters
        ----------
        line_free_channels : list of int, optional
            List of line-free channel indices. If None, uses first and last
            15 channels.
        recalculate : bool, optional
            If True, recalculate even if RMS already exists. Default False.

        Returns
        -------
        float
            RMS noise level in the same units as the data
        """
        if self.rms is not None and not recalculate:
            return self.rms

        nchans = len(self.velax)

        # Default to edge channels if not specified
        if line_free_channels is None:
            line_free_channels = list(range(0, 15)) + list(range(nchans - 15, nchans))

        # Get line-free data
        line_free_data = self.data[line_free_channels, :, :].flatten()
        line_free_data = line_free_data[np.isfinite(line_free_data)]

        # Iterative sigma clipping (CARTA-style)
        data_clipped = line_free_data.copy()
        for _ in range(3):  # 3 iterations
            mean = np.mean(data_clipped)
            std = np.std(data_clipped)
            mask = np.abs(data_clipped - mean) < 3 * std
            data_clipped = data_clipped[mask]

        self.rms = np.std(data_clipped)
        return self.rms

    def get_smoothed_cube(self, smooth_sigma=1.7):
        """
        Get smoothed cube using CARTA-style Gaussian smoothing.

        CARTA uses a smoothing factor of 4 pixels (FWHM), which corresponds
        to sigma = 4 / 2.355 ≈ 1.7 pixels. Results are cached.

        Parameters
        ----------
        smooth_sigma : float, optional
            Gaussian smoothing sigma in pixels. Default is 1.7 (CARTA-style).
            Set to 0 or None for no smoothing.

        Returns
        -------
        ndarray
            Smoothed data cube [nchan, ny, nx]
        """
        # Return original if no smoothing requested
        if smooth_sigma is None or smooth_sigma <= 0:
            return self.data

        # Return cached version if sigma matches
        if (self._smoothed_cube is not None and
            self._smooth_sigma == smooth_sigma):
            return self._smoothed_cube

        # Compute smoothed cube
        nchans = len(self.velax)
        self._smoothed_cube = np.zeros_like(self.data)

        for i in range(nchans):
            self._smoothed_cube[i, :, :] = gaussian_filter(
                self.data[i, :, :],
                sigma=smooth_sigma
            )

        self._smooth_sigma = smooth_sigma
        return self._smoothed_cube

    def calculate_rms_smoothed(self, smooth_sigma=1.7, line_free_channels=None):
        """
        Calculate RMS on smoothed data.

        When smoothing data, the RMS changes and must be recalculated on
        the smoothed cube.

        Parameters
        ----------
        smooth_sigma : float, optional
            Gaussian smoothing sigma in pixels. Default 1.7 (CARTA-style).
        line_free_channels : list of int, optional
            List of line-free channel indices.

        Returns
        -------
        float
            RMS noise level on smoothed data
        """
        smoothed_cube = self.get_smoothed_cube(smooth_sigma)

        nchans = len(self.velax)
        if line_free_channels is None:
            line_free_channels = list(range(0, 15)) + list(range(nchans - 15, nchans))

        # Get line-free data from smoothed cube
        line_free_smooth = smoothed_cube[line_free_channels, :, :].flatten()
        line_free_smooth = line_free_smooth[np.isfinite(line_free_smooth)]

        # Sigma clipping
        data_clipped = line_free_smooth.copy()
        for _ in range(3):
            mean = np.mean(data_clipped)
            std = np.std(data_clipped)
            mask = np.abs(data_clipped - mean) < 3 * std
            data_clipped = data_clipped[mask]

        return np.std(data_clipped)

    def plot_channel_maps(self, start_channel=None, end_channel=None,
                         sigma_levels=[3, 5], smooth_sigma=1.7,
                         show_keplerian_mask=True, zoom_size=None,
                         channels_per_page=20, figsize_per_row=3,
                         cmap='RdBu_r', vmin_sigma=-2, vmax_sigma=10,
                         keplerian_color='lime', keplerian_linewidth=1.5,
                         sigma_color='black', sigma_linewidth=1.5):
        """
        Plot channel maps with sigma contours and optional Keplerian mask.

        Parameters
        ----------
        start_channel : int, optional
            First channel to plot. Default: 0
        end_channel : int, optional
            Last channel to plot (inclusive). Default: last channel
        sigma_levels : list of float, optional
            Sigma levels for contours (e.g., [3, 5, 10]). Default: [3, 5]
        smooth_sigma : float, optional
            Gaussian smoothing sigma in pixels for sigma contours.
            Default: 1.7 (CARTA-style). Set to 0 for no smoothing.
        show_keplerian_mask : bool, optional
            Whether to show Keplerian mask contours. Default: True
        zoom_size : int, optional
            Size in pixels from center to display. If None, shows full image.
        channels_per_page : int, optional
            Number of channels per figure. Default: 20
        figsize_per_row : float, optional
            Figure height per row in inches. Default: 3
        cmap : str, optional
            Matplotlib colormap name. Default: 'RdBu_r'
        vmin_sigma : float, optional
            Lower color limit in units of sigma. Default: -2
        vmax_sigma : float, optional
            Upper color limit in units of sigma. Default: 10
        keplerian_color : str, optional
            Color for Keplerian mask contours. Default: 'lime'
        keplerian_linewidth : float, optional
            Line width for Keplerian mask contours. Default: 1.5
        sigma_color : str, optional
            Color for sigma contours. Default: 'black'
        sigma_linewidth : float, optional
            Line width for sigma contours. Default: 1.5

        Returns
        -------
        None
            Displays matplotlib figures
        """
        # Set defaults
        nchans = len(self.velax)
        if start_channel is None:
            start_channel = 0
        if end_channel is None:
            end_channel = nchans - 1

        # Calculate RMS
        if smooth_sigma and smooth_sigma > 0:
            rms = self.calculate_rms_smoothed(smooth_sigma)
            cube_to_plot = self.get_smoothed_cube(smooth_sigma)
            print(f"Using smoothed data (sigma={smooth_sigma:.2f} pixels)")
            print(f"RMS on smoothed data: {rms*1000:.3f} mJy/beam")
        else:
            rms = self.calculate_rms()
            cube_to_plot = self.data
            print(f"Using native resolution data")
            print(f"RMS: {rms*1000:.3f} mJy/beam")

        print(f"Sigma levels: {sigma_levels}")
        for level in sigma_levels:
            print(f"  {level}σ = {level*rms*1000:.3f} mJy/beam")

        # Get WCS extent
        extent = self.get_wcs_extent()
        _, ny, nx = self.data.shape
        cy, cx = ny // 2, nx // 2

        # Determine zoom
        if zoom_size is None:
            y_start, y_end = 0, ny
            x_start, x_end = 0, nx
        else:
            y_start = max(0, cy - zoom_size)
            y_end = min(ny, cy + zoom_size)
            x_start = max(0, cx - zoom_size)
            x_end = min(nx, cx + zoom_size)

        # Create pages
        nchans_to_plot = end_channel - start_channel + 1
        n_pages = int(np.ceil(nchans_to_plot / channels_per_page))

        for page in range(n_pages):
            page_start = start_channel + page * channels_per_page
            page_end = min(page_start + channels_per_page, end_channel + 1)
            n_chans_this_page = page_end - page_start

            # Create subplot grid
            ncols = 5
            nrows = int(np.ceil(n_chans_this_page / ncols))

            fig, axes = plt.subplots(nrows, ncols,
                                    figsize=(15, figsize_per_row * nrows))
            if nrows == 1:
                axes = axes.reshape(1, -1)

            for idx, chan in enumerate(range(page_start, page_end)):
                row = idx // ncols
                col = idx % ncols
                ax = axes[row, col]

                # Get channel data
                channel_data_full = cube_to_plot[chan, :, :]
                channel_data = channel_data_full[y_start:y_end, x_start:x_end]

                # Calculate peak SNR
                peak_signal = np.nanmax(channel_data)
                snr = peak_signal / rms

                # Color scaling
                vmin = vmin_sigma * rms
                vmax = max(vmax_sigma * rms, peak_signal * 1.2)

                # Plot image
                if extent:
                    # Calculate zoomed extent
                    x_max, x_min, y_min_full, y_max_full = extent
                    x_range = np.linspace(x_max, x_min, nx)
                    y_range = np.linspace(y_min_full, y_max_full, ny)

                    extent_zoomed = [
                        x_range[x_end - 1], x_range[x_start],
                        y_range[y_start], y_range[y_end - 1]
                    ]
                    im = ax.imshow(channel_data, origin='lower', cmap=cmap,
                                  vmin=vmin, vmax=vmax, extent=extent_zoomed,
                                  aspect='auto')
                else:
                    im = ax.imshow(channel_data, origin='lower', cmap=cmap,
                                  vmin=vmin, vmax=vmax, aspect='auto')

                # Add sigma contours (positive)
                if sigma_levels:
                    levels_pos = np.array(sigma_levels) * rms
                    levels_pos = levels_pos[levels_pos < peak_signal]
                    if len(levels_pos) > 0:
                        if extent:
                            X, Y = np.meshgrid(
                                np.linspace(extent_zoomed[0], extent_zoomed[1],
                                          channel_data.shape[1]),
                                np.linspace(extent_zoomed[2], extent_zoomed[3],
                                          channel_data.shape[0])
                            )
                            ax.contour(X, Y, channel_data, levels=levels_pos,
                                     colors=sigma_color, linewidths=sigma_linewidth,
                                     alpha=0.8)
                        else:
                            ax.contour(channel_data, levels=levels_pos,
                                     colors=sigma_color, linewidths=sigma_linewidth,
                                     alpha=0.8)

                # Add Keplerian mask contours
                if show_keplerian_mask and self.keplerian_mask is not None:
                    mask_channel = self.keplerian_mask[chan, y_start:y_end,
                                                       x_start:x_end]
                    if np.any(mask_channel > 0):
                        if extent:
                            ax.contour(X, Y, mask_channel, levels=[0.5],
                                     colors=keplerian_color,
                                     linewidths=keplerian_linewidth,
                                     alpha=0.8)
                        else:
                            ax.contour(mask_channel, levels=[0.5],
                                     colors=keplerian_color,
                                     linewidths=keplerian_linewidth,
                                     alpha=0.8)

                # Add center marker if zoomed
                if zoom_size is not None:
                    if extent:
                        ax.plot(0, 0, '+', color='yellow', markersize=8,
                               markeredgewidth=1.5)
                    else:
                        center_x = channel_data.shape[1] // 2
                        center_y = channel_data.shape[0] // 2
                        ax.plot(center_x, center_y, '+', color='yellow',
                               markersize=8, markeredgewidth=1.5)

                # Title with SNR color coding
                color = 'green' if snr > 5 else ('orange' if snr > 3 else 'red')
                ax.set_title(
                    f'Ch {chan}: {self.velax[chan]:.2f} km/s\n'
                    f'Peak: {snr:.1f}σ',
                    fontsize=10, color=color,
                    fontweight='bold' if snr > 5 else 'normal'
                )

                # Axis labels
                if extent:
                    ax.set_xlabel('ΔRA [arcsec]', fontsize=8)
                    ax.set_ylabel('ΔDEC [arcsec]', fontsize=8)
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])

            # Hide unused subplots
            for idx in range(n_chans_this_page, nrows * ncols):
                row = idx // ncols
                col = idx % ncols
                axes[row, col].axis('off')

            # Overall title
            contour_desc = f"σ contours: {sigma_levels}"
            if show_keplerian_mask and self.keplerian_mask is not None:
                contour_desc += " (black), Keplerian mask (lime)"

            plt.suptitle(
                f'Channel Maps - Channels {page_start}-{page_end-1}\n{contour_desc}',
                fontsize=12, y=0.995
            )
            plt.tight_layout()
            plt.show()

    def get_info(self):
        """
        Get information about the loaded cube.

        Returns
        -------
        dict
            Dictionary with cube information
        """
        return {
            'shape': self.data.shape,
            'velocity_range': (float(self.velax[0]), float(self.velax[-1])),
            'n_channels': len(self.velax),
            'has_keplerian_mask': self.keplerian_mask is not None,
            'rms': self.rms,
            'rms_mjy': self.rms * 1000 if self.rms is not None else None
        }

    def plot_interactive(self, sigma_levels=[3, 5], smooth_sigma=1.7,
                        zoom_size=None, contour_region=None, cmap='RdBu_r',
                        vmin_sigma=-2, vmax_sigma=10, keplerian_color='lime',
                        keplerian_linewidth=1.5, sigma_color='black',
                        sigma_linewidth=1.5, figsize=(8, 8)):
        """
        Create an interactive channel map viewer with slider controls.

        This creates an ipywidgets interface with:
        - Channel slider to navigate through the cube
        - Toggle for showing/hiding sigma contours
        - Toggle for showing/hiding Keplerian mask
        - Smoothing sigma slider

        Parameters
        ----------
        sigma_levels : list of float, optional
            Sigma levels for contours (e.g., [3, 5, 10]). Default: [3, 5]
        smooth_sigma : float, optional
            Initial Gaussian smoothing sigma in pixels. Default: 1.7 (CARTA-style)
        zoom_size : int, optional
            Size in pixels from center to display. If None, shows full image.
        contour_region : float, optional
            If specified, only show contours within this radius (in arcseconds)
            from the center. Useful for masking noise at image edges. Default: None
        cmap : str, optional
            Matplotlib colormap name. Default: 'RdBu_r'
        vmin_sigma : float, optional
            Lower color limit in units of sigma. Default: -2
        vmax_sigma : float, optional
            Upper color limit in units of sigma. Default: 10
        keplerian_color : str, optional
            Color for Keplerian mask contours. Default: 'lime'
        keplerian_linewidth : float, optional
            Line width for Keplerian mask contours. Default: 1.5
        sigma_color : str, optional
            Color for sigma contours. Default: 'black'
        sigma_linewidth : float, optional
            Line width for sigma contours. Default: 1.5
        figsize : tuple, optional
            Figure size (width, height) in inches. Default: (8, 8)

        Returns
        -------
        ipywidgets.interact
            Interactive widget object
        """
        nchans = len(self.velax)
        _, ny, nx = self.data.shape
        cy, cx = ny // 2, nx // 2

        # Pre-calculate native RMS to avoid repeated calculation
        print("Pre-calculating RMS for native resolution...")
        native_rms = self.calculate_rms()
        print(f"Native RMS: {native_rms*1000:.3f} mJy/beam")

        # Pre-calculate initial smoothed cube and RMS
        if smooth_sigma > 0:
            print(f"Pre-calculating smoothed cube (sigma={smooth_sigma:.1f})...")
            initial_smoothed = self.get_smoothed_cube(smooth_sigma)
            initial_rms = self.calculate_rms_smoothed(smooth_sigma)
            print(f"Smoothed RMS: {initial_rms*1000:.3f} mJy/beam")

        # Determine zoom
        if zoom_size is None:
            y_start, y_end = 0, ny
            x_start, x_end = 0, nx
        else:
            y_start = max(0, cy - zoom_size)
            y_end = min(ny, cy + zoom_size)
            x_start = max(0, cx - zoom_size)
            x_end = min(nx, cx + zoom_size)

        # Get WCS extent
        extent = self.get_wcs_extent()

        # Calculate zoomed extent if available
        extent_zoomed = None
        if extent:
            x_max, x_min, y_min_full, y_max_full = extent
            x_range = np.linspace(x_max, x_min, nx)
            y_range = np.linspace(y_min_full, y_max_full, ny)
            extent_zoomed = [
                x_range[x_end - 1], x_range[x_start],
                y_range[y_start], y_range[y_end - 1]
            ]

        # Cache for RMS values at different smoothing levels
        rms_cache = {0: native_rms}
        if smooth_sigma > 0:
            rms_cache[smooth_sigma] = initial_rms

        # Create the figure once outside the update function
        fig, ax = plt.subplots(figsize=figsize)

        # Initialize plot elements that we'll update
        im_artist = None
        contour_artists = []
        mask_contour_artists = []
        center_marker = None
        title_obj = None

        def plot_channel(channel, show_sigma_contours, show_keplerian,
                        smoothing_sigma):
            """Inner plotting function for interactive widget."""
            nonlocal im_artist, contour_artists, mask_contour_artists, center_marker, title_obj

            # Get smoothed or native data (use cache when possible)
            if smoothing_sigma > 0:
                # Check if we need to recalculate
                if smoothing_sigma not in rms_cache:
                    rms_cache[smoothing_sigma] = self.calculate_rms_smoothed(smoothing_sigma)
                rms = rms_cache[smoothing_sigma]
                cube_to_plot = self.get_smoothed_cube(smoothing_sigma)
            else:
                rms = native_rms
                cube_to_plot = self.data

            # Get channel data
            channel_data_full = cube_to_plot[channel, :, :]
            channel_data = channel_data_full[y_start:y_end, x_start:x_end]

            # Calculate peak SNR
            peak_signal = np.nanmax(channel_data)
            snr = peak_signal / rms

            # Color scaling
            vmin = vmin_sigma * rms
            vmax = max(vmax_sigma * rms, peak_signal * 1.2)

            # Clear previous contours - more robust approach
            for cs in contour_artists:
                # QuadContourSet uses 'collections' (a LineCollection list)
                if hasattr(cs, 'collections'):
                    for coll in cs.collections:
                        if coll in ax.collections:
                            coll.remove()
                # Fallback for other contour types
                elif hasattr(cs, 'remove'):
                    cs.remove()
            contour_artists.clear()

            for cs in mask_contour_artists:
                if hasattr(cs, 'collections'):
                    for coll in cs.collections:
                        if coll in ax.collections:
                            coll.remove()
                elif hasattr(cs, 'remove'):
                    cs.remove()
            mask_contour_artists.clear()

            # Remove center marker
            if center_marker is not None:
                for marker in center_marker:
                    if marker in ax.lines:
                        marker.remove()
                center_marker = None

            # Create mask for contour region if specified
            contour_mask = None
            if contour_region is not None and extent_zoomed:
                # Create a circular mask centered on (0, 0) in arcseconds
                Y_grid, X_grid = np.mgrid[0:channel_data.shape[0], 0:channel_data.shape[1]]
                # Map to physical coordinates
                x_coords = np.linspace(extent_zoomed[0], extent_zoomed[1], channel_data.shape[1])
                y_coords = np.linspace(extent_zoomed[2], extent_zoomed[3], channel_data.shape[0])
                X_phys, Y_phys = np.meshgrid(x_coords, y_coords)
                radius = np.sqrt(X_phys**2 + Y_phys**2)
                contour_mask = radius > contour_region

            # Update or create image
            if im_artist is None:
                # First time - create everything
                if extent_zoomed:
                    im_artist = ax.imshow(channel_data, origin='lower', cmap=cmap,
                                  vmin=vmin, vmax=vmax, extent=extent_zoomed,
                                  aspect='auto')
                else:
                    im_artist = ax.imshow(channel_data, origin='lower', cmap=cmap,
                                  vmin=vmin, vmax=vmax, aspect='auto')

                # Add colorbar
                cbar = plt.colorbar(im_artist, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Intensity [Jy/beam]', fontsize=10)

                # Set axis labels
                if extent_zoomed:
                    ax.set_xlabel('ΔRA [arcsec]', fontsize=11)
                    ax.set_ylabel('ΔDEC [arcsec]', fontsize=11)
                else:
                    ax.set_xlabel('X [pixels]', fontsize=11)
                    ax.set_ylabel('Y [pixels]', fontsize=11)
            else:
                # Update existing image data
                im_artist.set_data(channel_data)
                im_artist.set_clim(vmin, vmax)

            # Add sigma contours
            if show_sigma_contours and sigma_levels:
                levels_pos = np.array(sigma_levels) * rms
                levels_pos = levels_pos[levels_pos < peak_signal]
                if len(levels_pos) > 0:
                    # Apply mask if specified
                    data_for_contours = channel_data.copy()
                    if contour_mask is not None:
                        data_for_contours[contour_mask] = np.nan

                    if extent_zoomed:
                        X, Y = np.meshgrid(
                            np.linspace(extent_zoomed[0], extent_zoomed[1],
                                      channel_data.shape[1]),
                            np.linspace(extent_zoomed[2], extent_zoomed[3],
                                      channel_data.shape[0])
                        )
                        cs = ax.contour(X, Y, data_for_contours, levels=levels_pos,
                                     colors=sigma_color, linewidths=sigma_linewidth,
                                     alpha=0.8)
                    else:
                        cs = ax.contour(data_for_contours, levels=levels_pos,
                                     colors=sigma_color, linewidths=sigma_linewidth,
                                     alpha=0.8)
                    contour_artists.append(cs)

            # Add Keplerian mask contours
            if show_keplerian and self.keplerian_mask is not None:
                mask_channel = self.keplerian_mask[channel, y_start:y_end,
                                                   x_start:x_end]
                if np.any(mask_channel > 0):
                    # Apply region mask if specified
                    mask_for_contours = mask_channel.copy()
                    if contour_mask is not None:
                        mask_for_contours[contour_mask] = 0

                    if extent_zoomed:
                        cs_mask = ax.contour(X, Y, mask_for_contours, levels=[0.5],
                                 colors=keplerian_color,
                                 linewidths=keplerian_linewidth,
                                 alpha=0.8)
                    else:
                        cs_mask = ax.contour(mask_for_contours, levels=[0.5],
                                 colors=keplerian_color,
                                 linewidths=keplerian_linewidth,
                                 alpha=0.8)
                    mask_contour_artists.append(cs_mask)

            # Add center marker if zoomed
            if zoom_size is not None:
                if extent_zoomed:
                    center_marker = ax.plot(0, 0, '+', color='yellow', markersize=12,
                           markeredgewidth=2)
                else:
                    center_x = channel_data.shape[1] // 2
                    center_y = channel_data.shape[0] // 2
                    center_marker = ax.plot(center_x, center_y, '+', color='yellow',
                           markersize=12, markeredgewidth=2)

            # Update title with SNR info
            color = 'green' if snr > 5 else ('orange' if snr > 3 else 'red')
            title_text = (f'Channel {channel}: {self.velax[channel]:.2f} km/s\n'
                         f'Peak: {snr:.1f}σ ({peak_signal*1000:.2f} mJy/beam), '
                         f'RMS: {rms*1000:.3f} mJy/beam')
            ax.set_title(title_text, fontsize=12, color=color, fontweight='bold',
                        pad=10)

            # Redraw the canvas
            fig.canvas.draw_idle()

        # Create interactive widget
        return interact(
            plot_channel,
            channel=IntSlider(
                value=nchans // 2,
                min=0,
                max=nchans - 1,
                step=1,
                description='Channel:',
                continuous_update=False,
                layout=Layout(width='600px')
            ),
            show_sigma_contours=Checkbox(
                value=True,
                description=f'Show sigma contours ({sigma_levels})',
                layout=Layout(width='300px')
            ),
            show_keplerian=Checkbox(
                value=self.keplerian_mask is not None,
                description='Show Keplerian mask',
                disabled=self.keplerian_mask is None,
                layout=Layout(width='300px')
            ),
            smoothing_sigma=FloatSlider(
                value=smooth_sigma,
                min=0,
                max=5.0,
                step=0.1,
                description='Smooth σ:',
                continuous_update=False,
                layout=Layout(width='600px')
            )
        )
