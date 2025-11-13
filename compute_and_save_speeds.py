import pandas as pd
import numpy as np

def compute_and_save_speeds(df, fps, start_frame, end_frame, meters_per_pixel=None,
                            csv_outpath=None, smooth_window=3):
    """
    Compute instantaneous and smoothed ball speeds from tracking data.
    Either real_distance_meters or meters_per_pixel must be provided.
    """
    # Subset df to requested range
    df_seg = df[(df['frame'] >= start_frame) & (df['frame'] <= end_frame)].reset_index(drop=True).copy()
    if df_seg.empty:
        raise ValueError("No frames in requested range.")

    # Compute per-frame pixel displacement and speed
    dp_pixels = [np.nan]
    for i in range(1, len(df_seg)):
        dx = df_seg.loc[i, 'x'] - df_seg.loc[i-1, 'x']
        dy = df_seg.loc[i, 'y'] - df_seg.loc[i-1, 'y']
        dp_pixels.append(float(np.hypot(dx, dy)))

    df_seg['dp_pixels'] = dp_pixels
    df_seg['dm_m'] = df_seg['dp_pixels'] * meters_per_pixel
    df_seg['speed_m_s'] = df_seg['dm_m'] * fps
    df_seg['speed_mph'] = df_seg['speed_m_s'] * 2.2369362920544

    # Smoothing
    if smooth_window is not None and smooth_window > 1:
        df_seg['speed_m_s_smooth'] = df_seg['speed_m_s'].rolling(window=smooth_window, min_periods=1, center=True).mean()
        df_seg['speed_mph_smooth'] = df_seg['speed_mph'].rolling(window=smooth_window, min_periods=1, center=True).mean()
    else:
        df_seg['speed_m_s_smooth'] = df_seg['speed_m_s']
        df_seg['speed_mph_smooth'] = df_seg['speed_mph']

    # Summary
    valid_speeds = df_seg['speed_m_s'].dropna()
    if not valid_speeds.empty:
        avg_m_s = float(valid_speeds.mean())
        max_m_s = float(valid_speeds.max())
        avg_mph = avg_m_s * 2.2369362920544
        max_mph = max_m_s * 2.2369362920544
    else:
        avg_m_s = max_m_s = avg_mph = max_mph = np.nan

    summary = {
        'meters_per_pixel': meters_per_pixel,
        'avg_speed_m_s': avg_m_s,
        'max_speed_m_s': max_m_s,
        'avg_speed_mph': avg_mph,
        'max_speed_mph': max_mph
    }

    if csv_outpath:
        df_seg.to_csv(csv_outpath, index=False)

    return df_seg, summary
