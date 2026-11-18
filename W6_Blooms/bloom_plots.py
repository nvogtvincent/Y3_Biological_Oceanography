"""Science calculations and figure styling for the W6 BGC-Argo practical.

Place this file beside W6_Blooms_modern.ipynb. Students change scientific
settings in notebook cells; colour palettes, font sizes and layout live here.
Data are not modified in place. Pressure (dbar) approximates depth (m).
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize

# Presentation-only choices: adjust here, not in student notebook cells.
INK = '#203248'
MUTED = '#637387'
GRID = '#DDE5EC'
SEA = '#EAF3F7'
LAND = '#CDD7D6'
ACCENT = '#D55639'
COLOURS = {'TEMP':'#DF744A','PSAL':'#247B91','DOXY':'#5878BA',
           'CHLA':'#3B906C','BBP':'#9264A7','MLD':'#34445A'}
CMAPS = {'TEMP':'magma', 'PSAL':'YlGnBu', 'DOXY':'cividis',
         'CHLA':'viridis','BBP':'PuBuGn','LOCATION':'plasma'}
COLOURBAR_STEPS = {'TEMP':5.,'PSAL':0.5,'DOXY':25.,'CHLA':0.1,'BBP':0.0005}


def _style():
    """Consistent clean figure typography and restrained axes."""
    mpl.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':11, 'axes.titlesize':13,
        'axes.titleweight':'bold', 'axes.labelsize':11,
        'axes.labelcolor':INK, 'text.color':INK,
        'xtick.color':MUTED, 'ytick.color':MUTED,
        'axes.edgecolor':GRID, 'axes.spines.top':False,
        'axes.spines.right':False, 'axes.linewidth':0.8,
        'figure.facecolor':'white', 'axes.facecolor':'white',
        'savefig.facecolor':'white', 'grid.color':GRID,
        'grid.linewidth':0.65, 'grid.alpha':0.8,
    })


def _tidy(ax, horizontal=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=0, pad=7)
    ax.grid(axis='y' if horizontal else 'x', linewidth=0.6)
    ax.set_axisbelow(True)


def _projection(lon, float_number):
    central_lon = 180 if str(float_number) in ('6901658','5907058') else 0
    return ccrs.PlateCarree(central_longitude=central_lon), central_lon


def _map_extent(ax, lon, lat, margin=0.2, minimum=3):
    valid = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(valid):
        raise ValueError('No valid geographical coordinates')
    x0,x1=np.min(lon[valid]),np.max(lon[valid])
    y0,y1=np.min(lat[valid]),np.max(lat[valid])
    dx=max((x1-x0)*margin,minimum)
    dy=max((y1-y0)*margin,minimum)
    bounds=[x0-dx,x1+dx,max(-89,y0-dy),min(89,y1+dy)]
    ax.set_extent(bounds, crs=ax.projection)
    return bounds


def _base_map(ax, coast_width=0.6):
    ax.add_feature(cfeature.OCEAN,facecolor=SEA,zorder=0)
    ax.add_feature(cfeature.LAND,facecolor=LAND,edgecolor='none',zorder=1)
    ax.coastlines(resolution='110m',linewidth=coast_width,color=INK,zorder=2)


def _track(ax, lon, lat, times, cmap='plasma', size=17):
    """Chronological track; avoid spurious lines across map seams."""
    good=np.isfinite(lon)&np.isfinite(lat)&np.isfinite(times)
    index=np.flatnonzero(good)
    index=index[np.argsort(times[index],kind='stable')]
    for a,b in zip(index[:-1],index[1:]):
        if abs(lon[b]-lon[a])<=180:
            ax.plot([lon[a],lon[b]],[lat[a],lat[b]],color=INK,
                    linewidth=1.0,alpha=0.55,transform=ax.projection,zorder=3)
    if not len(index):
        return None,index
    sc=ax.scatter(lon[index],lat[index],c=times[index],cmap=cmap,
                  vmin=np.min(times[index]),vmax=np.max(times[index]),
                  s=size,linewidths=0,transform=ax.projection,zorder=4)
    return sc,index


def _dates_numeric(dates):
    return mdates.date2num(np.asarray(dates,dtype='datetime64[ms]').astype(object))

def _as_float(values):
    """Convert NetCDF masked arrays into floating arrays containing NaNs."""
    return np.asarray(np.ma.filled(np.ma.asarray(values, dtype=float), np.nan), dtype=float)


def _vertical_grid(data, pressure, depth, max_vertical_gap=20,
                   nonnegative=False):
    """
    Interpolate profiles onto a specified depth grid.

    - Uses valid pressure/value pairs only.
    - Does not extrapolate beyond observed depths.
    - Does not interpolate across gaps larger than max_vertical_gap.
    - Retains individual observations at matching grid levels.
    - Optionally excludes negative values.
    """
    values = _as_float(data)
    pres = _as_float(pressure)

    if values.shape != pres.shape or values.ndim != 2:
        raise ValueError("Pressure and data must be matching 2-D arrays")

    if max_vertical_gap <= 0:
        raise ValueError("max_vertical_gap must be positive")

    depth = np.asarray(depth, dtype=float)
    result = np.full((len(values), len(depth)), np.nan)

    for i, (p, v) in enumerate(zip(pres, values)):

        # Identify valid observations
        valid = np.isfinite(p) & np.isfinite(v) & (p >= 0)

        if nonnegative:
            valid &= v >= 0

        p, v = p[valid], v[valid]

        if len(p) == 0:
            continue

        # Sort by pressure and remove duplicate levels
        order = np.argsort(p, kind='stable')
        p, v = p[order], v[order]
        p, idx = np.unique(p, return_index=True)
        v = v[idx]

        # Identify bounding pressure levels for each target depth
        left = np.searchsorted(p, depth, side='right') - 1
        safe = np.clip(left, 0, len(p)-1)
        right = np.clip(left+1, 0, len(p)-1)

        # Only interpolate within sufficiently small observed gaps
        valid_depth = (
            (left >= 0) &
            (left < len(p)-1) &
            (depth >= p[safe]) &
            (depth <= p[right]) &
            ((p[right] - p[safe]) <= max_vertical_gap)
        )

        if np.any(valid_depth):
            result[i, valid_depth] = np.interp(
                depth[valid_depth], p, v
            )

        # Preserve exact observations, including isolated levels
        nearest = np.searchsorted(p, depth)

        for ix in (
            np.clip(nearest, 0, len(p)-1),
            np.clip(nearest-1, 0, len(p)-1)
        ):
            match = np.isclose(
                depth, p[ix], atol=1e-8, rtol=0
            )
            result[i, match] = v[ix[match]]

    return result


def _chronological_order(dates):
    dates = np.asarray(dates, dtype='datetime64[ns]')
    valid = np.flatnonzero(~np.isnat(dates))
    if valid.size == 0:
        raise ValueError("No valid profile dates")
    return valid[np.argsort(dates[valid], kind='stable')]

def plot_float_trajectory(data, settings):
    """Figure 1: dated track, shaded coastlines and globe of map footprint."""
    _style()
    lon=_as_float(data['LONGITUDE']); lat=_as_float(data['LATITUDE'])
    dates=np.asarray(data['TIME_BGC'],dtype='datetime64[ns]')
    proj,central=_projection(lon,data['FLOAT_NUMBER'])
    x=((lon-central+180)%360)-180
    good=np.isfinite(x)&np.isfinite(lat)&~np.isnat(dates)
    if good.sum()<2: raise ValueError('At least two dated positions are required')
    t=np.full(len(dates),np.nan)
    t[good]=_dates_numeric(dates[good])

    fig=plt.figure(figsize=settings.get('figsize',(10.2,7.2)))
    ax=fig.add_axes([.09,.19,.84,.72],projection=proj)
    bounds=_map_extent(ax,x[good],lat[good],.12,1)
    _base_map(ax,.85)
    sc,index=_track(ax,x,lat,t,CMAPS['LOCATION'],size=25)
    ax.scatter(x[index[0]],lat[index[0]],s=92,marker='o',color='#25A77A',
               edgecolor='white',linewidth=1.4,transform=proj,zorder=6,label='First profile')
    ax.scatter(x[index[-1]],lat[index[-1]],s=90,marker='X',color=ACCENT,
               edgecolor='white',linewidth=.9,transform=proj,zorder=6,label='Last profile')
    grid=ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=.5,
                      color=MUTED,alpha=.35,linestyle=':')
    grid.top_labels=grid.right_labels=False
    grid.xlabel_style={'size':9,'color':MUTED};grid.ylabel_style={'size':9,'color':MUTED}
    ax.set_title('FIGURE 1  /  Float trajectory',loc='left',pad=15,color=INK)
    ax.legend(loc='lower left',frameon=True,facecolor='white',edgecolor=GRID,
              framealpha=.93,fontsize=9,markerscale=.65)

    # Globe: red outline is the geographical area visible on the main map.
    gx=((bounds[0]+bounds[1])/2+central+180)%360-180
    gy=(bounds[2]+bounds[3])/2
    globe=fig.add_axes([.715,.635,.19,.23],projection=ccrs.Orthographic(gx,gy))
    globe.set_global();_base_map(globe,.45)
    globe.gridlines(linewidth=.35,color=MUTED,alpha=.35,linestyle=':')
    n=160
    outline_x=np.r_[np.linspace(bounds[0],bounds[1],n),np.full(n,bounds[1]),
                    np.linspace(bounds[1],bounds[0],n),np.full(n,bounds[0]),bounds[0]]+central
    outline_y=np.r_[np.full(n,bounds[2]),np.linspace(bounds[2],bounds[3],n),
                    np.full(n,bounds[3]),np.linspace(bounds[3],bounds[2],n),bounds[2]]
    globe.plot(outline_x,outline_y,transform=ccrs.PlateCarree(),color=ACCENT,
               linewidth=1.8,zorder=6)

    cax=fig.add_axes([.19,.095,.63,.025])
    cb=fig.colorbar(sc,cax=cax,orientation='horizontal')
    cb.set_ticks([t[index[0]],t[index[-1]]]);cb.set_ticklabels([
        np.datetime_as_string(dates[index[0]],unit='D'),
        np.datetime_as_string(dates[index[-1]],unit='D')])
    cb.set_label('Observation date',color=MUTED,labelpad=5)
    cb.outline.set_visible(False);cb.ax.tick_params(length=0,labelsize=9)
    plt.show()

def plot_vertical_profile(data,settings):
    """Figure 2: measured vertical profile for an actual Argo cycle number."""
    _style()
    selected=settings['INDEX_PROFILE']
    cycle=np.asarray(data['CYCLE_NUMBER'])
    match=np.flatnonzero(cycle==selected)
    if not len(match):
        print(f'Cycle {selected} not found; available cycles: {np.unique(cycle)}')
        return selected
    i=int(match[0]); p=_as_float(data['PRESSURE'])[i]
    specs=[('TEMP','Temperature','°C',.5),('PSAL','Salinity','PSU',.05),
           ('DISS_OXY','Dissolved oxygen','µmol kg$^{-1}$',10),
           ('CHLA','Chlorophyll-a','mg m$^{-3}$',.05),
           ('BBP','Backscattering','m$^{-1}$',.0001)]
    fig,axes=plt.subplots(1,5,sharey=True,figsize=settings.get('figsize',(16,5.8)))
    fig.subplots_adjust(left=.075,right=.98,bottom=.19,top=.75,wspace=.23)
    for ax,(name,label,units,pad) in zip(axes,specs):
        values=_as_float(data[name])[i]
        good=np.isfinite(values)&np.isfinite(p)
        key='DOXY' if name=='DISS_OXY' else name
        ax.plot(values[good],p[good],color=COLOURS[key],linewidth=1.4,alpha=.65)
        ax.scatter(values[good],p[good],color=COLOURS[key],s=15,alpha=.85,zorder=3)
        ax.set_title(label,loc='left',fontsize=11,color=COLOURS[key],pad=12)
        ax.set_xlabel(units,labelpad=9)
        _tidy(ax)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
        if good.any():ax.set_xlim(np.min(values[good])-pad,np.max(values[good])+pad)
    axes[0].set_ylabel('Pressure (dbar)')
    axes[0].set_ylim(settings.get('max_depth',200),0)
    dates=np.asarray(data['TIME_BGC'],dtype='datetime64[ns]')
    lat=_as_float(data['LATITUDE']);lon=_as_float(data['LONGITUDE'])
    date=np.datetime_as_string(dates[i],unit='D') if not np.isnat(dates[i]) else 'unknown date'
    lon180=((lon[i]+180)%360)-180
    fmt=lambda x,positive,negative: f'{abs(x):.2f}°{positive if x>=0 else negative}' if np.isfinite(x) else 'unknown'
    caption=f"Cycle {selected}   ·   {date}   ·   {fmt(lat[i],'N','S')}, {fmt(lon180,'E','W')}"
    fig.text(.075,.89,'FIGURE 2  /  Vertical profile',fontsize=15,fontweight='bold',color=INK)
    fig.text(.075,.84,caption,fontsize=10,color=MUTED)

    good=np.isfinite(lon)&np.isfinite(lat)&~np.isnat(dates)
    if good.any():
        proj,central=_projection(lon,data['FLOAT_NUMBER']);x=((lon-central+180)%360)-180
        # Align inset's right edge with the rightmost profile panel.
        right_edge = axes[-1].get_position().x1
        inset_width, inset_height = .19, .16
        inset = fig.add_axes([right_edge-inset_width, .795,
                              inset_width, inset_height], projection=proj)
        _map_extent(inset,x[good],lat[good],.25,5)
        _base_map(inset,.5)
        t=np.full(len(dates),np.nan);t[good]=_dates_numeric(dates[good])
        valid_idx=np.flatnonzero(good)
        valid_idx=valid_idx[np.argsort(t[valid_idx],kind='stable')]
        # All other observations and the trajectory are deliberately neutral.
        for a,b in zip(valid_idx[:-1],valid_idx[1:]):
            if abs(x[b]-x[a])<=180:
                inset.plot([x[a],x[b]],[lat[a],lat[b]],color='#8A969F',
                           linewidth=1.1,alpha=.8,transform=proj,zorder=3)
        inset.scatter(x[valid_idx],lat[valid_idx],s=12,color='#8997A1',
                      alpha=.75,edgecolors='none',transform=proj,zorder=4)
        if np.isfinite(x[i]) and np.isfinite(lat[i]):
            inset.scatter([x[i]],[lat[i]],s=190,marker='*',color=ACCENT,
                          edgecolors='white',linewidths=1.5,
                          transform=proj,zorder=8)
        inset.set_title('Selected profile',fontsize=9,pad=5)
    plt.show()
    return selected

def plot_bgc_timeseries(data,settings):
    """Figure 3: sorted observations; no vertical or temporal extrapolation."""
    _style()
    order=_chronological_order(data['TIME_BGC'])
    dates=np.asarray(data['TIME_BGC'],dtype='datetime64[ns]')[order]
    t=_dates_numeric(dates)
    if len(t)<2 or t[-1]<=t[0]: raise ValueError('At least two different profile dates are required')
    pressure=_as_float(data['PRESSURE'])[order]
    mld=_as_float(data['MLD'])[order]
    depth_max=settings.get('MAX_DEPTH',200)
    depth_step=settings.get('DEPTH_STEP',2)
    max_vertical=settings.get('MAX_VERTICAL_GAP',20)
    depth=np.arange(0,depth_max+depth_step*.5,depth_step)
    specs=[('TEMP','Temperature','°C',False,0),
           ('PSAL','Salinity','PSU',True,1),
           ('DISS_OXY','Dissolved oxygen','µmol kg$^{-1}$',True,0),
           ('CHLA','Chlorophyll-a','mg m$^{-3}$',True,1),
           ('BBP','Backscattering','m$^{-1}$',True,4)]
    fields={name:_vertical_grid(_as_float(data[name])[order],pressure,depth,max_vertical,nonneg)
            for name,_,_,nonneg,_ in specs}

    # Mask the *centres* of long sampling gaps in pcolormesh; no values are invented.
    max_time_gap=settings.get('MAX_TEMPORAL_GAP_DAYS',45)
    if max_time_gap<=0:raise ValueError('MAX_TEMPORAL_GAP_DAYS must be positive')
    extended_t=[t[0]];source_rows=[0]
    for j,dt in enumerate(np.diff(t)):
        if dt>max_time_gap:
            half=min(max_time_gap/2,dt/3)
            extended_t.extend([t[j]+half,t[j+1]-half]);source_rows.extend([-1,-1])
        extended_t.append(t[j+1]);source_rows.append(j+1)
    extended_t=np.asarray(extended_t)

    fig,axes=plt.subplots(5,1,sharex=True,figsize=settings.get('figsize',(13.5,13.7)),
                          gridspec_kw={'hspace':.27})
    fig.subplots_adjust(left=.095,right=.855,top=.84,bottom=.13)
    fig.text(.095,.94,'FIGURE 3  /  Seasonal evolution of the water column',
             fontsize=16,fontweight='bold',color=INK)
    fig.text(.095,.915,'BGC-Argo observations · profiles in chronological order',
             fontsize=10,color=MUTED)

    def scale(field,step,allow_negative):
        finite=field[np.isfinite(field)]
        if not len(finite):raise ValueError('No valid data for colour scale')
        lo,hi=np.percentile(finite,[settings.get('Percentiles_lower',1),
                                   settings.get('Percentiles_upper',99)])
        lo=np.floor(lo/step)*step;hi=np.ceil(hi/step)*step
        if not allow_negative:lo=max(0,lo)
        if hi<=lo:hi=lo+step
        while (hi-lo)/step>8:
            step*=2;lo=np.floor(lo/step)*step;hi=np.ceil(hi/step)*step
            if not allow_negative:lo=max(0,lo)
        return lo,hi,np.arange(lo,hi+step*.5,step)

    for ax,(name,title,units,nonnegative,decimals) in zip(axes,specs):
        field=fields[name]
        lo,hi,ticks=scale(field,COLOURBAR_STEPS['DOXY' if name=='DISS_OXY' else name],
                           not nonnegative)
        expanded=np.full((len(source_rows),len(depth)),np.nan)
        for pos,idx in enumerate(source_rows):
            if idx>=0:expanded[pos]=np.clip(field[idx],lo,hi)
        im=ax.pcolormesh(extended_t,depth,np.ma.masked_invalid(expanded.T),
                         cmap=CMAPS['DOXY' if name=='DISS_OXY' else name],
                         vmin=lo,vmax=hi,shading='nearest',rasterized=True)
        ax.set_ylim(depth_max,0);ax.set_ylabel('Pressure (dbar)',labelpad=8)
        ax.set_title(title,loc='left',pad=8,fontsize=12)
        ax.tick_params(length=0,pad=6)
        ax.spines[['top','right','bottom']].set_visible(False)
        cb=fig.colorbar(im,ax=ax,pad=.018,fraction=.026,ticks=ticks)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0,labelsize=9)
        cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f'%.{decimals}f'))
        cb.set_label(units,fontsize=9,labelpad=8,color=MUTED)
        ax.set_xlim(t[0],t[-1])
        for year in pd.date_range(f'{pd.Timestamp(dates[0]).year}-01-01',
                                  f'{pd.Timestamp(dates[-1]).year+1}-01-01',freq='YS'):
            ax.axvline(mdates.date2num(year),color='white',alpha=.78,linewidth=.9,zorder=5)
    axes[3].plot(t,mld,color='white',linewidth=1.35,linestyle='--',
                 label='Mixed-layer depth',zorder=6)
    axes[3].legend(loc='lower right',fontsize=9,framealpha=.85,facecolor='white',edgecolor='none')
    
    # Show monthly labels for records shorter than three years.
    # Longer records retain annual labels to avoid overcrowding.
    ax_time = axes[-1]
    start = pd.Timestamp(dates[0])
    end = pd.Timestamp(dates[-1])

    if end < start + pd.DateOffset(years=3):

        # First letter of each month along the main time axis
        ax_time.xaxis.set_major_locator(mdates.MonthLocator())
        ax_time.xaxis.set_major_formatter(
            mticker.FuncFormatter(
                lambda x, pos: mdates.num2date(x).strftime('%b')[0]
            )
        )
        ax_time.tick_params(axis='x', labelsize=9, length=3, pad=5)

        # Show each year centred beneath its visible months
        year_positions = []
        year_labels = []

        for year in range(start.year, end.year + 1):
            first = max(start, pd.Timestamp(year, 1, 1))
            last = min(end, pd.Timestamp(year + 1, 1, 1))

            if last > first:
                midpoint = first + (last - first) / 2
                year_positions.append(mdates.date2num(midpoint))
                year_labels.append(str(year))

        year_axis = ax_time.secondary_xaxis('bottom')
        year_axis.spines['bottom'].set_position(('outward', 22))
        year_axis.spines['bottom'].set_visible(False)
        year_axis.set_xticks(year_positions, labels=year_labels)
        year_axis.tick_params(length=0, labelsize=10, pad=3)

        ax_time.set_xlabel('')

    else:
        ax_time.xaxis.set_major_locator(mdates.YearLocator())
        ax_time.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax_time.set_xlabel('Year', labelpad=10)


    # Regional location inset and a slim matching time reference strip.
    lon=_as_float(data['LONGITUDE'])[order];lat=_as_float(data['LATITUDE'])[order]
    valid=np.isfinite(lon)&np.isfinite(lat)
    if valid.any():
        proj,central=_projection(lon,data['FLOAT_NUMBER']);x=((lon-central+180)%360)-180
        inset=fig.add_axes([.69,.863,.16,.115],projection=proj)
        _map_extent(inset,x[valid],lat[valid],.25,5);_base_map(inset,.5)
        _track(inset,x,lat,t,CMAPS['LOCATION'],size=11)
        good=np.flatnonzero(valid)
        inset.plot(x[good[0]],lat[good[0]],'o',markersize=5,color='#25A77A',
                   markeredgecolor='white',transform=proj,zorder=6)
        inset.plot(x[good[-1]],lat[good[-1]],'X',markersize=6,color=ACCENT,
                   markeredgecolor='white',transform=proj,zorder=6)
        inset.set_title('Float location',fontsize=9,pad=4)
    pos=axes[-1].get_position()
    bar=fig.add_axes([pos.x0,.065,pos.width,.011])
    bar.imshow(np.linspace(t[0],t[-1],512)[None,:],extent=[t[0],t[-1],0,1],
               cmap=CMAPS['LOCATION'],aspect='auto',interpolation='nearest')
    bar.set_xticks([]);bar.set_yticks([])
    for spine in bar.spines.values():spine.set_visible(False)
    bar.set_title('Colour = float position through time',fontsize=9,pad=6,color=MUTED)
    plt.show()




def calculate_density_mld(
    pressure, temperature, salinity, longitude, latitude,
    threshold=0.03, reference=10,
    median_window=5, persistence=10, max_gap=20
):
    """
    Calculate TEOS-10 potential density anomaly and mixed-layer depth.

    MLD is defined as the first depth where potential density exceeds
    the density at the reference pressure by 'threshold' kg/m3,
    remaining above this threshold for at least 'persistence' dbar.

    A median filter reduces sensitivity to isolated density spikes.
    Interpolation does not bridge gaps larger than 'max_gap'.

    Returns
    -------
    density : 2D array of potential density anomaly (sigma0)
    mld     : 1D array of mixed-layer depths (dbar)
    """
    import numpy as np
    import gsw
    from scipy.ndimage import median_filter

    def as_float(x):
        return np.asarray(np.ma.filled(x, np.nan), dtype=float)

    p = as_float(pressure)
    t = as_float(temperature)
    sp = as_float(salinity)
    lon = as_float(longitude).ravel()
    lat = as_float(latitude).ravel()

    if p.ndim != 2 or t.shape != p.shape or sp.shape != p.shape:
        raise ValueError(
            "Pressure, temperature and salinity must be matching 2D arrays."
        )

    if lon.size != len(p) or lat.size != len(p):
        raise ValueError("Longitude and latitude must match profile count.")

    if median_window < 1 or median_window % 2 != 1:
        raise ValueError("median_window must be a positive odd integer.")

    if persistence < 1 or int(persistence) != persistence:
        raise ValueError("persistence must be a positive integer.")

    if threshold <= 0 or max_gap <= 0 or reference < 0:
        raise ValueError("Invalid MLD parameters.")

    # -----------------------------------------------------
    # CALCULATE TEOS-10 POTENTIAL DENSITY ANOMALY
    # -----------------------------------------------------

    density = np.full(p.shape, np.nan)

    valid = (
        np.isfinite(p) &
        np.isfinite(t) &
        np.isfinite(sp) &
        (sp >= 0) &
        np.isfinite(lon[:, None]) &
        np.isfinite(lat[:, None])
    )

    if np.any(valid):
        lon2 = np.broadcast_to(lon[:, None], p.shape)
        lat2 = np.broadcast_to(lat[:, None], p.shape)

        SA = gsw.SA_from_SP(
            sp[valid], p[valid],
            lon2[valid], lat2[valid]
        )

        CT = gsw.CT_from_t(SA, t[valid], p[valid])

        density[valid] = gsw.sigma0(SA, CT)

    # -----------------------------------------------------
    # CALCULATE MLD FOR EACH PROFILE
    # -----------------------------------------------------

    mld = np.full(p.shape[0], np.nan)

    for i in range(p.shape[0]):

        ok = np.isfinite(p[i]) & np.isfinite(density[i])

        if np.sum(ok) < 2:
            continue

        pi = p[i, ok]
        rho = density[i, ok]

        # Sort pressure and remove duplicate levels
        idx = np.argsort(pi, kind='stable')
        pi, rho = pi[idx], rho[idx]

        pi, unique = np.unique(pi, return_index=True)
        rho = rho[unique]

        if not (pi[0] <= reference < pi[-1]):
            continue

        # -------------------------------------------------
        # INTERPOLATE ONTO A 1 DBAR GRID
        # -------------------------------------------------

        grid = np.arange(0, np.floor(pi[-1]) + 1, 1.0)
        rho_grid = np.full(grid.shape, np.nan)

        for j in range(len(pi) - 1):

            if pi[j+1] - pi[j] > max_gap:
                continue

            section = (
                (grid >= pi[j]) &
                (grid <= pi[j+1])
            )

            rho_grid[section] = np.interp(
                grid[section],
                pi[j:j+2],
                rho[j:j+2]
            )

        # -------------------------------------------------
        # MEDIAN FILTER WITHOUT BRIDGING MISSING DATA
        # -------------------------------------------------

        rho_smooth = np.full_like(rho_grid, np.nan)

        valid_grid = np.isfinite(rho_grid)

        edges = np.diff(
            np.r_[False, valid_grid, False].astype(int)
        )

        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]

        for start, end in zip(starts, ends):
            rho_smooth[start:end] = median_filter(
                rho_grid[start:end],
                size=median_window,
                mode='nearest'
            )

        # -------------------------------------------------
        # REFERENCE DENSITY
        # -------------------------------------------------

        r0 = int(np.floor(reference))
        r1 = int(np.ceil(reference))

        if r1 >= len(grid):
            continue

        if not np.all(np.isfinite(rho_smooth[r0:r1+1])):
            continue

        rho_ref = np.interp(reference, grid, rho_smooth)

        # Fixed density threshold: 0.03 kg/m3 by default
        target = rho_ref + threshold

        # -------------------------------------------------
        # FIND FIRST PERSISTENT THRESHOLD CROSSING
        # -------------------------------------------------

        first = int(np.floor(reference)) + 1
        n = int(persistence)

        for j in range(first, len(grid) - n):

            section = rho_smooth[j:j+n+1]

            # Require continuous valid data
            if not np.all(np.isfinite(section)):
                continue

            # Density must exceed threshold for 10 dbar
            if not np.all(section >= target):
                continue

            # Require crossing from below
            if not np.isfinite(rho_smooth[j-1]):
                continue

            if rho_smooth[j-1] >= target:
                continue

            # Interpolate exact crossing pressure
            mld[i] = np.interp(
                target,
                [rho_smooth[j-1], rho_smooth[j]],
                [grid[j-1], grid[j]]
            )

            break

    return density, mld



def interpolate_original_profiles(data, max_vertical_gap=20):
    """Interpolate onto existing pressure levels, respecting joint data masks and gaps.

    No extrapolation and no filling across gaps larger than max_vertical_gap dbar.
    Uses the same vertical interpolation rules as Figures 3 and 4.
    Returns TEMP_INT, PSAL_INT, CHLA_INT, BBP_INT, DISS_OXY_INT.
    """
    pressure = _as_float(data['PRESSURE'])
    # Apply shared vertical interpolation profile by profile at the ORIGINAL levels.
    # This retains the original figure's resolution, without filling unobserved levels.
    def interpolate_at_levels(values, nonnegative=False):
        v = _as_float(values)
        out = np.full(pressure.shape, np.nan)
        for i in range(len(pressure)):
            row = _vertical_grid(v[i:i+1], pressure[i:i+1],
                                 pressure[i, np.isfinite(pressure[i])],
                                 max_vertical_gap, nonnegative=nonnegative)
            out[i, np.isfinite(pressure[i])] = row[0]
        return out
    temp = interpolate_at_levels(data['TEMP'])
    psal = interpolate_at_levels(data['PSAL'], nonnegative=True)
    chla = interpolate_at_levels(data['CHLA'], nonnegative=True)
    bbp = interpolate_at_levels(data['BBP'], nonnegative=True)
    doxy = interpolate_at_levels(data['DISS_OXY'], nonnegative=True)
    # Preserve the original median smoothing of sufficiently long BBP profiles.
    from scipy.signal import medfilt
    for i in range(len(bbp)):
        good = np.flatnonzero(np.isfinite(bbp[i]))
        # Filter only contiguous valid sections to avoid bridging missing-data gaps.
        for segment in np.split(good, np.where(np.diff(good)>1)[0]+1):
            if len(segment) > 11:
                bbp[i,segment] = medfilt(bbp[i,segment], kernel_size=11)
    temp[temp >= 50] = np.nan
    return temp, psal, chla, bbp, doxy



def plot_monthly_climatology(data, settings):
    """
    Figure 4: Monthly climatology of the upper ocean.

    Each profile is averaged over the upper 100 dbar (default).
    Monthly means and standard deviations are then calculated
    across all available profile averages.

    Missing measurements are excluded, and no extrapolation
    beyond observed depths is performed.
    """
    _style()

    times = pd.to_datetime(np.asarray(data['TIME_BGC']))
    months = times.month
    pressure = _as_float(data['PRESSURE'])
    mld = _as_float(data['MLD'])

    # Student-adjustable parameters
    averaging_depth = settings.get('AVERAGING_DEPTH', 100)
    step = settings.get('DEPTH_STEP', 2)
    max_gap = settings.get('MAX_VERTICAL_GAP', 20)

    if averaging_depth <= 0 or step <= 0 or max_gap <= 0:
        raise ValueError("Depth and interpolation settings must be positive.")

    if (pressure.ndim != 2 or pressure.shape[0] != len(mld)
            or len(times) != len(mld)):
        raise ValueError("Pressure, MLD and dates must have matching profiles.")

    # Common depth grid for averaging the upper ocean
    depth = np.arange(0, averaging_depth, step)
    depth = np.append(depth, averaging_depth)

    variables = [
        ('TEMP', 'Temperature', '°C', False),
        ('PSAL', 'Salinity', 'PSU', True),
        ('DISS_OXY', 'Dissolved oxygen', 'µmol kg$^{-1}$', True),
        ('CHLA', 'Chlorophyll-a', 'mg m$^{-3}$', True),
        ('BBP', 'Backscattering', 'm$^{-1}$', True)
    ]

    # Monthly mean and standard deviation of profile averages
    def monthly_stats(values):
        means = np.full(12, np.nan)
        stds = np.full(12, np.nan)

        for i, month in enumerate(range(1, 13)):
            sample = values[
                (months == month) & np.isfinite(values)
            ]
            if sample.size:
                means[i] = np.mean(sample)
                stds[i] = np.std(sample)

        return means, stds

    # Create figure
    fig, axes = plt.subplots(
        6, 1, sharex=True,
        figsize=settings.get('figsize', (11, 14.2)),
        gridspec_kw={'hspace': 0.35}
    )

    fig.subplots_adjust(
        left=0.16, right=0.94,
        top=0.90, bottom=0.085
    )

    fig.suptitle(
        'FIGURE 4  /  Monthly climatology',
        x=0.16, y=0.975, ha='left',
        fontsize=16, fontweight='bold', color=INK
    )

    fig.text(
        0.16, 0.942,
        f'Mean ± 1 SD · upper {averaging_depth:g} dbar · all available years',
        fontsize=10, color=MUTED
    )

    mon = np.arange(1, 13)
    labels = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]

    # Plot the five biogeochemical variables
    for ax, (name, title, units, nonnegative) in zip(
        axes[:5], variables
    ):

        # Interpolate each profile onto the upper-ocean grid
        grid = _vertical_grid(
            data[name], pressure, depth,
            max_gap, nonnegative
        )

        # Average available values over the upper 100 dbar
        counts = np.isfinite(grid).sum(axis=1)
        averages = np.full(len(mld), np.nan)
        usable = counts > 0

        averages[usable] = (
            np.nansum(grid[usable], axis=1) / counts[usable]
        )

        # Monthly climatology of profile averages
        means, stds = monthly_stats(averages)

        key = 'DOXY' if name == 'DISS_OXY' else name

        ax.errorbar(
            mon, means, yerr=stds,
            color=COLOURS[key],
            marker='o', markersize=5,
            linestyle='-', linewidth=1.6,
            capsize=3, capthick=1, elinewidth=1,
            markerfacecolor='white',
            markeredgewidth=1.8
        )

        ax.set_title(
            title, loc='left', fontsize=11,
            pad=7, color=COLOURS[key]
        )
        ax.set_ylabel(units, labelpad=8)
        _tidy(ax)

    # Final panel: monthly mixed-layer depth
    valid_mld = np.isfinite(mld) & (mld > 0)
    mld_values = np.where(valid_mld, mld, np.nan)
    means, stds = monthly_stats(mld_values)

    ax = axes[-1]

    ax.errorbar(
        mon, means, yerr=stds,
        color=COLOURS['MLD'],
        marker='o', markersize=5,
        linestyle='-', linewidth=1.6,
        capsize=3, elinewidth=1,
        markerfacecolor='white',
        markeredgewidth=1.8
    )

    ax.set_title(
        'Mixed-layer depth', loc='left',
        fontsize=11, pad=7, color=COLOURS['MLD']
    )
    ax.set_ylabel('dbar', labelpad=8)
    ax.invert_yaxis()
    _tidy(ax)

    ax.set_xticks(mon)
    ax.set_xticklabels(labels)
    ax.set_xlim(0.75, 12.25)
    ax.set_xlabel('Calendar month', labelpad=10)

    plt.show()


