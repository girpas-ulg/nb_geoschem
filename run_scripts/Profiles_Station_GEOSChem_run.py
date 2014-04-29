
# coding: utf-8

"""
Extract tracer profiles and compute total columns above a given station from GEOS-Chem outputs 
==============================================================================================

Script that should be run only from the '_run' corresponding IPython notebooks.

"""

import os
import string

import numpy as np
import pandas as pd
import plotly
import matplotlib.pyplot as plt

import iris
import iris.pandas as ipandas
import iris.quickplot as iqplot

import pygchem.grid as gcgrid
from pygchem.utils import iris_tools


in_abspaths = [os.path.join(in_dir, fname) if not os.path.isabs(fname) else fname
               for fname in in_files]


# 4. Load data
# ------------ 
print "Loading data..." 

def assign_coord_nd49_or_subset_ctm(cube, field, filename):
    """
    A callback for GEOS-Chem datafields loading with Iris.
    
    If `cube` is loaded from a ND49 diagnostic file (i.e., some
    undefined dimension coordinates), generate the coordinates values
    from the grid indices of the 3D region box.
    (Else) If `cube` is loaded from a CTM file, extract a subset
    that correspond to the region box.
    """
    global i_coord, j_coord, l_coord
    
    cube_is_ctm = True
    
    if not cube.coords('longitude'):
        cube.add_dim_coord(i_coord, 0)
        cube_is_ctm = False
    if not cube.coords('latitude'):
        cube.add_dim_coord(j_coord, 1)
        cube_is_ctm = False
    if not cube.coords('model_level_number'):
        cube.add_dim_coord(l_coord, 2)
    
    if cube_is_ctm:
        lonlat_subset = iris.Constraint(longitude=i_coord.points,
                                        latitude=j_coord.points)
        cube = cube.extract(lonlat_subset)


station_region_indices = (imin, imax, jmin, jmax, lmin, lmax)
i_coord, j_coord, l_coord = iris_tools.gcgrid_2_coords(grid_model_name,
                                                       grid_model_resolution,
                                                       region_box=station_region_indices)

tracers2load = iris.AttributeConstraint(category=lambda category: category in categories,
                                        name=lambda name: name in tracers)

all_cubes = iris.load(in_abspaths,
                      [tracers2load] + other_fields,
                      callback=assign_coord_nd49_or_subset_ctm)

tracer_cubes = all_cubes.extract(tracers2load)
other_cubes = all_cubes.extract(other_fields, strict=False)

# datafields required for columns and profiles calculation
pedges_cube = other_cubes.extract_strict('PSURF_PEDGE-$')
box_height_cube = other_cubes.extract_strict('BXHEIGHT_BXHGHT-$')
try:
    n_air_cube = other_cubes.extract_strict('N(AIR)_BXHGHT-$')
except iris.exceptions.ConstraintMismatchError:
    # ND49: air density datafields have a different name
    n_air_cube = other_cubes.extract_strict('AIRDEN_TIME-SER')
    # ND49: fix missing pressure level
    pedges_cube = iris_tools.fix_nd49_pressure(pedges_cube)

# convert units for hydrocarbon tracers 
for cube in tracer_cubes:
    iris_tools.ppbC_2_ppbv(cube)


# 5. Extract and regrid profiles above the station
# ------------------------------------------------
print "Extract profiles (nearest neighbour)..." 

extract_method = iris.analysis.interpolate.extract_nearest_neighbour

station_coords = [('latitude', station_lat), ('longitude', station_lon)]

tracer_profiles = iris.cube.CubeList(extract_method(cube, station_coords)
                                     for cube in tracer_cubes)

pedges_profile, box_height_profile, n_air_profile = [
    extract_method(cube, station_coords)
    for cube in [pedges_cube, box_height_cube, n_air_cube]
]

all_profiles = tracer_profiles + [pedges_profile, box_height_profile, n_air_profile]

global_topography = iris.load_cube(global_topography_datafile)

altitude_coord = iris_tools.get_altitude_coord(box_height_profile,
                                               global_topography)

for cube in tracer_profiles + [n_air_profile]:
    if cube.coords('air_pressure'):
        cube.remove_coord('air_pressure')
    if cube.coords('altitude'):
        cube.remove_coord('altitude')
    cube.add_aux_coord(altitude_coord,
                       data_dims=range(0, box_height_profile.ndim))

print "Regridding the profiles..."

station_profile = extract_method(iris.load_cube(station_vertical_grid_file),
                                 station_coords)

def regrid_profile(profile_cube, station_profile):
    """
    Vertical regridding of one profile
    (slicing over time, regrid the slices, and re-merge).
    
    """
    rprof_timeslices = []
    
    for prof_timeslice in profile_cube.slices('model_level_number'):
        rprof = iris_tools.regrid_conservative_vertical(prof_timeslice, station_profile)
        rprof_timeslices.append(rprof)
    
    return iris.cube.CubeList(rprof_timeslices).merge()[0]
    

regridded_tracer_profiles = iris.cube.CubeList([regrid_profile(p, station_profile)
                                                for p in tracer_profiles])

regridded_n_air_profile = regrid_profile(n_air_profile, station_profile)

regridded_all_profiles = regridded_tracer_profiles + [regridded_n_air_profile]

# 6. Compute total columns above the station
# ------------------------------------------
print "Compute total columns..."

columns = [iris_tools.compute_tracer_columns(p,
                                             regridded_n_air_profile,
                                             'altitude')
           for p in regridded_tracer_profiles]

tracer_columns = iris.cube.CubeList(columns)

# 6. Save extracted profiles and columns to disk
# ----------------------------------------------
print "Save profiles and columns to disk..."

out_profiles_basename = string.replace(out_profiles_basename, "*", station_name)
out_columns_basename = string.replace(out_columns_basename, "*", station_name)

out_profiles_basepath = os.path.join(out_dir, out_profiles_basename)
out_columns_basepath = os.path.join(out_dir, out_columns_basename)

# make altitude as dimension coord
for profile_cube in regridded_tracer_profiles + [regridded_n_air_profile]:
    if isinstance(profile_cube.coord('altitude'), iris.coords.AuxCoord):
        iris_tools.permute_dim_aux_coords(profile_cube,
                                          'model_level_number',
                                          'altitude')

for cube in regridded_tracer_profiles + [regridded_n_air_profile]:
    iris_tools.fix_cube_attributes_vals(cube)

for cube in tracer_columns:
    iris_tools.fix_cube_attributes_vals(cube)


iris.save(regridded_tracer_profiles + [regridded_n_air_profile],
          '.'.join([out_profiles_basepath, 'nc']))
print "\t Written {}".format('.'.join([out_profiles_basepath, 'nc']))
iris.save(tracer_columns,
          '.'.join([out_columns_basepath, 'nc']))
print "\t Written {}".format('.'.join([out_columns_basepath, 'nc']))


dataframe_profiles = {}
for profile in regridded_tracer_profiles + [regridded_n_air_profile]:
    profile_cube = profile.copy()
    
    # there must be only one defined dimension coordinate for each
    # cube dimension (no auxilliary coordinate (convert iris to pandas)
    z_dim = profile_cube.coord_dims(profile_cube.coord(name='altitude'))
    iris_tools.remove_dim_aux_coords(profile_cube, z_dim)
    
    dataframe_units = profile_cube.units.format()
    if dataframe_units == 'unknown':
        dataframe_units = profile_cube.attributes['no_udunits2']
    dataframe_name = "{tracer} ({units})".format(
        tracer=profile_cube.attributes['name'],
        units=dataframe_units
    )
    
    # scalar time coordinate
    if profile_cube.ndim == 1:
        series = ipandas.as_series(profile)
        time_coord = profile_cube.coord('time')  
        date = time_coord.units.num2date(time_coord.points[0])
        dataframe_profiles[dataframe_name] = pd.DataFrame({date : series}).transpose()
    # dimensional time coordinate
    else:
        dataframe_profiles[dataframe_name] = ipandas.as_data_frame(profile_cube)
    
panel_profiles = pd.Panel(dataframe_profiles).transpose(0, 2, 1).astype(np.float64)


series_columns = {}
for column in tracer_columns:
    series_name = "{tracer} ({units})".format(
        tracer=column.attributes['name'],
        units='molec cm-2'
    )
    series_columns[series_name] = ipandas.as_series(column)
    time_coord = column.coord('time')  
    date = time_coord.units.num2date(time_coord.points)
    series_columns[series_name].index = date

dataframe_columns = pd.DataFrame(series_columns).astype(np.float64)


if out_format in ('hdf', 'hdf5'):
    panel_profiles.to_hdf('.'.join([out_profiles_basepath, 'hdf5']),
                          'profiles')
    print "\t Written {}".format('.'.join([out_profiles_basepath, 'hdf5']))
    dataframe_columns.to_hdf('.'.join([out_columns_basepath, 'hdf5']),
                            'columns')
    print "\t Written {}".format('.'.join([out_columns_basepath, 'hdf5']))

elif out_format in ('xls', 'xlsx'):
    panel_profiles.to_excel('.'.join([out_profiles_basepath, out_format]),
                            'profiles')
    print "\t Written {}".format('.'.join([out_profiles_basepath, out_format]))
    dataframe_columns.to_excel('.'.join([out_columns_basepath, out_format]),
                              'columns')
    print "\t Written {}".format('.'.join([out_columns_basepath, out_format]))

elif out_format == 'csv':
    panel_profiles.transpose(2, 0, 1).to_frame().to_csv(
        '.'.join([out_profiles_basepath, 'csv'])
    )
    print "\t Written {}".format('.'.join([out_profiles_basepath, 'csv']))
    dataframe_columns.to_csv('.'.join([out_columns_basepath, out_format]))
    print "\t Written {}".format('.'.join([out_columns_basepath, out_format]))

