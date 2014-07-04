#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
Extract tracer profiles and compute total columns
above a given station from GEOS-Chem outputs 

This file should be executed from the main script
'extract_gcprof_station.py' and also on all engines
of any used IPython parallel cluster.

"""

import os
import sys
import time

import numpy as np
import pandas as pd
import iris
from iris.analysis.interpolate import extract_nearest_neighbour
import iris.pandas as ipandas
import yaml
from six import string_types

from pygchem.utils import iris_tools


# ---------------------------------------
# Functions to load and process input
# parameters and/or arguments
# ---------------------------------------

def load_params(infile):
    """
    Load the parameters and arguments given in
    `input_file` (YAML format).
    """
    with open(infile) as f:
        params = yaml.load(f)
    
    return params


def parse_in_files(in_dir, in_files):
    """
    Given the `in_dir` and `in_files` parameters,
    return a list of absolute paths to individual files.
    
    `in_files` may be a filename in `in_dir`, or an
    absolute path to a file, or a relative/absolute
    file-matching pattern, or an iterable with any
    combination of these.
    """
    if isinstance(in_files, string_types):
        in_files = [in_files]
    
    abspaths = []
    for f in in_files:
        if not os.path.isabs(f):
            f = os.path.join(in_dir, f)
        abspaths.extend(iglob(f))
    
    return abspaths


# ---------------------------------------
# Preprocessing functions
# ---------------------------------------

def preprocessing(params):
    """
    Preprocessing.
    """
    global global_topography, station_vgrid
    
    global_topography = iris.load_cube(
                            params['global_topography_datafile']
                        )
    station_vgrid = iris.load_cube(
                        params['station_vertical_grid_file']
                    )


# ---------------------------------------
# Sub-routines needed for processing
# ---------------------------------------

def regrid_profile(profile_cube, station_profile):
    """
    Vertical regridding of one profile
    (slicing over time, regrid the slices, and re-merge).
    
    """
    rprof_timeslices = []
    
    for prof_timeslice in profile_cube.slices('model_level_number'):
        rprof = iris_tools.regrid_conservative_vertical(
                    prof_timeslice, station_profile
                )
        rprof_timeslices.append(rprof)
    
    return iris.cube.CubeList(rprof_timeslices).merge()[0]


# ---------------------------------------
# Main processing functions
# ---------------------------------------

def processing(in_files, params):
    """
    Main processing function to execute either on the main
    thread or on IPython parallel engines.
    """
    
    #print in_files
    #print '------------'
    #print params
    
    # LOAD DATA
    #global i_coord, j_coord, l_coord
    station_region_indices = (params['iminmax'] + 
                              params['jminmax'] + 
                              params['lminmax'])
    i_coord, j_coord, l_coord = iris_tools.gcgrid_2_coords(
        params['grid_model_name'],
        params['grid_model_resolution'],
        region_box=station_region_indices
    )
    
    def assign_coord_nd49_or_subset_ctm(cube, field, filename):
        """
        A callback for GEOS-Chem datafields loading with Iris.

        If `cube` is loaded from a ND49 diagnostic file
        (i.e., some undefined dimension coordinates), generate
        the coordinates values from the grid indices of the 3D
        region box.
        (Else) If `cube` is loaded from a CTM file, extract
        a subset that correspond to the region box.
        """
        #global i_coord, j_coord, l_coord

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
    
    tracers2load = iris.AttributeConstraint(
        category=lambda category: category in params['categories'],
        name=lambda name: name in params['tracers']
    )

    all_cubes = iris.load(in_files,
                      [tracers2load] + params['other_fields'],
                      callback=assign_coord_nd49_or_subset_ctm)

    tracer_cubes = all_cubes.extract(tracers2load)
    other_cubes = all_cubes.extract(params['other_fields'], strict=False)
    
    # datafields required for columns and profiles calculation
    pedges_cube = other_cubes.extract_strict('PSURF_PEDGE-$')
    box_height_cube = other_cubes.extract_strict('BXHEIGHT_BXHGHT-$')
    try:
        n_air_cube = other_cubes.extract_strict('N(AIR)_BXHGHT-$')
    except iris.exceptions.ConstraintMismatchError:
        # ND49: air density datafields have a different name
        n_air_cube = other_cubes.extract_strict('AIRDEN_TIME-SER')
        # ND49: fix missing pressure level
        #pedges_cube = iris_tools.fix_nd49_pressure(pedges_cube)

    # convert units for hydrocarbon tracers 
    for cube in tracer_cubes:
        iris_tools.ppbC_2_ppbv(cube)
    
    #print tracer_cubes
    #print "------------------------"
    #print tracer_cubes[0]
    #print "------------------------"
    #print other_cubes
    
    # EXTRACT AND REGRID PROFILES
    station_coords = [('latitude', params['station_lat']),
                      ('longitude', params['station_lon'])]

    tracer_profiles = iris.cube.CubeList(
        [extract_nearest_neighbour(cube, station_coords)
         for cube in tracer_cubes]
    )
    
    pedges_profile, box_height_profile, n_air_profile = [
        extract_nearest_neighbour(cube, station_coords)
        for cube in [pedges_cube, box_height_cube, n_air_cube]
    ]

    all_profiles = tracer_profiles + \
                   [pedges_profile, box_height_profile, n_air_profile]

    #global_topography = iris.load_cube(params['global_topography_datafile'])
    
    bh = box_height_profile.copy()
    gt = global_topography.copy()
    
    altitude_coord = iris_tools.get_altitude_coord(bh, gt)
    
    #print altitude_coord
    #return
    
    for cube in tracer_profiles + [n_air_profile]:
        if cube.coords('air_pressure'):
            cube.remove_coord('air_pressure')
        if cube.coords('altitude'):
            cube.remove_coord('altitude')
        cube.add_aux_coord(altitude_coord,
                           data_dims=range(0, box_height_profile.ndim))
    
    station_profile = extract_nearest_neighbour(
        #iris.load_cube(params['station_vertical_grid_file']),
        station_vgrid,
        station_coords
    )

    regridded_tracer_profiles = iris.cube.CubeList(
        [regrid_profile(p, station_profile)
        for p in tracer_profiles]
    )
    regridded_n_air_profile = regrid_profile(n_air_profile, station_profile)
    regridded_profiles = regridded_tracer_profiles + \
                             [regridded_n_air_profile]
    
    # CALCULATE COLUMNS
    columns = [iris_tools.compute_tracer_columns(
                   p, regridded_n_air_profile, 'altitude'
               )
               for p in regridded_tracer_profiles]

    tracer_columns = iris.cube.CubeList(columns)
    
    return regridded_profiles, tracer_columns


def merge_results(res):
    """
    Merge results from parallel processing.
    """
    profs, cols = zip(*res)
        
    regridded_profiles = iris.cube.CubeList()
    for ps in zip(*profs):
        cubelist = iris.cube.CubeList(ps)
        regridded_profiles.append(cubelist.concatenate()[0])

    tracer_columns = iris.cube.CubeList()
    for cs in zip(*cols):
        cubelist = iris.cube.CubeList(cs)
        tracer_columns.append(cubelist.concatenate()[0])
    
    return regridded_profiles, tracer_columns


def save_results(regridded_profiles, tracer_columns, params):
    """
    Save (merged) results.
    """
    
    # SET OUTPUT FILE NAMES
    out_profiles_basename = string.replace(
        params['out_profiles_basename'], "*", params['station_name']
    )
    out_columns_basename = string.replace(
        params['out_columns_basename'], "*", params['station_name'])

    if params['out_dir'] is None:
        params['out_dir'] = os.path.abspath(os.getcwd())
    out_profiles_basepath = os.path.join(params['out_dir'],
                                         out_profiles_basename)
    out_columns_basepath = os.path.join(params['out_dir'],
                                        out_columns_basename)
    
    out_profiles_cubes_fn = out_profiles_basepath + '.nc'
    out_columns_cubes_fn = out_columns_basepath + '.nc'
    
    out_profiles_tables_fn = '.'.join([out_profiles_basepath,
                                       params['out_format']])
    out_columns_tables_fn = '.'.join([out_columns_basepath,
                                      params['out_format']])                                   
    

    # SAVE IN NETCDF FILES (CUBES)
    # make altitude as dimension coord
    for profile_cube in regridded_profiles:
        if isinstance(profile_cube.coord('altitude'), iris.coords.AuxCoord):
            iris_tools.permute_dim_aux_coords(profile_cube,
                                              'model_level_number',
                                              'altitude')

    for cube in regridded_profiles:
        iris_tools.fix_cube_attributes_vals(cube)

    for cube in tracer_columns:
        iris_tools.fix_cube_attributes_vals(cube)

    iris.save(regridded_profiles, out_profiles_cubes_fn)
    iris.save(tracer_columns, out_columns_cubes_fn)
    
    # SAVE AS DATA TABLES (EXCEL, CSV...)
    dataframe_profiles = {}
    for profile in regridded_profiles:
        profile_cube = profile.copy()

        # there must be only one defined dimension coordinate for each
        # cube dimension (no auxilliary coordinate (convert iris to pandas)
        z_dim = profile_cube.coord_dims(profile_cube.coord(name='altitude'))
        iris_tools.remove_dim_aux_coords(profile_cube, z_dim)

        dataframe_units = str(profile_cube.units).replace('/', '_')
        if dataframe_units == 'unknown':
            dataframe_units = profile_cube.attributes['no_udunits2']
        dataframe_name = "{tracer}_{units}".format(
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

    panel_profiles = pd.Panel(dataframe_profiles).astype(np.float64)

    series_columns = {}
    for column in tracer_columns:
        series_name = "{tracer}_{units}".format(
            tracer=column.attributes['name'],
            units='molec_cm-2'
        )
        series_columns[series_name] = ipandas.as_series(column)
        time_coord = column.coord('time')  
        date = time_coord.units.num2date(time_coord.points)
        series_columns[series_name].index = date

    dataframe_columns = pd.DataFrame(series_columns).astype(np.float64)

    if params['out_format'] in ('hdf', 'hdf5'):
        panel_profiles.to_hdf(out_profiles_tables_fn, 'profiles')
        dataframe_columns.to_hdf(out_columns_tables_fn, 'columns')
    
    elif params['out_format'] in ('xls', 'xlsx'):
        panel_profiles.to_excel(out_profiles_tables_fn, 'profiles')
        dataframe_columns.to_excel(out_columns_tables_fn, 'columns')

    elif params['out_format'] == 'csv':
        for pr in panel_profiles:
            panel_profiles[pr].to_csv('{0}_{1}.csv'
                                      .format(out_profiles_basepath, pr))
        dataframe_columns.to_csv(out_columns_tables_fn)
    
    return ([panel_profiles, dataframe_columns],
            [out_profiles_cubes_fn, out_columns_cubes_fn,
             out_profiles_tables_fn, out_columns_tables_fn])