'''
Created on Oct 29, 2013

@author: pengshi

This class contains the locations of the most important files and folders.

'''

import os


class Folders(object):
    BASE=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir,os.path.pardir)),'data')
    
    PLANS=os.path.join(BASE,'assignment_plans')
    LOGS=os.path.join(BASE,'logs')
    INPUTS=os.path.join(BASE,'klm_raw_inputs')
    RESULTS=os.path.join(BASE,'results')
          
    
class PlanFolders(object):
    
    SIMULATIONS=os.path.join(Folders.LOGS,'simulations')
    PRIORITIES=os.path.join(Folders.PLANS,'large_market_plans')
    PARAMS=os.path.join(Folders.PLANS,'plans_for_simulation')
    RESULT_TABLES=os.path.join(Folders.RESULTS,'tables')
    RESULT_FIGURES=os.path.join(Folders.RESULTS,'figures')
    RESULT_OTHER=os.path.join(Folders.RESULTS,'other')

    
class PlanFiles(object):
    # FIXED_EFFECT=os.path.join(Folders.INPUTS,'school_characteristics.csv')
    # GEO_CHAR=os.path.join(Folders.INPUTS,'geocode_characteristics.csv')
    # MILES_BUSED=os.path.join(Folders.INPUTS,'miles_of_busing_for_optimization.csv')
    # CAP_SCHOOLS=os.path.join(Folders.INPUTS,'capacity_schools.csv')

    DISTANCES = os.path.join(Folders.INPUTS,'distances.csv')  # updated
    STUDENT_COUNTS = os.path.join(Folders.INPUTS, "geocode_student_counts.csv")
    CAPACITIES = os.path.join(Folders.INPUTS,'school_caps.csv')
    PROGRAM_IDXS = os.path.join(Folders.INPUTS, "program_idxs.csv")
    SQRT_DIST_CITP = os.path.join(Folders.INPUTS, "sqrt_dist_ctip.csv")
    SQRT_DIST_NOCTIP = os.path.join(Folders.INPUTS, "sqrt_dist_noctip.csv")
    WALK_ZONE = os.path.join(Folders.INPUTS, "walk_zone.csv")
    AA_MATCH = os.path.join(Folders.INPUTS, "aa_match.csv")
    LANG_MATCH = os.path.join(Folders.INPUTS, "lang_match.csv")
    FRL = os.path.join(Folders.INPUTS, "geocode_frl.csv")
    SIBLING_FRL = os.path.join(Folders.INPUTS, "sibling_frl.csv")
    FIXED_EFFECT = os.path.join(Folders.INPUTS, "fixed_effects.csv")
    PROGRAM_ELIGIBILITY = os.path.join(Folders.INPUTS, "program_eligibility_adjustment.csv")

    # REGION_RATIO=os.path.join(Folders.INPUTS,'regional_population.csv')
    # SAMPLING_WEIGHTS=os.path.join(Folders.INPUTS,'geocode_sampling_weights.csv')

class ZoneFiles(object):
    HEADER='Zone'
    NEIGHBORHOOD_NEW=os.path.join(Folders.INPUTS,'geocode_to_region_mapping.csv')
    
class Files(object):
    # input files
    GOOGLE_GEO_DIST=os.path.join(Folders.INPUTS,'geocode_to_geocode_distances.csv')
    GEOCODE_COORDS=os.path.join(Folders.INPUTS,'geocode_location.csv')
    WALK_ZONE=os.path.join(Folders.INPUTS,'walk_zone_table.csv')