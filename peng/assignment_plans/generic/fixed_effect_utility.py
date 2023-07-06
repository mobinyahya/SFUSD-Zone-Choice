'''
Created on Jun 27, 2013

@author: pengshi
'''
import numpy as np
import pandas as pd

# from peng.utils.data_cache import DataLoader
from peng.assignment_plans.generic.utility_param import UtilityParam
import os
from peng.constants.locations import Folders, PlanFiles
# from peng.utils.distance_cache import DistanceCacheParams, DistanceCache
# from peng.utils.school_snapshot import SchoolSnapshot


class FixedEffectUtility(UtilityParam):
    ''' A class that implements the MNL utility model for the Boston data set.
        FixedEffectUtility.simple2013K12() is the model estimated from 2013 preferences, which is the main model for optimization 
        FixedEffectUtility.simple2014K12() is the updated model estimated from 2014 preferences.
    '''

    def __init__(
            self,
            program_idxs,
            # sqrt_dist_ctip,
            # sqrt_dist_noctip,
            # walk_zone,
            # aa_match,
            # lang_match,
            # sqrt_dist_ctip_param,
            # sqrt_dist_noctip_param,
            # walk_zone_param,
            # aa_match_param,
            # lang_match_param,
            # fixed_effect,
            # program_eligibility
    ):

        # DistanceCacheParams.update('2014', geoDist)
        self._iset = set(range(922))  #931))

        # schoolChar=SchoolSnapshot.loadCSV(charFile)
        self._jset = set(program_idxs).union({len(program_idxs)})

        # self._idioSize=1.0/distParam
        self._baseUtility = self._buildBaseUtility(
            # sqrt_dist_ctip,
            # sqrt_dist_noctip,
            # walk_zone,
            # aa_match,
            # lang_match,
            # sqrt_dist_ctip_param,
            # sqrt_dist_noctip_param,
            # walk_zone_param,
            # aa_match_param,
            # lang_match_param,
            # fixed_effect,
            # program_eligibility
        )

    #        self._geoEstDist=False

    @staticmethod
    def sfusd2018K():
        # charFile = os.path.join(Folders.INPUTS, 'school_characteristics_2014.csv')
        # charHeader = 'Quality'
        # distParam = 0.610
        # walkParam = 0.223
        # geoDist = os.path.join(PlanFiles.DISTANCES)

        sqrt_dist_ctip_param = -3.0
        sqrt_dist_noctip_param = -2.8
        walk_zone_param = 0.035
        aa_match_param = 0.93
        lang_match_param = 1.28
        program_idxs = pd.read_csv(PlanFiles.PROGRAM_IDXS)["program_idx"].to_numpy()
        # sqrt_dist_noctip = pd.read_csv(PlanFiles.SQRT_DIST_NOCTIP, index_col=[0, 1])
        # sqrt_dist_ctip = pd.read_csv(PlanFiles.SQRT_DIST_CITP, index_col=[0, 1])
        # walk_zone = pd.read_csv(PlanFiles.WALK_ZONE, index_col=[0, 1])
        # aa_match = pd.read_csv(PlanFiles.AA_MATCH, index_col=[0, 1])
        # lang_match = pd.read_csv(PlanFiles.LANG_MATCH, index_col=[0, 1])
        # fixed_effects = pd.read_csv(PlanFiles.FIXED_EFFECT, index_col=0)[["fixed_effect"]].to_numpy()
        # program_eligibility = pd.read_csv(PlanFiles.PROGRAM_ELIGIBILITY, index_col=[0,1])
        return FixedEffectUtility(
            program_idxs,
            # sqrt_dist_ctip,
            # sqrt_dist_noctip,
            # walk_zone,
            # aa_match,
            # lang_match,
            # sqrt_dist_ctip_param,
            # sqrt_dist_noctip_param,
            # walk_zone_param,
            # aa_match_param,
            # lang_match_param,
            # fixed_effects,
            # program_eligibility
        )

    # @staticmethod
    # def simple2014K12():
    #     charFile=os.path.join(Folders.INPUTS,'school_characteristics_2014.csv')
    #     charHeader='Quality'
    #     distParam=0.610
    #     walkParam=0.223
    #     geoDist=os.path.join(PlanFiles.DISTANCES)
    #     return FixedEffectUtility(charFile,charHeader,distParam,walkParam,geoDist)
    #
    # @staticmethod
    # def simple2013K12():
    #     charFile=os.path.join(PlanFiles.FIXED_EFFECT)
    #     charHeader='Quality'
    #     distParam=0.531
    #     walkParam=0.456
    #     geoDist=os.path.join(PlanFiles.DISTANCES)
    #     return FixedEffectUtility(charFile,charHeader,distParam,walkParam,geoDist)

    @property
    def iset(self):
        return self._iset

    def jset(self, i):
        return self._jset

    def idioSize(self,i):
        return 1  #self._idioSize

    def baseUtility(self, i, j):
        return self._baseUtility[int(i), int(j)]
        # return self._baseUtility[i][j]

    def _buildBaseUtility(
            self,
            # sqrt_dist_ctip,
            # sqrt_dist_noctip,
            # walk_zone,
            # aa_match,
            # lang_match,
            # sqrt_dist_ctip_param,
            # sqrt_dist_noctip_param,
            # walk_zone_param,
            # aa_match_param,
            # lang_match_param,
            # fixed_effects,
            # program_eligibility
    ):
        # base_utility = np.zeros((len(self._iset), len(self._jset) - 1))
        # base_utility += sqrt_dist_ctip_param * sqrt_dist_ctip.values
        # base_utility += sqrt_dist_noctip_param * sqrt_dist_noctip.values
        # base_utility += walk_zone_param * walk_zone.values
        # base_utility += aa_match_param * aa_match.values
        # base_utility += lang_match_param * lang_match.values
        # base_utility += np.outer(np.ones(len(self._iset)), fixed_effects)
        # base_utility += 50
        # base_utility = np.multiply(base_utility, program_eligibility.values)
        # base_utility -= 1 - program_eligibility.values
        # base_utility = np.hstack([base_utility, -1 * np.ones((base_utility.shape[0], 1))])
        # np.save("../../../data/base_utility.npy", base_utility)
        base_utility = np.load("../../../data/avg_base_utilities2.npy")
        # base_utility = np.load("../../../data/avg_base_utilities_bigpenalty.npy")

        return base_utility

        # dc = DistanceCache()
        # ans = {}
        #
        # #        geoUtil=GeoUtil()
        # walk = DataLoader().walkGeo
        # for i in self.iset:
        #     ans[i] = {}
        #     for j in self.jset(i):
        #         fix = schoolChar.get(j, charHeader, True)
        #         # dist=geoUtil.walkDist(i, schoolChar.get(j,H.GEO,False))
        #         dist = dc.walkDist('2014', i, j)
        #         tmp = fix - distParam * dist
        #         if j in walk[i]:
        #             tmp += walkParam
        #         ans[i][j] = tmp / distParam
        # return ans
