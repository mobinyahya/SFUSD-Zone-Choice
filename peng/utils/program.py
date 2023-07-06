'''
Created on Dec 21, 2012

@author: pengshi
'''

from peng.utils.data_equal import DataEqualable
from peng.utils import data_equal, data_reader
from collections import OrderedDict


class Program(DataEqualable):
    '''
    A helper class to read in certain types of school information from BPS data sets.
    '''

    def __init__(self,id,char=OrderedDict()):
        '''
        Constructor
        '''
        if not isinstance(id,ProgramId):
            raise TypeError
        self._id=id
        self._char=self._inferChar(char)
        self._char.update(self._id.toDict())
        
    @staticmethod
    def fromDict(_dict):
        id=ProgramId.fromDict(_dict)
        return Program(id,_dict)
    
    def toDict(self):
        ans=self._char.copy()
        for k in ProgramCharHeader.POSSIBLE_NONE:
            if self._char[ProgramCharHeader.NO+k]:
                ans[k]=''
            del ans[ProgramCharHeader.NO+k]
        return ans
    
    @property
    def id(self):
        return self._id
    
    @property
    def char(self):
        return self._char
    
    @property
    def kind(self):
        H=ProgramCharHeader
        if self._char[H.IS_SPED4]:
            return ProgramKind.SPED4
        elif self._char[H.IS_ELL]:
            return ProgramKind.ELL
        else:
            return ProgramKind.REGULAR
        
    def passFilter(self,reqFields=[],reqValues=[],banFields=[],banValues=[]):
        if data_reader.passFilter(self._char, reqFields, reqValues, banFields, banValues):
            return True

    @property
    def shortName(self):
        ans = '%s-%s'%(self._char[ProgramCharHeader.SCHOOL_LABEL],self.id.pid)
        return ans
        
    def dataEquals(self,object):
        if not isinstance(object,Program):
            return False
        return data_equal.mapEqual(self.char, object.char)
    
    def __str__(self):
        return str(self.id)
    
    def canApply(self,student):
        if student.year!=self.id.year or student.grade!=self.id.grade:
            return False
        if self.kind not in ProgramKind.canAccess[student.kind]:
            return False
        if self.kind==ProgramKind.ELL and self._char[ProgramCharHeader.ELL_LANG_CODE]:
            if student.ellLangCode != self._char[ProgramCharHeader.ELL_LANG_CODE]:
                return False
        return True
    
    def getPriority(self,student):
        sid=self.id.sid
        pid=self.id.pid
        guarantee=(sid,pid) in student.progAff[ProgAffHeader.PRESENT_PROG]
        present=sid in student.schoolAff[SchoolAffHeader.PRESENT]
        sibling=sid in student.schoolAff[SchoolAffHeader.SIBLING]
        walk=sid in student.schoolAff[SchoolAffHeader.WALK]
        eb=(self.char['East Boston']==student.char['East Boston'])
        if guarantee:
            return Priority.GUARANTEE
        else:
            ans=''
            if present:
                ans+=Priority.PRESENT
            if sibling:
                ans+=Priority.SIBLING
            if walk:
                ans+=Priority.WALK
            if eb:
                ans+=Priority.EAST_BOSTON
            if not ans:
                ans=Priority.NO_PRIORITY
            return ans

    def _inferChar(self,_dict):
        ans=OrderedDict()
        for k in ProgramCharHeader.TYPES:
            if k in ProgramCharHeader.INT:
                ans[k]=int(_dict[k]) if k in _dict and _dict[k]!='' else 0
            elif k in ProgramCharHeader.FLOAT:
                ans[k]=float(_dict[k]) if k in _dict and _dict[k]!='' else 0
            else:
                ans[k]=str(_dict[k]) if k in _dict else ''
            
            if k in ProgramCharHeader.POSSIBLE_NONE:
                if k in _dict and str(_dict[k]).strip()!='':
                    ans[ProgramCharHeader.NO+k]=0
                else:
                    ans[ProgramCharHeader.NO+k]=1
        for klist in ProgramCharHeader.INDICATORS:
            ans[klist[0]]=1-sum(ans[klist[i]] for i in range(1,len(klist)))
        return ans
    
class ProgramCharHeader(object):
    YEAR='Year'
    GRADE='Grade'
    SID='School Code'
    PID='Program Code'
    SCHOOL_NAME='School Name'
    SCHOOL_LABEL='School Label'
    CUR_ZONE='3 Zone'
    ELL_CLUSTER='ELL Cluster'
    ELL_LANG_CODE='ELL Spec. Lang. Code'
    SCHOOL_HOURS='School Hours'
    
    # School types
    EAST_BOSTON='East Boston'
    ELC='ELC'
    PILOT='Pilot'
    CITYWIDE="Citywide"
    AWC='AWC'
    UNIFORM='Mandatory Uniform'
    SPORTS='Has Sports'
    
    #Quality metrics
    ART='Art/Music'
    GENFAC='General Facilities'
    SPORTFAC='Sports Facilities'
    FUNFAC='Fun Facilities'
    
    #Grades
    LOWGRADE='Low Grade'
    HIGHGRADE='High Grade'
    
    #Locations
    GEO='Geocode'
    LONGITUDE='Longitude'
    LATITUDE='Latitude'

    ENROLLMENT='Enrollment'
    
    PCT_BLACK='% Black'
    PCT_WHITE='% White'
    PCT_ASIAN='% Asian'
    PCT_HISPANIC='% Hispanic'
    PCT_OTHER='% Other'
    
    PCT_SPED='% SPED'
    PCT_LEP='% LEP'
    
    PCT_WHITE_ASIAN='% White/Asian'
    
    PCT_FREE='% Free Lunch'
    PCT_FREE_REDUCED='% Free/Reduced Lunch'
    PCT_NON_FREE_REDUCED='% Non-Free/Reduced Lunch'
    
    MCAS='MCAS Composite'
    BPS_RANK='BPS Rank'
    DESE='DESE'
    
    MATH35='Math % Adv/Prof (3-5)'
    ELA35='ELA % Adv/Prof (3-5)'
    NOMCAS='No MCAS'
    
    WALK_PROP='Walk Proportion'
    
    IS_ELL='ELL'
    IS_SPED4='SPED4'
    IS_REGULAR='Regular'
    
    TIER1='Tier1'
    TIER12='Tier12'
    TIER123='Tier123'
    CAPSCHOOL='Capacity School'

    MULTI_LANG_ELL='Multi-lingual ELL'
    SPEC_LANG_ELL='Spec. Lang. ELL'

    INDICATORS=[[PCT_OTHER,PCT_BLACK,PCT_WHITE,PCT_ASIAN,PCT_HISPANIC],[PCT_NON_FREE_REDUCED,PCT_FREE_REDUCED]]    
    NO='NO_'
    POSSIBLE_NONE=set([MCAS,BPS_RANK,DESE])
    FLOAT_LIST=[LONGITUDE,LATITUDE,PCT_BLACK,PCT_WHITE,PCT_ASIAN,PCT_HISPANIC,PCT_WHITE_ASIAN,PCT_FREE,PCT_FREE_REDUCED,MCAS,BPS_RANK,WALK_PROP,LONGITUDE,LATITUDE,PCT_SPED,PCT_LEP,MATH35,ELA35]
    FLOAT=set(FLOAT_LIST)
    INT_LIST=[DESE,IS_ELL,IS_SPED4,IS_REGULAR, EAST_BOSTON, ELC,PILOT,CITYWIDE,AWC,UNIFORM,SPORTS,ART,GENFAC,SPORTFAC,FUNFAC,ENROLLMENT,NOMCAS,TIER1,TIER12,TIER123,CAPSCHOOL]
    INT=set(INT_LIST)
    
    TYPES=[YEAR,GRADE,SID,PID,SCHOOL_NAME,SCHOOL_LABEL,GEO,CUR_ZONE,ELL_CLUSTER,ELL_LANG_CODE,LOWGRADE,HIGHGRADE,SCHOOL_HOURS]+FLOAT_LIST+INT_LIST
    
class Priority(object):
    GUARANTEE='Guarantee'
    PRESENT='PresentSchool'
    SIBLING='Sibling'
    WALK='Walk'
    NO_PRIORITY='NoPriority'
    EAST_BOSTON='SameSide'
    ADMIN='AdministrativeAssignment'
    
class ProgramKind(object):
    REGULAR='Reg. Ed.'
    ELL='Non-SPED4 ELL'
    SPED4='SPED4'
    TYPES=[REGULAR,ELL,SPED4]


class ProgramId(object):
    
    _JOIN_CHAR='-'
    HEADER='School-Program ID'
    
    def __init__(self,year,grade,sid,pid):
        self._year=str(year).strip()
        self._grade=str(grade).strip()
        self._sid=str(sid).strip()
        self._pid=str(pid).strip()
        
    @staticmethod
    def fromDict(_dict):
        P=ProgramCharHeader
        if P.YEAR not in _dict or P.GRADE not in _dict or P.SID not in _dict or P.PID not in _dict:
            raise ValueError
        return ProgramId(_dict[P.YEAR],_dict[P.GRADE],_dict[P.SID],_dict[P.PID])
    
    @staticmethod
    def fromStr(_str):
        seq=_str.split(ProgramId._JOIN_CHAR)
        return ProgramId.fromSeq(seq)
    
    @staticmethod
    def fromSeq(seq):
        if len(seq)!=4:
            raise ValueError('sequence %s needs to be length 4'%seq)
        return ProgramId(*seq)
    
    def toDict(self):
        ans={}
        P=ProgramCharHeader
        ans[P.YEAR]=self.year
        ans[P.GRADE]=self.grade
        ans[P.SID]=self.sid
        ans[P.PID]=self.pid
        return ans
        
    def toTuple(self):
        return self.year,self.grade,self.sid,self.pid
    
    def __str__(self):
        return ProgramId._JOIN_CHAR.join(self.toTuple())
    
    def __hash__(self):
        return hash(self.toTuple())
    
    def __eq__(self,item):
        if not isinstance(item,ProgramId):
            return False
        return self.toTuple()==item.toTuple()
    
    def __ne__(self, item):
        return not (self==item)
    
    @property
    def year(self):
        return self._year
    
    @property
    def grade(self):
        return self._grade
    
    @property
    def sid(self):
        return self._sid
    
    @property
    def pid(self):
        return self._pid
    
    
class ProgAffHeader(object):
    PRESENT_PROG='Present Program'
    TYPES=[PRESENT_PROG]
    
class SchoolAffHeader(object):
    SIBLING='Sibling'
    PRESENT='Present School'
    WALK='Walk'
    
    TYPES=[SIBLING,WALK,PRESENT]