#########################################################################################################
# Translated to .py by Evanthia Kazagli
# Oct. 2016
#########################################################################################################

# NL
from biogeme import *
from headers import *
from mev import *
from nested import *
from loglikelihood import *
from statistics import *
  
#########################################################################################################
# [Parameters]
# Arguments:
#   Beta('Name', starting value, lower bound, upper bound, 0: estimate the parameter, 1: keep it fixed)
ASC_CAR	= Beta('ASC_CAR',0,-1,1,0)
ASC_SBB	= Beta('ASC_SBB',0,-1,1,1)
ASC_SM = Beta('ASC_SM',0,-1,1,0)
B_CAR_TIME = Beta('B_CAR_TIME',0,-1,1,0)
B_COST = Beta('B_COST',0,-1,1,0)
B_GA = Beta('B_GA',0,-1,5,0)
B_HE = Beta('B_HE',0,-1,1,0)
B_SM_TIME = Beta('B_SM_TIME',-0,-1,1,0)
B_TRAIN_TIME = Beta('B_TRAIN_TIME',0,-1,1,0)
# parameters relevant to the nests
MU_classic = Beta('MU_classic',1,1,10,0)

#########################################################################################################
# [Expressions]one  = DefineVariable('one',1)
one  = DefineVariable('one',1)
CAR_AV_SP  = DefineVariable('CAR_AV_SP', CAR_AV    *  (  SP   !=  0  ))
SM_COST  = DefineVariable('SM_COST', SM_CO   * (  GA   ==  0  ))
TRAIN_AV_SP  = DefineVariable('TRAIN_AV_SP', TRAIN_AV    *  (  SP   !=  0  ))
TRAIN_COST  = DefineVariable('TRAIN_COST', TRAIN_CO   * (  GA   ==  0  ))

#########################################################################################################
#[Utilities]
V1 = ASC_SBB * one + B_TRAIN_TIME * TRAIN_TT + B_COST * TRAIN_COST + B_HE * TRAIN_HE + B_GA * GA
V2 = ASC_SM * one + B_SM_TIME * SM_TT + B_COST * SM_COST + B_HE * SM_HE + B_GA * GA
V3 = ASC_CAR * one + B_CAR_TIME * CAR_TT + B_COST * CAR_CO

V = {1: V1, 2: V2, 3: V3}

av = {1: TRAIN_AV_SP, 2: SM_AV, 3: CAR_AV_SP}

#[Exclude]
exclude = (( PURPOSE != 1 ) * (  PURPOSE   !=  3  ) + ( CHOICE == 0 )) + ( AGE   ==  6  ) > 0
BIOGEME_OBJECT.EXCLUDE = exclude

#########################################################################################################
#[Definition of nests]
innovative = 1.0, [2]
classic = MU_classic, [1, 3]

nests = classic, innovative

#########################################################################################################
#[Model]
# NL
logprob = lognested(V,av,nests,CHOICE)

# Defines an itertor on the data
rowIterator('obsIter')

#########################################################################################################
# [Estimation]
BIOGEME_OBJECT.ESTIMATE = Sum(logprob,'obsIter')

#########################################################################################################
# [Statistics]
nullLoglikelihood(av,'obsIter')
choiceSet = [1,2,3]
cteLoglikelihood(choiceSet,CHOICE,'obsIter')
availabilityStatistics(av,'obsIter')

#########################################################################################################
# [BIOGEME_OBJECT]
BIOGEME_OBJECT.PARAMETERS['optimizationAlgorithm'] = "BIO"

BIOGEME_OBJECT.FORMULAS['Train utility'] = V1
BIOGEME_OBJECT.FORMULAS['Swissmetro utility'] = V2
BIOGEME_OBJECT.FORMULAS['Car utility'] = V3


