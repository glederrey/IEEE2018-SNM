from biogeme import *
from headers import *
from loglikelihood import *
from statistics import *

#Parameters to be estimated
# Arguments:
#   1  Name for report. Typically, the same as the variable
#   2  Starting value
#   3  Lower bound
#   4  Upper bound
#   5  0: estimate the parameter, 1: keep it fixed

ASC_CAR	= Beta('ASC_CAR',0,-10000,10000,1)
ASC_TRAIN = Beta('ASC_TRAIN',0,-10000,10000,0)
ASC_SM = Beta('ASC_SM',0,-10000,10000,0)

BETA_CAR_TT = Beta('BETA_CAR_TT',0,-10000,10000,0)
BETA_TRAIN_TT = Beta('BETA_TRAIN_TT',0,-10000,10000,0)
BETA_SM_TT = Beta('BETA_SM_TT',0,-10000,10000,0)

BETA_CAR_CO = Beta('BETA_CAR_CO',0,-10000,10000,0)
BETA_TRAIN_CO = Beta('BETA_TRAIN_CO',0,-10000,10000,0)
BETA_SM_CO = Beta('BETA_SM_CO',0,-10000,10000,0)

BETA_HE = Beta('BETA_HE',0,-10000,10000,0)
BETA_SENIOR = Beta('BETA_SENIOR',0,-10000,10000,0)

# Define here arithmetic expressions for name that are not directly available from the data
one = DefineVariable('one',1)
SENIOR = DefineVariable('SENIOR', AGE == 5)

TRAIN_CO   = DefineVariable('TRAIN_COST', TRAIN_CO * (GA==0))
SM_CO     = DefineVariable('SM_COST', SM_CO * (GA==0))

# Utilities
CAR = ASC_CAR * one + BETA_CAR_TT * CAR_TT + BETA_CAR_CO * CAR_CO + BETA_SENIOR * SENIOR
TRAIN = ASC_TRAIN * one + BETA_TRAIN_TT * TRAIN_TT + BETA_TRAIN_CO * TRAIN_CO + BETA_HE * TRAIN_HE
SM = ASC_SM * one + BETA_SM_TT * SM_TT + BETA_SM_CO * SM_CO + BETA_HE * SM_HE + BETA_SENIOR * SENIOR

V = {1: TRAIN , 2: SM, 3: CAR}
av = {1: one, 2: one, 3: one}

#Exclude
exclude = ((CHOICE == 0) + (CAR_TT <= 0) + (AGE == 6)) > 0
BIOGEME_OBJECT.EXCLUDE = exclude

# MNL (Multinomial Logit model), with availability conditions
logprob = bioLogLogit(V, av, CHOICE)

# Defines an iterator on the data
rowIterator('obsIter')

# Define the likelihood function for the estimation
BIOGEME_OBJECT.ESTIMATE = Sum(logprob,'obsIter')

# Optimization algorithm
BIOGEME_OBJECT.PARAMETERS['optimizationAlgorithm'] = "BIO"

# Debug purpose
BIOGEME_OBJECT.PARAMETERS['checkDerivatives'] = "1"

# Print some statistics:
nullLoglikelihood(av, 'obsIter')
choiceSet = [1,2,3]
cteLoglikelihood(choiceSet, CHOICE, 'obsIter')
availabilityStatistics(av, 'obsIter')
BIOGEME_OBJECT.FORMULAS['Train'] = TRAIN
BIOGEME_OBJECT.FORMULAS['Car'] = CAR
BIOGEME_OBJECT.FORMULAS['SwissMetro'] = SM
