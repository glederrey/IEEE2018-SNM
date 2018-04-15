from helpers.data import *
from classes.MNLogit import *


def load_model(data_path, type):

    if type == 'raw':
        df_unorm_complex = get_data_raw_model(data_path)

        utility_complex = {
            'TRAIN': [('ASC_TRAIN', 1),
                      ('BETA_TRAIN_TT', 'TRAIN_TT'),
                      ('BETA_TRAIN_CO', 'TRAIN_CO'),
                      ('BETA_HE', 'TRAIN_HE')],
            'SM': [('ASC_SM', 1),
                   ('BETA_SM_TT', 'SM_TT'),
                   ('BETA_SM_CO', 'SM_CO'),
                   ('BETA_HE', 'SM_HE'),
                   ('BETA_SENIOR', 'SENIOR')],
            'CAR': [('ASC_CAR', 1, 0),
                    ('BETA_CAR_TT', 'CAR_TT'),
                    ('BETA_CAR_CO', 'CAR_CO'),
                    ('BETA_SENIOR', 'SENIOR')]
        }

        return MNLogit(df_unorm_complex, utility_complex)

    elif type == 'norm':
        df_norm_complex = get_data_norm_model(data_path)

        utility_complex = {
            'TRAIN': [('ASC_TRAIN', 1),
                      ('BETA_TRAIN_TT', 'TRAIN_TT'),
                      ('BETA_TRAIN_CO', 'TRAIN_CO'),
                      ('BETA_HE', 'TRAIN_HE')],
            'SM': [('ASC_SM', 1),
                   ('BETA_SM_TT', 'SM_TT'),
                   ('BETA_SM_CO', 'SM_CO'),
                   ('BETA_HE', 'SM_HE'),
                   ('BETA_SENIOR', 'SENIOR')],
            'CAR': [('ASC_CAR', 1, 0),
                    ('BETA_CAR_TT', 'CAR_TT'),
                    ('BETA_CAR_CO', 'CAR_CO'),
                    ('BETA_SENIOR', 'SENIOR')]
        }

        return MNLogit(df_norm_complex, utility_complex)


