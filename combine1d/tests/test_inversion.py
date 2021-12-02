import pytest

from combine1d.core.inversion import prepare_for_combine_inversion
from combine1d.core.data_logging import initialise_DataLogger
from combine1d.core.cost_function import create_cost_fct


class TestInversion:
    def test_default_initialisation(self, hef_gdir):
        # check if error is thrown if gdir is not prepared
        with pytest.raises(AttributeError, match='inversion_input_combine file not found!'):
            initialise_DataLogger(hef_gdir)

        # add default inversion_input_combine and try again
        prepare_for_combine_inversion(hef_gdir, inversion_settings=None, filesuffix='_combine')
        data_logger = initialise_DataLogger(hef_gdir, inversion_input_filesuffix='_combine')

        cost_fct = create_cost_fct(data_logger)
