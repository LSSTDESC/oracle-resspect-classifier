# utility for plotting sample lightcurves for data verification - may get merged into another file at some point

from oracle.custom_datasets.ELAsTiCC import ELAsTiCC_LC_Dataset

class ELAsTiCC_plotter(ELAsTiCC_LC_Dataset):
    def __init__(self):
        self.include_lc_plots = True