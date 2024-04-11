from torch import nn
from src.base import BaseModel
from src.model.generator import Generator
from src.model.mpd import MPD
from src.model.msd import MSD


class HIFIGAN(BaseModel):
    def __init__(self, generator_args, periods):
        super().__init__()

        self.generator = Generator(**generator_args)
        self.mpd = MPD()
        self.msd = MSD()
