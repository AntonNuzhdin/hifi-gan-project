from operator import xor

from torch.utils.data import ConcatDataset, DataLoader

# import src.augmentations
import src.datasets

# from src import batch_sampler as batch_sampler_module
from src.collate_fn.collate import collate_fn
from src.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == "train":
            #     wave_augs, spec_augs = src.augmentations.from_configs(configs)
            drop_last = True
        else:
            #     wave_augs, spec_augs = None, None
            drop_last = False

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            # datasets.append(configs.init_obj(ds, src.datasets, config_parser=configs, wave_augs=wave_augs, spec_augs=spec_augs))
            datasets.append(configs.init_obj(ds, src.datasets))

        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert xor("batch_size" in params,
                   "batch_sampler" in params), "You must provide batch_size or batch_sampler for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
            if shuffle in params.keys():
                shuffle = params["shuffle"]
            batch_sampler = None
        else:
            raise Exception()

        assert bs <= len(dataset), f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            batch_sampler=batch_sampler,
            drop_last=drop_last,
        )
        dataloaders[split] = dataloader
    return dataloaders
