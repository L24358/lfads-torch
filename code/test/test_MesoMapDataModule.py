from lfads_torch.datamodules import MesoMapDataModule

dm = MesoMapDataModule(456774, "all",)
dm.setup()

for batch_data in dm.train_dataloader():
    import pdb; pdb.set_trace()