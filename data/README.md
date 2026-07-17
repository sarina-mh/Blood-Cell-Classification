# Data

This folder is where the dataset lives locally. It's not tracked in git —
5,000 microscopy images don't belong in version control, and GitHub rejects
any single file over 100 MB outright, which a zipped version of this set
will often exceed.

## Expected layout

`BloodCellDataModule` uses `torchvision.datasets.ImageFolder`, which expects
one subfolder per class:

```
data/raw/
├── basophil/
│   ├── img001.jpg
│   └── ...
├── erythroblast/
├── monocyte/
├── myeloblast/
└── seg_neutrophil/
```

5 classes, ~1,000 images per class, 5,000 images total. Point `--data-root`
(or `DataConfig.data_root`) at the folder that contains those five
subfolders — `data/raw` is the default.

## Getting the data

Use the same source your course or project provided the dataset from
originally (the folder names above should match exactly what it ships with).
If you're rebuilding this from scratch, search Kaggle for the "Blood Cell
Cancer (ALL)" dataset family — several versions exist with slightly
different class sets, so double check the five class names line up with the
ones above before training.

If you'd rather keep a local zip archive around for backup, drop it in this
folder (e.g. `data/archive.zip`) and unzip it into `data/raw/` — it's already
covered by `.gitignore` so it won't get committed by accident.
