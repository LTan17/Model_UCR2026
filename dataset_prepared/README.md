# Train drivable area only

Source: ../dataset (Roboflow semantic masks: 0=background, 1=road).
Images and masks are copied without modifying the source; valid maps to val.
Each segment/<name>.png corresponds to images/<name>.jpg.
The single-task loader thresholds values >0, so these 0/1 masks are supported.

Run from the repository root:

python train_singletask.py --task DA --config small --width 640 --height 384 --batch_size 8 --num_workers 0 --savedir runs/drivable

The test split is reserved for final evaluation and is not used during training.
