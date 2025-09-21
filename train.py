from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(
    dataset_dir = "datasets/uavdt/RF_DETR_COCO",
    epochs = 10,
    batch_size = 2,
    grad_accum_steps = 4,
    lr = 1e-4,
    output_dir = "trained-models"
)