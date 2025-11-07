python3 generate-dataset-structure.py --root datasets/uavdt --dataset uavdt --category-map-vd-to-ivid uavdt_categories_3.json --category-map-ivid-to-coco uavdt_coco_categories_3.json

python3 generate-dataset-structure.py --root datasets/visdrone --dataset visdrone --category-map-vd-to-ivid visdrone_categories_10.json --category-map-ivid-to-coco visdrone_coco_categories_10.json

To setup:

git clone --recurse-submodules https://github.com/mozi30/TemporalAttentionPlayground.git

sudo cp $HOME/TemporalAttentionPlayground/model-setup.ipynb $HOME

Go trough jupyter to setup environment