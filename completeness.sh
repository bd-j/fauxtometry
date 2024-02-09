# for each realization

echo tag=$tag

image_dir=${base}/${pid}/${ver}_mosaic
mock_dir=${base}/${pid}/${ver}_mock
mkdir -p $mock_dir

# generate fake images in detection bands
python source_injection.py --image_names $image_dir/mosaic_F200W.fits $image_dir/mosaic_F150W.fits \
                           --n_fake 6400 \
                           --tag $tag \
                           --mock_dir $mock_dir
# run detection
python ecat_sandro.py --mosaic_dir $mock_dir \
                      --tag $tag \
                      --bands F200W F150W

# compare input and output, constructing completeness catalog
    # 1 = detected
    # 0 = undetected
python completeness.py --mock_dir $mock_dir \
                       --tag $tag \
                       --band F200W
