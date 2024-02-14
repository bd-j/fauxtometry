# for each realization

image_dir=${base}/${pid}/${ver}_mosaic
mock_dir=${base}/${pid}/${ver}_mock
mkdir -p $mock_dir

echo tag=$tag
echo mock_dir=$mock_dir

# generate fake images in detection bands
echo "injecting fake sources"
python source_injection.py --image_names $image_dir/mosaic_F200W.fits $image_dir/mosaic_F150W.fits $image_dir/mosaic_F444W.fits \
                           --n_fake 6400 \
                           --tag $tag \
                           --mock_dir $mock_dir
# run detection
echo "detecting sources in fake images"
python ecat_sandro.py --mosaic_dir $mock_dir \
                      --tag $tag \
                      --bands F200W F150W

# compare input and output, constructing completeness catalog
    # 1 = detected
    # 0 = undetected
echo "constructing recovery catalog."
python completeness.py --mock_dir $mock_dir \
                       --tag $tag \
                       --band F200W
