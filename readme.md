# fauxtometry

## requirements

numpy
scipy
astropy
sep
galsim
webbpsf


```sh
module purge
module load python/3.10.9-fasrc01
mamba create -n faux python=3.10 pip ipython pyyaml numpy scipy astropy

mamba activate faux
python -m pip install webbpsf
python -m pip install galsim
python -m pip install sep

wget https://stsci.box.com/shared/static/qxpiaxsjwo15ml6m4pkhtk36c9jgj70k.gz
mv qxpiaxsjwo15ml6m4pkhtk36c9jgj70k.gz webbpsf-data-LATEST.tar.gz
tar -xf webbpsf-data-LATEST.tar.gz
export WEBBPSF_PATH=$PWD/webbpsf-data
```
