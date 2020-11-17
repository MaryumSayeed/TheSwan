# The_Swan
This is the code to use local linear regression on Kepler power spectra to infer stellar surface gravities using data-driven modeling. The code also references, and is based on the infrastructure, of The Cannon ([Ness et al. 2015](https://ui.adsabs.harvard.edu/abs/2018ApJ...866...15N/abstract); [The Cannon](https://github.com/mkness/TheCannon)).

## Authors
* Maryum Sayeed (UBC)
* Daniel Huber (IfA)
* Adam Wheeler (Columbia)
* Melissa Ness (Columbia, Flatiron)

## License
Copyright 2020 the authors. The_Swan is free software made available under the MIT License. For details see the file [LICENSE.md](LICENSE.md).

## At a Glance:
* Run quicklook.py on light curves to generate power spectra.
* Run psmaketrainfile_one.py to convert stellar power spectra to log(power spectra).
* Run LLR_logg.py to run local linear regression on both samples.