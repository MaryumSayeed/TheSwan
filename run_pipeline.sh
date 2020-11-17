# first generate power spectra:
echo "Generating power spectra."
python quicklook.py -s 'Gaia' 
python quicklook.py -s 'Seismic' 

echo "Preparing data using psmaketrainfile_one.py"

# prepare input data aka log(power spectra) with labels:

# ---Gaia sample:
python psmaketrainfile_one.py -f pande_pickle_1
echo "---Gaia #1 done!"
python psmaketrainfile_one.py -f pande_pickle_2
echo "---Gaia #2 done!"
python psmaketrainfile_one.py -f pande_pickle_3
echo "---Gaia #3 done!"

# ---Seismic sample:
python psmaketrainfile_one.py -f astero_final_sample_1 
echo "---Seismic #1 done!"
python psmaketrainfile_one.py -f astero_final_sample_2
echo "---Seismic #2 done!"
python psmaketrainfile_one.py -f astero_final_sample_3
echo "---Seismic #3 done!"
python psmaketrainfile_one.py -f astero_final_sample_4
echo "---Seismic #4 done!"

# run The Swan:
echo "Runinng LLR on Gaia..."
python LLR_logg.py -sample Gaia -d LLR_gaia
echo "Runinng LLR on Seismic..."
python LLR_logg.py -sample Seismic -d LLR_seismic

