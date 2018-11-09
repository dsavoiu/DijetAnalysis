INFILE_MC="/ceph/dsavoiu/Dijet/results/DijetAna_QCD_RunIISummer16MiniAODv2_2018-10-29/merged.root"
INFILE_DATA="/ceph/dsavoiu/Dijet/results/DijetAna_JetHT_Run2016G-Legacy-07Aug2017-v1_2018-10-29/merged.root"
OUTPUT_FILE_SUFFIX="2018-10-31"

echo "Sourcing master ROOT stack... [/cvmfs/sft.cern.ch/lcg/views/dev3/latest/x86_64-slc6-gcc62-opt/setup.sh]..."
source /cvmfs/sft.cern.ch/lcg/views/dev3/latest/x86_64-slc6-gcc62-opt/setup.sh

pp.py -i "$INFILE_MC" \
      --type mc \
      -j10 \
      --log --progress \
      $@ \
      task JetResponse EventYieldMC Flavors OccupancyMC PFEnergyFractionsMC \
      --output-file-suffix "$OUTPUT_FILE_SUFFIX"

pp.py -i "$INFILE_DATA" \
      --type data \
      -j10 \
      --log --progress \
      $@ \
      task EventYield Occupancy TriggerEfficiencies PFEnergyFractions \
      --output-file-suffix "$OUTPUT_FILE_SUFFIX"