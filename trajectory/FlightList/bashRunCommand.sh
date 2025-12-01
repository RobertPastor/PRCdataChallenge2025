#!/bin/bash
echo "----------python version------------"
python --version
echo $PYTHINPATH
export PYTHONPATH="C/Users/rober/AppData/Roaming/Python/Python311"

echo "----------PYTHONPATH------------"
echo $PYTHONPATH
export PYTHONPATH=$PYTHONPATH:"/c/Users/rober/git/PRCdataChallenge2025"
export PYTHONPATH=$PYTHONPATH:"/c/Users/rober/git/PRCdataChallenge2025/trajectory/"
export PYTHONPATH=$PYTHONPATH:"/c/Users/rober/git/PRCdataChallenge2025/trajectory/FlightList/"
echo "----------PYTHONPATH------------"
echo $PYTHONPATH

cd "/c/Users/rober/git/PRCdataChallenge2025"
echo "----------pwd------------"
pwd

echo $HOME

python -c "$HOME/git/PRCdataChallenge2025/trajectory/FlightList/MAIN_RebuildFlightTrajectory_Rank.py"