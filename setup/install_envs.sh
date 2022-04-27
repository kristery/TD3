cd src/env/dm_control
pip install -e .

cd ../dmc2gym
pip install -e .

cd ../../..

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch