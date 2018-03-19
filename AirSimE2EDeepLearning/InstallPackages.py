import os

# Run this script from within an anaconda virtual environment to install the required packages
# Be sure to run this script as root or as administrator.

os.system('python -m pip install --upgrade pip')
#os.system('conda update -n base conda')
os.system('conda install jupyter')
os.system('pip install matplotlib==2.1.2')
os.system('pip install image')
os.system('pip install keras_tqdm')
os.system('conda install -c conda-forge opencv')
os.system('pip install msgpack-rpc-python')
os.system('pip install pandas')
os.system('pip install numpy')
os.system('conda install scipy')