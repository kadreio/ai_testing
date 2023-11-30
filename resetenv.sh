rm -rf ./.diarize   
python3 -m venv ./.diarize
source ./.diarize/bin/activate
pip3 install cython    
# SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip3 install -r requirements.txt
pip install git+https://github.com/m-bain/whisperx.git
