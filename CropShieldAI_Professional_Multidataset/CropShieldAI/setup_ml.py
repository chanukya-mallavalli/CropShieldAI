from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parent
req=['datasets>=3.0','scikit-learn>=1.4','numpy>=1.26','pillow>=10.0','joblib>=1.3']
print('Installing ML dependencies...')
subprocess.check_call([sys.executable,'-m','pip','install',*req])
print('\nTraining PlantVillage ensemble. Default is 200 images/class for a faster laptop-friendly run.')
subprocess.check_call([sys.executable,str(ROOT/'training'/'train_disease_ensemble.py'),'--per-class','200'])
print('\nDone. Model:', ROOT/'models'/'disease_ensemble.joblib')
print('The irrigation ensemble is already included in models/moisture_ensemble.joblib')
