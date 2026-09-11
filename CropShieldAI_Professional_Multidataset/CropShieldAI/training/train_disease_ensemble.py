"""Train a presentation-friendly multi-disease ensemble from PlantVillage.

Downloads the public PlantVillage color dataset via Hugging Face.
Default: 200 images/class, all 38 classes, using simple RGB features + three classifiers
combined by hard voting. This is intentionally lightweight enough for a laptop CPU.
"""
from pathlib import Path
import argparse, json, numpy as np
from PIL import Image

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'models/disease_ensemble.joblib'


def extract(img):
    img=img.convert('RGB').resize((24,24))
    x=np.asarray(img,dtype=np.float32)/255.0
    # raw low-res RGB + channel statistics + color histograms
    raw=x.reshape(-1)
    stats=np.array([x[:,:,i].mean() for i in range(3)]+[x[:,:,i].std() for i in range(3)],dtype=np.float32)
    hist=np.concatenate([np.histogram(x[:,:,i],bins=16,range=(0,1),density=True)[0] for i in range(3)]).astype(np.float32)
    return np.concatenate([raw,stats,hist])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--per-class',type=int,default=200)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    try:
        from datasets import load_dataset
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        import joblib
    except ImportError as e:
        raise SystemExit('Install requirements first: pip install -r requirements.txt') from e

    ds=load_dataset('mohanty/PlantVillage','color',split='train')
    rng=np.random.default_rng(args.seed)
    label_names=ds.features['label'].names
    indices_by=[]
    for label in range(len(label_names)):
        idx=np.array([i for i,y in enumerate(ds['label']) if y==label])
        rng.shuffle(idx)
        indices_by.extend(idx[:args.per_class].tolist())
    rng.shuffle(indices_by)

    X=[]; y=[]
    for k,i in enumerate(indices_by,1):
        X.append(extract(ds[i]['image']))
        y.append(ds[i]['label'])
        if k%250==0: print(f'features: {k}/{len(indices_by)}')
    X=np.vstack(X); y=np.array(y)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=args.seed,stratify=y)
    estimators=[
      ('lr',LogisticRegression(max_iter=1200,solver='liblinear',random_state=args.seed)),
      ('rf',RandomForestClassifier(n_estimators=180,n_jobs=-1,random_state=args.seed)),
      ('et',ExtraTreesClassifier(n_estimators=220,n_jobs=-1,random_state=args.seed))]
    ens=VotingClassifier(estimators=estimators,voting='hard',n_jobs=-1)
    ens.fit(Xtr,ytr)
    pred=ens.predict(Xte)
    acc=accuracy_score(yte,pred)
    print(f'Validation accuracy: {acc:.4f}')
    OUT.parent.mkdir(exist_ok=True)
    joblib.dump({'model':ens,'labels':label_names,'feature_type':'24x24 RGB + stats + hist','accuracy':float(acc),'per_class':args.per_class},OUT)
    (BASE/'data/disease/training_summary.json').write_text(json.dumps({'dataset':'PlantVillage','classes':len(label_names),'images_used':len(y),'per_class':args.per_class,'validation_accuracy':float(acc)},indent=2))

if __name__=='__main__': main()
