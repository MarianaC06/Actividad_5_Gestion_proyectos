"""Grid search con CV, holdout independiente y trazabilidad MLflow."""
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from datos_prep import limpiar

ROOT=Path(__file__).resolve().parents[1]
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as fd:
        for chunk in iter(lambda:fd.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default=str(ROOT/'datos/datos_ini/uber.csv'))
    ap.add_argument('--max-rows',type=int,default=30000)
    args=ap.parse_args()
    raw=Path(args.input)
    if not raw.exists():raise SystemExit('Falta uber.csv. Consulta README para descargarlo.')
    clean=ROOT/'datos/datos_limp/uber_limpio.csv'
    audit=limpiar(raw,clean)
    df=pd.read_csv(clean)
    if args.max_rows>0 and len(df)>args.max_rows:df=df.sample(n=args.max_rows,random_state=42)
    X=df.drop(columns=['fare_amount']);y=df.fare_amount
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    models={
      'Ridge':(Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',Ridge())]),{'model__alpha':[.1,1,10,100]}),
      'RandomForest':(Pipeline([('imputer',SimpleImputer(strategy='median')),('model',RandomForestRegressor(random_state=42,n_jobs=1))]),{'model__n_estimators':[80,140],'model__max_depth':[12,20],'model__min_samples_leaf':[2,5]})}
    cv=KFold(n_splits=5,shuffle=True,random_state=42)
    (ROOT/'evidencias').mkdir(exist_ok=True)
    # Configurar MLflow con SQLite
    DB_PATH = ROOT / "mlflow.db"

    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.resolve().as_posix()}")

    mlflow.set_experiment("Actividad5_Uber_Tarifas")
    results=[];errors={}
    for name,(pipe,grid) in models.items():
      with mlflow.start_run(run_name=name):
        search=GridSearchCV(pipe,grid,cv=cv,scoring='neg_root_mean_squared_error',n_jobs=-1,refit=True,return_train_score=True)
        search.fit(Xtr,ytr);pred=search.predict(Xte)
        metrics={'MAE':mean_absolute_error(yte,pred),'RMSE':np.sqrt(mean_squared_error(yte,pred)),'R2':r2_score(yte,pred),'CV_RMSE':-search.best_score_,'CV_RMSE_std':search.cv_results_['std_test_score'][search.best_index_]}
        mlflow.log_params({'modelo':name,'cv_folds':5,'seed':42,'n_train':len(Xtr),'n_test':len(Xte),'dataset_sha256':sha256(raw),'clean_sha256':sha256(clean),**search.best_params_})
        mlflow.log_metrics(metrics)
        mlflow.log_dict(audit,'auditoria_limpieza.json')
        mlflow.log_dict({'grid':grid,'best_params':search.best_params_},'configuracion.json')
        mlflow.sklearn.log_model(
          sk_model=search.best_estimator_,
          name="modelo",
          serialization_format="cloudpickle"
        )
        fig,ax=plt.subplots(figsize=(6,4));ax.scatter(yte,pred,s=5,alpha=.25);ax.plot([yte.min(),yte.max()],[yte.min(),yte.max()],'k--');ax.set(xlabel='Tarifa real',ylabel='Tarifa estimada',title=name);fig.tight_layout()
        chart=ROOT/'evidencias'/f'predicciones_{name}.png';fig.savefig(chart,dpi=140);plt.close(fig);mlflow.log_artifact(str(chart))
        errors[name]=np.abs(yte.to_numpy()-pred)
        results.append({'modelo':name,**metrics,'hiperparametros':str(search.best_params_)})
    out=pd.DataFrame(results);out.to_csv(ROOT/'evidencias/comparacion_modelos.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,4));ax.bar(out.modelo,out.RMSE);ax.set(ylabel='RMSE (USD)',title='Comparación en prueba');fig.tight_layout();fig.savefig(ROOT/'evidencias/comparacion_rmse.png',dpi=140);plt.close(fig)
    stat,pvalue=wilcoxon(errors['Ridge'],errors['RandomForest'],zero_method='wilcox')
    (ROOT/'evidencias/analisis_estadistico.json').write_text(json.dumps({'prueba':'Wilcoxon pareada sobre errores absolutos de test','estadistico':float(stat),'p_value':float(pvalue),'advertencia':'Dependencia entre observaciones y tamaño muestral afectan interpretación; comparación exploratoria.'},indent=2),encoding='utf-8')
    print(out.to_string(index=False));print('Wilcoxon p:',pvalue)
    print(
    f"MLflow: mlflow ui "
    f"--backend-store-uri sqlite:///{DB_PATH.resolve().as_posix()} "
    f"--port 5000"
    )
if __name__=='__main__':main()
