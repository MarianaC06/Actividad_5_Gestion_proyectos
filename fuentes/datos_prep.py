"""Preparación reproducible del conjunto de tarifas Uber."""
import numpy as np
import pandas as pd
from pathlib import Path

def limpiar(origen, destino):
    df = pd.read_csv(origen)
    inicial = len(df)
    df = df.drop(columns=['Unnamed: 0', 'key'], errors='ignore').drop_duplicates()
    requeridas = ['fare_amount','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','passenger_count']
    faltan = set(requeridas)-set(df.columns)
    if faltan: raise ValueError(f'Columnas ausentes: {faltan}')
    for c in requeridas:
        if c != 'pickup_datetime': df[c] = pd.to_numeric(df[c], errors='coerce')
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'],utc=True,errors='coerce')
    df = df.dropna(subset=requeridas)
    df = df[df.fare_amount.between(2.5, 250) & df.passenger_count.between(1, 6)]
    # Región metropolitana de Nueva York; supuesto explícito, no validación universal.
    df = df[df.pickup_latitude.between(40.4,41.1) & df.dropoff_latitude.between(40.4,41.1) & df.pickup_longitude.between(-74.5,-73.5) & df.dropoff_longitude.between(-74.5,-73.5)].copy()
    lat1=np.radians(df.pickup_latitude); lat2=np.radians(df.dropoff_latitude)
    dlat=lat2-lat1; dlon=np.radians(df.dropoff_longitude-df.pickup_longitude)
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    df['distancia_km']=6371*2*np.arcsin(np.sqrt(np.clip(a,0,1)))
    df=df[df.distancia_km.between(0.1,80)].copy()
    df['hora']=df.pickup_datetime.dt.hour
    df['dia_semana']=df.pickup_datetime.dt.dayofweek
    df['mes']=df.pickup_datetime.dt.month
    df['anio']=df.pickup_datetime.dt.year
    df=df.drop(columns=['pickup_datetime'])
    destino=Path(destino); destino.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(destino,index=False)
    return {'originales':inicial,'limpios':len(df),'descartados':inicial-len(df)}
