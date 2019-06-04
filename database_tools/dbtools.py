import tkp.db
from tkp.db.model import Varmetric
from tkp.db.model import Runningcatalog
from tkp.db.model import RunningcatalogFlux
from tkp.db.model import Newsource
from tkp.db.model import Image
from tkp.db.model import Extractedsource
from sqlalchemy import *
from sqlalchemy.orm import relationship


def access(engine,host,port,user,password,database):
    # Access the database using sqlalchemy
    db = tkp.db.Database(engine=engine, host=host, port=port,
                     user=user, password=password, database=database)
    db.connect()
    session = db.Session()
    print 'connected!'
    return session

def GetVarParams(session,dataset_id):
    # Returns all the variability parameters for sources in a given dataset
    VarParams = session.query(Varmetric,Runningcatalog).select_from(join(Varmetric,Runningcatalog)).filter(Runningcatalog.dataset_id == dataset_id).all()
    return VarParams

def GetTransDataForML(session,dataset_id):
    transients = (session.query(Newsource,Runningcatalog)
                      .select_from(join(Newsource,Runningcatalog))
                      .filter(Runningcatalog.dataset_id == dataset_id)
                     .all())
    return transients

def GetRuncatDataForML(session,dataset_id):
    transients = (session.query(Runningcatalog)
                      .join(Varmetric)
                      .join(RunningcatalogFlux)
#                      .select_from(join(Runningcatalog))
                     .filter(Runningcatalog.dataset_id == dataset_id)
                     .all())
    return transients
