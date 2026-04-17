from pydantic import BaseModel
from typing import Optional 

"""
Reason: 
    We use Pydantic to define the "contract" for your API. 
    It ensures that any data sent to your endpoint is correctly formatted and
     contains all required features before it even touches your model.
"""

class PredictionRequest(BaseModel):
    # taking raw indixces so the user doesnt have to caluclate
    sweat_index:float
    showalter_index: float
    lifted_index: float
    k_index: float
    totals_totals_index: float
    cape: float
    cine: float
    precipitable_water: float
    thickness_1000_500: float
    plcl: float


class PrediuctionResponse(BaseModel):
    prediction: int
    probability: float
    status: str
    
