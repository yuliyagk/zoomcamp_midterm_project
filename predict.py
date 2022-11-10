import numpy as np

import bentoml
from bentoml.io import JSON

model_ref= bentoml.xgboost.get("star_type_model:bk5qpydbg2too4yo")

dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()
svc = bentoml.Service("startype", runners=[model_runner])

# Brown Dwarf   -> Star Type = 0 -> 0.0
# Red Dwarf     -> Star Type = 1 -> 0.2
# White Dwarf   -> Star Type = 2 -> 0.4
# Main Sequence -> Star Type = 3 -> 0.6
# Supergiant    -> Star Type = 4 -> 0.8
# Hypergiant    -> Star Type = 5 -> 1.0

def get_star_type(prediction):
    if prediction < 0.1:
        return "brown_dwarf"
    if prediction < 0.3:
        return "red_dwarf"
    if prediction < 0.5:
        return "white_dwarf"
    if prediction < 0.7:
        return "main_sequence"
    if prediction < 0.9:
        return "supergigant"
    if prediction <= 1.0:
        return "hypergigant"
    return "unknown"


@svc.api(input=JSON(), output=JSON())
async def classify(star_description):
    vector = dv.transform(star_description)
    prediction = await model_runner.predict.async_run(vector)
    return {"startype": get_star_type(prediction)}
