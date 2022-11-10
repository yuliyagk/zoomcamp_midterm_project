import numpy as np

import bentoml
from bentoml.io import JSON

model_ref= bentoml.sklearn.get("star_type_model_skl:bk5qpxtbg2too4yo")

dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()
svc = bentoml.Service("startype", runners=[model_runner])

# Brown Dwarf   -> Star Type = 0
# Red Dwarf     -> Star Type = 1
# White Dwarf   -> Star Type = 2
# Main Sequence -> Star Type = 3
# Supergiant    -> Star Type = 4
# Hypergiant    -> Star Type = 5

def get_star_type(prediction):
    if prediction == 0:
        return "brown_dwarf"
    if prediction == 1:
        return "red_dwarf"
    if prediction == 2:
        return "white_dwarf"
    if prediction == 3:
        return "main_sequence"
    if prediction == 4:
        return "supergigant"
    if prediction == 5:
        return "hypergigant"
    return "unknown"


@svc.api(input=JSON(), output=JSON())
async def classify(star_description):
    vector = dv.transform(star_description)
    prediction = await model_runner.predict.async_run(vector)
    return {"startype": get_star_type(prediction)}
