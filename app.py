import os
import tempfile
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, File, UploadFile

from analyze import data_analyzer, meta_analyzor, metadata_writer
from main_sentiment import sentiment_product

from fastapi.responses import FileResponse


app = FastAPI()


@app.post("/uploadfile/", response_class=FileResponse)
# @app.post("/uploadfile/")
async def create_upload_file(files: List[UploadFile]):
    agent_conv, contact_conv = [], []
    for file in files:
        if "agent" in file.filename:
            content = await file.read()
            content = content.decode("utf-8")
            _, _, agent_conv = data_analyzer(content.splitlines())
        elif "contact" in file.filename:
            content = await file.read()
            content = content.decode("utf-8")
            _, _, contact_conv = data_analyzer(content.splitlines())

    dirpath = tempfile.mkdtemp()

    plt.plot(
        contact_conv, "o-", label=f"Contact Sentiment - Total: {np.mean(contact_conv)}"
    )
    plt.plot(agent_conv, "v-", label=f"Agent Sentiment - Total: {np.mean(agent_conv)}")
    plt.legend(loc="best")
    plt.savefig(f"{dirpath}/image.png")

    metadata_writer(
        contact_conv=contact_conv,
        agent_conv=agent_conv,
        path=f"{dirpath}/metadata.json",
    )
    # return {"list files": os.listdir(dirpath)}
    return f"{dirpath}/metadata.json"
    # return f"{dirpath}/image.png"



@app.get("/sentiment/{item_id}")
async def read_item(item_id: str, q: Union[str, None] = None):
    res_dict = {0: "Negative", 1: "Neutral", 2: "Positive"}
    if q:
        results = sentiment_product(str(q))[0]
        return {"Sentiment": res_dict[results]}

    return {"item_id": item_id}


@app.get("/")
async def index():
    return {"status": "ok"}
