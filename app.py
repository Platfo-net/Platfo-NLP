from typing import Union

from fastapi import FastAPI

from main_sentiment import sentiment_product, SentimentModel

app = FastAPI()


res_dict = {0: "Negative", 1: "Neutral", 2: "Positive"}


@app.get("/{item_id}")
async def read_item(item_id: str, q: Union[str, None] = None):
    if q:
        results = sentiment_product(str(q))

        return {"Sentiment": res_dict[results]}

    return {"item_id": item_id}
