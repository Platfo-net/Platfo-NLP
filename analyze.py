import json
from pathlib import Path
from pprint import pprint
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from main_sentiment import sentiment_product


def sentence_extractor(filename: str) -> List:
    if Path(filename).exists():
        f = open(filename, "r")
        Lines = f.readlines()
        sentences = []
        for line in Lines:
            sentences.append(line.strip())
        return sentences


def data_analyzer(sentences: List[str]) -> Tuple[list, list, list]:
    sentiment_label = []
    sentiment_probability = []
    whole_conv = []

    for sentence in sentences:
        p1, p2 = sentiment_product(sentence)

        if p1 != 1:  # Neutral
            p1 -= 1
            p2 = np.max(p2)

            sentiment_label.append(p1)
            sentiment_probability.append(p2)
            whole_conv.append(p1 * p2)
        else:
            second_max = [x for x in p2[0] if x != np.max(p2)]
            second_max = np.max(second_max)
            if len(np.where(p2[0] == second_max)[0]) == 1:
                p1 = np.where(p2[0] == second_max)[0][0] - 1
            else:
                p1 = 0

            sentiment_label.append(p1)
            sentiment_probability.append(p2)
            whole_conv.append(p1 * second_max)

    return sentiment_label, sentiment_probability, whole_conv


def meta_analyzor(whole_conv: list) -> Tuple[list, list]:
    neg, pos = [], []
    for val in whole_conv:
        if val < 0:
            neg.append(val)
        elif val > 0:
            pos.append(val)
    return {"negative": neg, "positive": pos}


def metadata_writer(contact_conv: list, agent_conv: list, path: str):
    contact_meta = meta_analyzor(contact_conv)
    agent_meta = meta_analyzor(agent_conv)
    metadata = {
        "Contact": {
            "positive": {
                "count": len(contact_meta["positive"]),
                "mean": np.mean(contact_meta["positive"]),
            },
            "negative": {
                "count": len(contact_meta["negative"]),
                "mean": np.mean(contact_meta["negative"]),
            },
        },
        "Agent": {
            "positive": {
                "count": len(agent_meta["positive"]),
                "mean": np.mean(agent_meta["positive"]),
            },
            "negative": {
                "count": len(agent_meta["negative"]),
                "mean": np.mean(agent_meta["negative"]),
            },
        },
    }

    metadata_json = json.dumps(metadata)
    with open(path, "w") as f:
        f.write(metadata_json)


if __name__ == "__main__":
    contact_sentences = sentence_extractor("./contact.txt")
    agent_sentences = sentence_extractor("./agent.txt")

    contact_sent_label, contact_sent_prob, contact_conv = data_analyzer(
        contact_sentences
    )
    agent_sent_label, agent_sent_prob, agent_conv = data_analyzer(agent_sentences)

    print(f"Contact: {contact_conv}")
    print(f"Agent: {agent_conv}")
    plt.plot(
        contact_conv, "o-", label=f"Contact Sentiment - Total: {np.mean(contact_conv)}"
    )
    plt.plot(agent_conv, "v-", label=f"Agent Sentiment - Total: {np.mean(agent_conv)}")
    plt.legend(loc="best")
    plt.show()

    metadata_writer(
        contact_conv=contact_conv, agent_conv=agent_conv, path="metadata.json"
    )
