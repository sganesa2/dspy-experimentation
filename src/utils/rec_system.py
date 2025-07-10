import dspy
import os
import json
from pathlib import Path
import pandas as pd

from utils.retriever import Product, ColbertRetriever
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.joinpath(".env"))

class RecommendationSignature(dspy.Signature):
    __doc__ = """You are an AI Assistant that recommends products to users based on their queries."""
    product_query:str = dspy.InputField()
    retrieved_products: list[Product] = dspy.InputField()
    recommended_product_ids: list[int] = dspy.OutputField(desc = "Give me top 3 recommended product ids. You have to return 3 product ids")


class RecommendationSystem(dspy.Module):
    def __init__(self, k:int)->None:
        super().__init__()
        self.recommendation_system = dspy.ChainOfThought(RecommendationSignature)
        self.k = k
    
    def forward(self, product_query:str)->dspy.Prediction:
        retriever = dspy.Retrieve(self.k)
        retrieved_products = retriever(product_query)
        response = self.recommendation_system(product_query = product_query, retrieved_products = retrieved_products)
        return dspy.Prediction(
            recommended_product_ids = response.recommended_product_ids,
            reasoning = response.reasoning,
            retrieved_products = retrieved_products.passages
            )


class RecSysDataset:
    def __init__(self, json_file_name:str)->None:
        self.json_path = Path(__file__).parent.parent.joinpath(json_file_name)
        self.df = pd.read_json(self.json_path)
    
    def train(self, start:int, end:int)->list[dspy.Example]:
        return [dspy.Example(product_query = v[0], rec_id = v[1]).with_inputs("product_query") for v in self.df.values[start:end]]
    def test(self, start:int, end:int)->list[dspy.Example]:
        return [dspy.Example(product_query = v[0], rec_id = v[1]).with_inputs("product_query") for v in self.df.values[start:end]]
    def dev(self, start:int, end:int)->list[dspy.Example]:
        return [dspy.Example(product_query = v[0], rec_id = v[1]).with_inputs("product_query") for v in self.df.values[start:end]]
    



if __name__=="__main__":
    #Complete list of products to retrieve from
    with open(Path(__file__).parent.parent.joinpath("products.json"), 'r') as f:
        product_json = json.loads(f.read())
    
    #Input
    query = "2 in pvc pipe"
    k= 5

    #Model initializations
    rm = ColbertRetriever(model_name = "colbert", documents = product_json["products"])
    lm = dspy.LM(model = os.getenv("BASE_LLM"), api_key = os.getenv("GROQ_API_KEY"), temperature = 0.7)
    dspy.settings.configure(lm = lm, rm = rm)

    recsys = RecommendationSystem(k=k)
    recommended_products = recsys(product_query = query)