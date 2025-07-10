import dspy
import json
from rerankers import Reranker
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Product:
    id: int
    long_text:str

class ColbertRetriever(dspy.Retrieve):
    def __init__(self, model_name:str, documents:list[str])->None:
        self.retriver = Reranker(model_name=model_name)
        self.docs = documents
        
    def forward(self, query:str, k:int)->list[Product]:
        ranked_products = self.retriver.rank(query=query, docs= self.docs, doc_ids=list(range(1,len(self.docs)+1)))
        top_k_products = ranked_products.top_k(k=k)
        return [Product(id=product.document.doc_id, long_text=product.document.text) for product in top_k_products]
    

if __name__ == "__main__":
    #Complete list of products to retrieve from
    with open(Path(__file__).parent.parent.joinpath("products.json"), 'r') as f:
        product_json = json.loads(f.read())
    
    #Input
    query = "2 in pvc pipe"
    k= 5

    retriever = ColbertRetriever("colbert", product_json["products"])
    top_k_products = retriever(query=query, k=k)
    