import dspy

def precision_at_k(example:dspy.Example, prediction:dspy.Prediction, trace = None):
    example_recs, predicted_recs = example.rec_id, prediction.recommended_product_ids
    if not trace:
        return len(set(example_recs).intersection(set(predicted_recs)))/3
    else:
        return True