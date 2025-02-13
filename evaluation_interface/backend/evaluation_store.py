import os
import json
from datetime import datetime

def store_evaluation(username, evaluation_data):
    """
    Write the evaluator's submission to a file in results/evaluation.
    The filename uses the evaluator’s username and a timestamp.
    """
    evaluation_dir = "results/evaluation"
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{username}_{timestamp}.json"
    file_path = os.path.join(evaluation_dir, filename)
    try:
        with open(file_path, "w") as f:
            json.dump(evaluation_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error storing evaluation: {e}")
        return False
