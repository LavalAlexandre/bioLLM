from src.classify import classify_file
from src.model.model import Model
if __name__ == "__main__":
    file = "data/hackathon-test.jsonl"
    model = Model()
    classify_file(file, model)