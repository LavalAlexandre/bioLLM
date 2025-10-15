from src.classify import classify_file
from src.model.model import Model

if __name__ == "__main__":
    file = "data/hackathon-test.jsonl"
    # for now turn off biorxiv
    model = Model(enable_biorxiv=False)
    classify_file(file, model, use_agent=True)
