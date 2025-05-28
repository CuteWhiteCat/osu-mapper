import hydra

from config import InferenceConfig
from inference import load_model
from osuT5.osuT5.model import Mapperatorinator
from osuT5.osuT5.tokenizer import Tokenizer


@hydra.main(config_path="configs", config_name="inference_v29", version_base="1.1")
def main(args: InferenceConfig):
    model_name = "OliBomby/Mapperatorinator-v29.1"

    model, tokenizer = load_model(args.model_path, args.osut5, args.device)

    model.push_to_hub(model_name, private=True)
    tokenizer.push_to_hub(model_name, private=True)

    model = Mapperatorinator.from_pretrained(model_name)
    tokenizer = Tokenizer.from_pretrained(model_name)

    print("Done")

if __name__ == "__main__":
    main()
