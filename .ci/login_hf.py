from argparse import ArgumentParser
from huggingface_hub.commands.user import _login
from huggingface_hub.hf_api import HfApi

def main():
    parser = ArgumentParser()
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    _login(hf_api=HfApi(), token=args.token)

if __name__ == "__main__":
    main()
