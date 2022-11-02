from argparse import ArgumentParser
from huggingface_hub import HfApi
from huggingface_hub.commands.user import _login
from huggingface_hub import HfFolder


def main():
    parser = ArgumentParser()
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    _login(HfApi(), token=args.token)
    HfFolder.save_token(args.token)

if __name__ == "__main__":
    main()
