import argparse

from ultralytics import YOLO


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self) -> None:
        super().__init__()

        platforms = [
            "rk3562",
            "rk3566",
            "rk3568",
            "rk3576",
            "rk3588",
            "rv1126b",
            "rv1109",
            "rv1126",
            "rk1808",
        ]

        self.add_argument("model_path")
        self.add_argument("-p", "--platform", default="rk3588", choices=platforms)
        self.add_argument("-o", "--output_path", default="model.rknn")


def main():
    parser = ArgumentParser()
    args = parser.parse_args()

    model_path: str = args.model_path
    platform: str = args.platform

    model = YOLO(model_path)
    model.export(format="rknn", name=platform)


if __name__ == "__main__":
    main()
