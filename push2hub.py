import click
from tqdm.auto import tqdm
from datasets import load_dataset


@click.command()
@click.argument("data_dir", default="data")
@click.argument("hub_id")
@click.option("--public", default=False)
def main(
    public: bool,
    data_dir: str,
    hub_id: str
    ):
    ds = load_dataset("json", data_files={
        "train": f"{data_dir}/*.json",
    })

    print(ds)

    ds.push_to_hub(hub_id, not public)


if __name__ == "__main__":
    main()