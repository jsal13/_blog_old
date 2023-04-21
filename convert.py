import re
import subprocess
import sys
from pathlib import Path


def convert_nb_to_md(target_notebook: str) -> None:
    """Convert notebook at ``target_notebook`` path to an MD file in ``./_posts``."""

    # Read notebook and parse out date and nb name.
    nb_name = Path(target_notebook).stem
    date_pattern = r"date:.*?(\d{4}-\d{2}-\d{2})"

    with open(target_notebook, "r", encoding="utf-8") as f:
        f_raw_nb = f.read()

    datetime_val = re.findall(date_pattern, f_raw_nb)[0]
    md_path = f"./_posts/{datetime_val}-{nb_name}.md"

    # Create and run command to convert nb to markdown.
    # Note that the output is a path relative to the input, ugh.
    cmd = [
        "jupyter",
        "nbconvert",
        "--log-level",
        "WARN",
        "--to",
        "markdown",
        target_notebook,
        "--output",
        f"../{md_path}",
    ]

    subprocess.run(cmd, check=True)

    # Convert assets to site-ready assets.
    with open(md_path, "r", encoding="utf-8") as f:
        f_raw_md = f.read()

    nb_asset_pattern = r"!\[\]\(./(assets/images/.*?)\)"
    md_asset_pattern = r"![]({{site.baseurl}}/\1)"

    asset_replacements = re.findall(nb_asset_pattern, f_raw_md)

    if asset_replacements:
        print(f"> All asset replacements: {asset_replacements}")
    else:
        print("> No assets replaced.")

    f_raw_md = re.sub(nb_asset_pattern, md_asset_pattern, f_raw_md)

    # Re-write results back to md file.
    with open(md_path, "w+", encoding="utf-8") as f:
        f.write(f_raw_md)


if __name__ == "__main__":
    convert_nb_to_md(sys.argv[1])
