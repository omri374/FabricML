import os
import uuid
from pathlib import Path

import click
import jupytext


# Script for generating a notebook based on a markdown file.
# Example run: python generate_notebook.py --name my_notebook.ipynb


@click.command(
    name="Generate a new notebook based on a markdown template "
    "(found in notebook_templates/notebook_template.md"
)
@click.option("--name", help="The name of the notebook to be generated")
@click.option(
    "--path",
    help="Folder in which the notebook will be generated in",
    default="notebooks",
)
@click.option(
    "--template_file",
    help="Template to generate notebook from (from the notebook_templates folder)",
    default="notebook_template.md",
)
@click.option("--file_type", help="Notebook file type (e.g. .ipynb)", default="ipynb")
def generate_notebook(template_file, name, path="notebooks", file_type="ipynb"):
    # Read a notebook from a file

    md_notebook = jupytext.read(Path("notebook_templates", template_file))

    # If name was not provided, create a random name
    if not name:
        name = f"notebook-{uuid.uuid4()}"
        print(f"Name not provided, generating random name: {name}")

    # add ipynb if missing
    if file_type not in name:
        name += "." + file_type

    if os.path.isabs(name):
        # Notebook path is absolute, no need to add full path
        output_path = name
    else:
        # set full path
        current_dir_path = os.path.dirname(os.path.realpath(__file__))
        absolute_notebooks_path = str(Path(current_dir_path, path).resolve())
        output_path = str(Path(absolute_notebooks_path, name).resolve())
    print(f"Saving notebook {name} to {output_path}")
    jupytext.write(md_notebook, output_path)


if __name__ == "__main__":
    generate_notebook()
