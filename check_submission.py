import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

from pypdf import PdfReader

# CONFIGURE
RUN_NOTEBOOK = True # for a sanity check that doesn't run the notebook set to False 
NOTEBOOK = "solution.ipynb"
NOTEBOOK_TIMEOUT = 600  # seconds, for the whole notebook
PLOT_EXTENSIONS = (".pdf", ".png")  # allowed formats for plot files

# Public schema for results.json: types only, no reference values.
schema = {
    "1a": {
        "gamma": "number",
        "sigma_gamma": "number",
        "chi2": "number",
        "ndof": "integer",
        "p-value": "number",
    },
    "1b": {
        "A": "file",
        "B": "file",
        "C": "file",
    },
    "1c": {
        "lower": "number",
        "upper": "number",
    },
    "2": {
        "error_vs_k": "file",
    },
    "3a": {
        "coverage": "file",
    },
    "3b": {
        "kappas": "number_list",
        "p(T<1)": "number_list",
        "t_with_68": "number_list",
    },
    "5b": {
        "coverage_calib": "file",
    },
}

def require(condition, message):
    if not condition:
        sys.exit(f"FAIL: {message}")

def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def check_plot_file(file, path):
    """Check a plot file has an allowed extension and can be read."""
    extension = file.suffix.lower()
    require(
        extension in PLOT_EXTENSIONS,
        f"{path}: file '{file}' must be one of {', '.join(PLOT_EXTENSIONS)}",
    )

    try:
        content = file.read_bytes()
    except OSError as exc:
        sys.exit(f"FAIL: {path}: cannot read file '{file}': {exc}")

    require(len(content) > 0, f"{path}: file '{file}' is empty")

    if extension == ".pdf":
        try:
            pages = len(PdfReader(file).pages)
        except Exception as exc:
            sys.exit(f"FAIL: {path}: '{file}' is not a readable PDF: {exc}")
        require(pages > 0, f"{path}: '{file}' has no pages")

    elif extension == ".png":
        require(
            content.startswith(b"\x89PNG\r\n\x1a\n"),
            f"{path}: '{file}' is not a valid PNG file",
        )


def check_format(value, specification, path="results"):
    if isinstance(specification, dict):
        require(isinstance(value, dict), f"{path} must be a JSON object")

        missing = set(specification) - set(value)
        extra = set(value) - set(specification)
        require(
            not missing and not extra,
            f"{path}: missing keys {sorted(missing)}; "
            f"unexpected keys {sorted(extra)}",
        )

        for key, child_specification in specification.items():
            check_format(
                value[key],
                child_specification,
                f"{path}[{key!r}]",
            )

    elif specification == "number":
        require(is_number(value), f"{path} must be a finite number")

    elif specification == "integer":
        require(type(value) is int, f"{path} must be an integer")

    elif specification == "number_list":
        require(isinstance(value, list), f"{path} must be a list")
        for index, item in enumerate(value):
            require(
                is_number(item),
                f"{path}[{index}] must be a finite number",
            )

    elif specification == "file":
        require(
            isinstance(value, str) and bool(value.strip()),
            f"{path} must contain a filename",
        )

        filename = Path(value)
        root = Path.cwd().resolve()
        candidates = [filename, Path("results") / filename]

        found = None
        if not filename.is_absolute():
            for candidate in candidates:
                if (
                    candidate.resolve().is_relative_to(root)
                    and candidate.is_file()
                ):
                    found = candidate
                    break

        require(
            found is not None,
            f"{path}: file {value!r} must exist in the "
            "current directory or results/",
        )
        check_plot_file(found, path)


    else:
        raise ValueError(f"Unknown schema type: {specification}")


def check_results_format(data):
    check_format(data, schema)

    columns = ("kappas", "p(T<1)", "t_with_68")
    lengths = [len(data["3b"][column]) for column in columns]

    require(
        len(set(lengths)) == 1,
        "3b: kappas, p(T<1), and t_with_68 must have equal lengths",
    )

def run_notebook(results):
    """Run the notebook and check it writes a new results file."""
    notebook = Path(NOTEBOOK)
    require(notebook.is_file(), f"Missing {NOTEBOOK}")

    # Note when any existing results file was last modified, so we can
    # tell afterwards whether the notebook rewrote it.
    if results.is_file():
        modified_before = results.stat().st_mtime_ns
    else:
        modified_before = None

    # Execute a copy written to a temporary directory so the submitted
    # notebook is not modified. Cells run in the notebook's own directory.
    with tempfile.TemporaryDirectory() as tmp:
        command = [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute", str(notebook),
            "--output-dir", tmp,
            "--ExecutePreprocessor.kernel_name=python3",
        ]
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=NOTEBOOK_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            sys.exit(
                f"FAIL: {NOTEBOOK} did not finish within "
                f"{NOTEBOOK_TIMEOUT} seconds"
            )

    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-30:])
        sys.exit(f"FAIL: {NOTEBOOK} failed to run:\n{tail}")

    require(
        results.is_file(),
        f"{NOTEBOOK} did not produce {results}",
    )
    require(
        results.stat().st_mtime_ns != modified_before,
        f"{NOTEBOOK} ran but did not rewrite {results}",
    )
    print(f"PASS: {NOTEBOOK} ran and wrote {results}")


if __name__ == "__main__":
    
    # Check a README.md exists
    readme = Path("README.md")
    require(readme.is_file(), "Missing README.md")
    print("PASS: README.md exists")


    # Check the submitted report.
    report = Path("report/report.pdf")
    require(report.is_file(), "Missing report/report.pdf")

    try:
        reader = PdfReader(report)
        require(len(reader.pages) > 0, "report.pdf has no pages")
    except Exception as exc:
        sys.exit(f"FAIL: Cannot read report.pdf: {exc}")
    
    print("PASS: report/report.pdf exists and is readable")

    # Optionally run the notebook, which must regenerate the results file.
    results = Path("results/results.json")
    if RUN_NOTEBOOK:
        print(f"WAIT: running notebook {NOTEBOOK}...")
        run_notebook(results)

    # Check the results path 
    require(results.is_file(), "Missing results/results.json")

    try:
        with open(results) as f:
            data = json.load(f)
    except Exception as exc:
        sys.exit(f"FAIL: Cannot read results/results.json: {exc}")

    check_results_format(data)
    print("PASS: results/results.json exists and is in correct format")

