import pathlib

pathlib.Path("build").mkdir(exist_ok=True)
pathlib.Path("build/test").mkdir(exist_ok=True)
pathlib.Path("build/target").mkdir(exist_ok=True)
pathlib.Path("build/target/objects").mkdir(exist_ok=True)