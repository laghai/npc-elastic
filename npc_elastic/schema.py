"""JSON Schema export. `python -m npc_elastic --schema elastic/schema` rewrites the schemas."""

from __future__ import annotations

import json
from pathlib import Path

from .spec import ElasticResult, ElasticSpec

SCHEMAS = {
    "elastic-spec.v0.schema.json": (
        ElasticSpec,
        "https://github.com/laghai/nanophotonics-cloud/elastic/spec/0",
    ),
    "elastic-result.v0.schema.json": (
        ElasticResult,
        "https://github.com/laghai/nanophotonics-cloud/elastic/result/0",
    ),
}


def schema_text(name: str) -> str:
    model, uri = SCHEMAS[name]
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": uri}
    schema.update(model.model_json_schema(by_alias=True))
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def write_all(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in SCHEMAS:
        p = out / name
        p.write_text(schema_text(name), encoding="utf-8")
        paths.append(p)
    return paths
