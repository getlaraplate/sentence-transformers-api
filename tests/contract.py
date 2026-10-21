"""A small checker of JSON values against the schemas of openapi.json, without a validator library.

It covers what the document uses: $ref, type (also a list, for nullable values), required, properties,
additionalProperties as a schema, items, enum and const.
"""

import json
from pathlib import Path

DOCUMENT = json.loads((Path(__file__).resolve().parent.parent / "openapi.json").read_text())

JSON_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def resolve(schema):
    while "$ref" in schema:
        node = DOCUMENT
        for part in schema["$ref"].removeprefix("#/").split("/"):
            node = node[part]
        schema = node
    return schema


def type_matches(expected, value):
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, JSON_TYPES[expected])


def problems(schema, value, where="$"):
    """The list of places where value does not match schema; empty when it matches."""
    schema = resolve(schema)
    found = []
    expected = schema.get("type")
    if expected is not None:
        allowed = expected if isinstance(expected, list) else [expected]
        if not any(type_matches(kind, value) for kind in allowed):
            return [f"{where}: expected {expected}, got {type(value).__name__}"]
    if "const" in schema and value != schema["const"]:
        found.append(f"{where}: expected {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        found.append(f"{where}: {value!r} not in {schema['enum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                found.append(f"{where}: missing {key}")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                found += problems(properties[key], item, f"{where}.{key}")
            elif isinstance(schema.get("additionalProperties"), dict):
                found += problems(schema["additionalProperties"], item, f"{where}.{key}")
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            found += problems(schema["items"], item, f"{where}[{index}]")
    return found


def response_schema(path, method, status):
    return DOCUMENT["paths"][path][method]["responses"][str(status)]["content"]["application/json"]["schema"]
