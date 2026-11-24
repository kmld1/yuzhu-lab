"""Minimal JSON Schema validation for tool arguments.

Only the keywords the tool registry actually uses are implemented.  The
validator is strict on purpose: unknown argument names are rejected when the
schema says ``additionalProperties: false`` so a model cannot smuggle extra
fields into an executor.
"""

from __future__ import annotations

from typing import Any

_TYPE_NAMES = ("object", "array", "string", "integer", "number", "boolean", "null")


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def validate(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    if expected:
        options = expected if isinstance(expected, list) else [expected]
        if not any(_type_matches(instance, option) for option in options):
            errors.append(f"{path}: expected {'|'.join(options)}")
            return errors

    enum = schema.get("enum")
    if enum is not None and instance not in enum:
        errors.append(f"{path}: value must be one of {enum}")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < int(schema["minLength"]):
            errors.append(f"{path}: string shorter than {schema['minLength']}")
        if "maxLength" in schema and len(instance) > int(schema["maxLength"]):
            errors.append(f"{path}: string longer than {schema['maxLength']}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")

    if isinstance(instance, dict):
        properties = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in instance:
                errors.append(f"{path}: missing required property '{name}'")
        if schema.get("additionalProperties") is False:
            for name in instance:
                if name not in properties:
                    errors.append(f"{path}: unexpected property '{name}'")
        for name, subschema in properties.items():
            if name in instance:
                errors.extend(validate(instance[name], subschema, f"{path}.{name}"))

    if isinstance(instance, list):
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(instance):
                errors.extend(validate(item, item_schema, f"{path}[{index}]"))
        if "maxItems" in schema and len(instance) > int(schema["maxItems"]):
            errors.append(f"{path}: more than {schema['maxItems']} items")

    return errors


def object_schema(
    properties: dict[str, Any],
    *,
    required: tuple[str, ...] = (),
    additional: bool = False,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": additional,
    }