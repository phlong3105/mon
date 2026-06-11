"""
We have a concept in Hafnia Training-aaS called the "Command Builder" that helps the user construct
a CLI command that will be passed to and executed on the trainer.

An example of this could be `python scripts/train.py --batch-size 32`, that tells the trainer to
launch the `scripts/train.py` python script and set the `batch_size` parameter to `32`.

The "Command Builder" is a dynamic form-based interface that is specific for each training script and
should contain available parameters that the script accepts. Each parameter is described by
its name and type and optionally parameter title, description, and default value.

The CommandBuilderSchema is a json description/configuration for creating a command builder form.
This python module provides functionality for constructing or automatically generating such a schema from
a CLI function (e.g. the `main` function in `scripts/train.py`), and save it as a JSON file.

The convention is that if the CLI function is defined in `./scripts/train.py`, the corresponding
CommandBuilderSchema JSON file should be saved as `./scripts/train.json`.
"""

from __future__ import annotations

import enum
import inspect
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    get_args,
    get_origin,
)

import docstring_parser
import flatten_dict
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

DEFAULT_CASE_CONVERSION: Literal["none", "kebab"] = "none"
DEFAULT_PARAMETER_PREFIX: str = "--"
DEFAULT_NESTED_PARAMETER_HANDLING: Literal["dot", "not-supported"] = "dot"
DEFAULT_ASSIGNMENT_SEPARATOR: Literal["space", "equals"] = "space"
DEFAULT_BOOL_HANDLING: Literal["none", "flag-negation"] = "none"
DEFAULT_ORDER: int = 100


def auto_save_command_builder_schema(
    cli_function: Callable[..., Any],
    name: Optional[str] = None,
    positional_args: Optional[List[str]] = None,
    cli_tool: Optional[str] = "cyclopts",
    ignore_params: tuple[str, ...] = (),
    handle_union_types: bool = True,
    path_schema: Optional[Path] = None,
    cmd: Optional[str] = None,
    case_conversion: Optional[Literal["none", "kebab"]] = None,
    parameter_prefix: Optional[str] = None,
    nested_parameter_handling: Optional[Literal["dot", "not-supported"]] = None,
    assignment_separator: Optional[Literal["space", "equals"]] = None,
    bool_handling: Optional[Literal["none", "flag-negation"]] = None,
    order: int = DEFAULT_ORDER,
) -> Path:
    """
    Magic function to create and save CommandBuilderSchema as JSON file from a CLI function.
    If the function is invoked in e.g. 'scripts/train.py' it will create a JSON schema file
    'scripts/train.json' next to the script file.

    The the auto-generated schema might not work for all CLI functions and settings. In that case,
    you can manually create the CommandBuilderSchema using the 'CommandBuilderSchema.from_function(...)'
    method and save it using the 'to_json_file(...)' method. Or directly create the CommandBuilderSchema
    by providing the 'json_schema' argument as shown in the example below.

    ```python
    function_schema = schema_from_cli_function(main_cli_function)
    command_builder = CommandBuilderSchema(
        name="Trainer",
        cmd="python scripts/train.py",
        json_schema=function_schema,
        case_conversion="kebab",
        parameter_prefix="--",
        nested_parameter_handling="dot",
        assignment_separator="space",
        bool_handling="flag-negation",
    )
    command_builder.to_json_file(Path("scripts/train.json"))

    """

    launch_schema: CommandBuilderSchema = CommandBuilderSchema.from_function(
        cli_function,
        name=name,
        cli_tool=cli_tool,
        ignore_params=ignore_params,
        handle_union_types=handle_union_types,
        cmd=cmd,
        positional_args=positional_args,
        case_conversion=case_conversion,
        parameter_prefix=parameter_prefix,
        nested_parameter_handling=nested_parameter_handling,
        assignment_separator=assignment_separator,
        bool_handling=bool_handling,
        order=order,
    )

    path_schema = path_schema or path_of_function(cli_function).with_suffix(".schema.json")
    launch_schema.to_json_file(path_schema)
    return path_schema


class CommandBuilderSchema(BaseModel):
    """
    A schema or configuration for the Command Builder Form. It is a data model that describes
    the command to execute, which parameters are available for the command, and how the command line
    arguments should be formatted.

    The 'CommandBuilderSchema' data model can be constructed manually by providing the 'cmd' and
    'json_schema' arguments, or it can be automatically generated from a CLI function using the
    'from_function(...)' method.

    """

    name: str = Field(
        ...,
        description=("The name of the command builder schema."),
    )

    cmd: str = Field(
        ...,
        description=(
            "The command to execute the trainer launcher 'python scripts/train.py', "
            "'uv run scripts/train.py', 'trainer-classification'"
        ),
    )

    json_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema representing the command parameters. ",
    )

    # Below defines how the specified parameters/arguments are formatted as command line arguments
    case_conversion: Literal["none", "kebab"] = Field(
        DEFAULT_CASE_CONVERSION,
        description=(
            "CLI format: The command line arguments will use kebab-case (e.g. --batch-size). "
            "This handles what is commonly done in CLI tools where function argument names are converted from "
            "'snake_case' to '--kebab-case' when used as command line arguments."
        ),
    )
    parameter_prefix: str = Field(
        DEFAULT_PARAMETER_PREFIX,
        description="CLI format: The prefix used for command line arguments. E.g. '--' for '--batch-size' ",
    )

    nested_parameter_handling: Literal["dot", "not-supported"] = Field(
        DEFAULT_NESTED_PARAMETER_HANDLING,
        description=(
            "CLI format: Separator used for nested parameters in the command line interface. "
            "E.g. 'optimizer.lr' to set the learning rate inside an optimizer configuration."
        ),
    )

    assignment_separator: Literal["space", "equals"] = Field(
        DEFAULT_ASSIGNMENT_SEPARATOR,
        description=(
            "CLI format: Separator between parameter names and their values in the command line. "
            "E.g. ' ' for '--batch-size 32' or '=' for '--batch-size=32'."
        ),
    )

    bool_handling: Literal["none", "flag-negation"] = Field(
        DEFAULT_BOOL_HANDLING,
        description=(
            "CLI format: How boolean parameters are handled in the command line. "
            "'none' means boolean flags must be explicitly set to True or False "
            "e.g. '--flag True' or '--flag False'. "
            "'flag-negation' uses flag negation where '--flag' sets True and '--no-flag' sets False."
        ),
    )

    order: int = Field(
        DEFAULT_ORDER,
        description=(
            "The order of the command builder in the UI. Command builders with lower order are displayed first. "
        ),
    )

    @property
    def positional_args(self) -> List[str]:
        """Names of parameters (from 'json_schema') that are passed as positional command line arguments."""
        return list(self.json_schema.get("positional_args", []))

    def to_json_file(self, path: Path) -> Path:
        """Write the launcher schema to a JSON file.

        Args:
            path (Path): The file path to write the schema to.
        """
        if not path.suffix == ".json":
            raise ValueError(f"Expected a .json file path, got: {path}")

        json_str = self.model_dump_json(indent=4)
        path.write_text(json_str)
        return path

    @model_validator(mode="after")
    def validate_json_schema(self) -> "CommandBuilderSchema":
        # Basic validation of the json_schema field to ensure it has 'properties'
        if "properties" not in self.json_schema:
            raise ValueError("The 'json_schema' must contain a 'properties' field.")

        if "$defs" in self.json_schema and self.nested_parameter_handling == "not-supported":
            raise ValueError(
                "The 'json_schema' contains '$defs' indicating nested parameters, "
                "but 'nested_parameter_handling' is set to 'not-supported'. "
                "Either remove nested parameters from the schema or change "
                "'nested_parameter_handling' to support nested parameters."
            )

        # Every name listed in 'positional_args' must refer to an actual parameter in 'properties'.
        properties = self.json_schema.get("properties", {})
        unknown_positional = [name for name in self.positional_args if name not in properties]
        if unknown_positional:
            raise ValueError(
                f"The 'positional_args' contain names that are not parameters in the 'json_schema' "
                f"properties: {unknown_positional}. Available parameters: {list(properties)}."
            )
        return self

    @staticmethod
    def from_json_file(path: Path) -> "CommandBuilderSchema":
        json_str = path.read_text()
        return CommandBuilderSchema.model_validate_json(json_str)

    @staticmethod
    def from_function(
        cli_function: Callable[..., Any],
        name: Optional[str] = None,
        cli_tool: Optional[str] = "cyclopts",
        ignore_params: tuple[str, ...] = (),
        cmd: Optional[str] = None,
        handle_union_types: bool = True,
        positional_args: Optional[List[str]] = None,
        case_conversion: Optional[Literal["none", "kebab"]] = None,
        parameter_prefix: Optional[str] = None,
        nested_parameter_handling: Optional[Literal["dot", "not-supported"]] = None,
        assignment_separator: Optional[Literal["space", "equals"]] = None,
        bool_handling: Optional[Literal["none", "flag-negation"]] = None,
        order: int = DEFAULT_ORDER,
    ) -> "CommandBuilderSchema":
        cmd = cmd or f"python {path_of_function(cli_function).as_posix()}"
        if cli_tool == "cyclopts":
            set_case_conversion = "kebab"
            set_parameter_prefix = "--"
            set_nested_parameter_handling = "dot"
            set_assignment_separator = "space"
            set_bool_handling = "flag-negation"
        elif cli_tool == "typer":
            set_case_conversion = "kebab"
            set_parameter_prefix = "--"
            set_nested_parameter_handling = "not-supported"
            set_assignment_separator = "space"
            set_bool_handling = "flag-negation"
        else:
            raise ValueError(f"Unsupported cli_tool: {cli_tool}")
        name = name or path_of_function(cli_function).as_posix()

        # Option to override individual settings
        set_case_conversion = case_conversion or set_case_conversion or DEFAULT_CASE_CONVERSION
        set_parameter_prefix = parameter_prefix or set_parameter_prefix or DEFAULT_PARAMETER_PREFIX
        set_nested_parameter_handling = (
            nested_parameter_handling or set_nested_parameter_handling or DEFAULT_NESTED_PARAMETER_HANDLING
        )
        set_assignment_separator = assignment_separator or set_assignment_separator or DEFAULT_ASSIGNMENT_SEPARATOR
        set_bool_handling = bool_handling or set_bool_handling or DEFAULT_BOOL_HANDLING
        function_schema = schema_from_cli_function(
            cli_function,
            ignore_params=ignore_params,
            handle_union_types=handle_union_types,
        )
        # Store the positional arguments inside the json_schema so the information travels with the schema.
        # This is not a standard JSON schema field, but it is used by our Command Builder implementation to
        # determine which arguments are positional.
        function_schema["positional_args"] = positional_args or []
        return CommandBuilderSchema(
            name=name,
            cmd=cmd,
            json_schema=function_schema,
            case_conversion=set_case_conversion,
            parameter_prefix=set_parameter_prefix,
            nested_parameter_handling=set_nested_parameter_handling,
            assignment_separator=set_assignment_separator,
            bool_handling=set_bool_handling,
            order=order,
        )

    def command_args_from_form_data(self, form_data: Dict[str, Any], include_cmd: bool = True) -> List[str]:
        """Convert form data dictionary to command line arguments list.

        Args:
            form_data (Dict[str, Any]): The form data dictionary.

        Returns:
            List[str]: The command line arguments list.
        """

        # Flatten nested parameters ({"optimizer": {"lr": 0.01}} -> {"optimizer.lr": 0.01})
        if self.nested_parameter_handling == "dot":
            sep = "."
            form_data = flatten_dict.flatten(form_data)  # Flatten doesn't support str directly
            form_data = {sep.join(keys): value for keys, value in form_data.items()}
        elif self.nested_parameter_handling == "none":
            pass  # No special handling
        elif self.nested_parameter_handling == "not-supported":
            # Ensure there are no nested parameters
            for value in form_data.values():
                if isinstance(value, dict):
                    raise ValueError("Nested parameters are not supported, but found nested parameter in form data.")
        else:
            raise ValueError(f"Unsupported nested_parameter_handling: {self.nested_parameter_handling}")

        cmd_args: List[str] = []

        if include_cmd:
            cmd_args.extend(self.cmd.split(" "))

        # Emit positional arguments first - in the order listed in 'positional_args' - then the remaining named
        # arguments in form-data order. Positional names not present in the form data are skipped.
        positional_names = [name for name in self.positional_args if name in form_data]
        named_names = [name for name in form_data if name not in positional_names]
        for name in [*positional_names, *named_names]:
            is_positional = name in positional_names
            arg_parts = name_and_value_as_command_line_arguments(
                name=name,
                value=form_data[name],
                is_positional=is_positional,
                case_conversion=self.case_conversion,
                parameter_prefix=self.parameter_prefix,
                assignment_separator=self.assignment_separator,
                bool_handling=self.bool_handling,
            )
            cmd_args.extend(arg_parts)
        return cmd_args


def name_and_value_as_command_line_arguments(
    name: str,
    value: Any,
    is_positional: bool,
    case_conversion: Literal["none", "kebab"],
    parameter_prefix: str,
    assignment_separator: Literal["space", "equals"],
    bool_handling: Literal["none", "flag-negation"],
) -> List[str]:
    # Uses 'repr' (instead of 'str') to convert values. This is is important when handling string parameter
    # values that include space. E.g. this '--some_str Some String' would not work and
    # should be "--some_str 'Some String'" to be parsed correctly by the CLI.
    value_str = repr(value)

    if case_conversion == "kebab":
        name = snake_case_to_kebab_case(name)
    elif case_conversion == "none":
        pass  # No conversion
    else:
        raise ValueError(f"Unsupported case_conversion: {case_conversion}")

    # Handle boolean flags with flag-negation
    if isinstance(value, bool):
        if bool_handling == "flag-negation":
            if value is False:
                name = f"no-{name}"  # E.g. '--no-flag' for False
            return [f"{parameter_prefix}{name}"]
        elif bool_handling == "none":
            pass  # No special handling
        else:
            raise ValueError(f"Unsupported bool_handling: {bool_handling}")

    name = f"{parameter_prefix}{name}"

    if is_positional:
        return [value_str]

    if assignment_separator == "equals":  # Is combined into one arg
        return [f"{name}={value_str}"]
    elif assignment_separator == "space":  # Is separated into two args
        return [name, value_str]
    else:
        raise ValueError(f"Unsupported assignment_separator: {assignment_separator}")


def path_of_function(cli_function: Callable[..., Any]) -> Path:
    """Get the default path for saving the launch schema JSON file from a CLI function."""
    path_script = Path(cli_function.__globals__["__file__"])
    script_relative_path = Path(path_script).relative_to(Path.cwd())
    return script_relative_path


def simulate_form_data(function: Callable[..., Any], user_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    The purpose of this module is to simulate the output of a form submission
    from a user.

    The function is mostly used for testing and validation purposes.
    """
    pydantic_model = pydantic_model_from_cli_function(function)
    cli_args = pydantic_model(**user_args)  # Validate args
    return cli_args.model_dump(mode="json")


def schema_from_cli_function(
    func: Callable[..., Any],
    ignore_params: tuple[str, ...] = (),
    handle_union_types: bool = True,
) -> Dict:
    """
    Generate a function schema from a cli function signature.

    This function creates a json schema based on the parameters of the provided
    function, preserving type annotations, default values, and descriptions from
    Annotated types. It then returns the JSON schema representation of that model.

    Args:
        func (Callable[..., Any]): The function to generate the schema from.
        convert_to_kebab_case (bool, optional): Whether to convert parameter names
            from snake_case to kebab-case in the resulting schema. Defaults to True.
        ignore_params (tuple[str, ...], optional): Parameters to ignore when generating the schema. Defaults to ().

    Returns:
        BaseModel: The Pydantic model representing the function's parameters.
    """

    pydantic_model: Type[BaseModel] = pydantic_model_from_cli_function(func, ignore_params=ignore_params)
    function_schema = pydantic_model.model_json_schema()
    if handle_union_types:
        function_schema = convert_union_types_in_schema(function_schema)
    return function_schema


def convert_union_types_in_schema(schema: Dict) -> Dict:
    """
    Convert 'anyOf' parameters (typically from 'Union[...]' or 'Optional[...]') in the JSON schema
    E.g.

    Example 1:
        The python type:
            Optional[int]

        Will be represented in the schema as:
        {
            "anyOf": [
                {"type": "integer"},
                {"type": "null"}
            ]
        }
        We will improve form readability by converting it to:
        {
            "anyOf": [
                {
                    "title": "Select Integer",
                    "type": "integer"
                },
                {
                    "title": "Select None",
                    "type": "null"
                }
            ],
        }

    Example 2:
    Python type:
        float_or_int: Annotated[float | int, "A float or int value"] = 3.14,
    Schema before conversion:
        {
            "anyOf": [
                {"type": "number"},
                {"type": "integer"}
            ]
        }
    Schema after conversion:
        {
            "anyOf": [
                {
                    "title": "Select Number",
                    "type": "number"
                },
                {
                    "title": "Select Integer",
                    "type": "integer"
                }
            ],
        },

    Args:
        schema (Dict): The JSON schema dictionary.

    Returns:
        Dict: The modified JSON schema dictionary with union types handled.
    """

    def type_to_title(type_name: str) -> str:
        """Convert JSON schema type to a display title"""
        if type_name == "null":
            return "Select None"
        return f"Select '{type_name.capitalize()}'"

    def add_titles_to_anyof(obj: Any) -> Any:
        """Recursively add titles to anyOf items"""
        if isinstance(obj, dict):
            converted: Dict[str, Any] = {}
            for key, value in obj.items():
                if key == "anyOf" and isinstance(value, list):
                    # Add titles to each item in anyOf
                    converted[key] = []
                    for item in value:
                        if isinstance(item, dict) and "type" in item:
                            new_item = item.copy()
                            if "title" not in new_item:
                                new_item["title"] = type_to_title(item["type"])
                            converted[key].append(new_item)
                        else:
                            converted[key].append(add_titles_to_anyof(item))
                else:
                    converted[key] = add_titles_to_anyof(value)
            return converted
        elif isinstance(obj, list):
            return [add_titles_to_anyof(item) for item in obj]
        else:
            return obj

    return add_titles_to_anyof(schema)


def pydantic_model_from_cli_function(
    func: Callable[..., Any],
    *,
    model_name: Optional[str] = None,
    ignore_params: tuple[str, ...] = (),
    forbid_varargs: bool = True,
    forbid_positional_only: bool = True,
    use_docstring_as_description: bool = True,
    config: Optional[ConfigDict] = None,
) -> type[BaseModel]:
    """
    Create a Pydantic BaseModel from a function signature.

    Preserves Annotated[...] metadata (e.g. Field constraints), because the
    parameter annotation is forwarded unchanged into pydantic.create_model().

    Notes:
    - Parameters must be type-annotated.
    - Required parameters become required model fields.
    - Defaults become model defaults.
    """
    sig = inspect.signature(func)

    # Collect fields as: name -> (annotation, default)
    fields: dict[str, tuple[Any, Any]] = {}

    params = list(sig.parameters.values())

    if len(ignore_params) > 0:
        params_reduced_set = []
        for param in params:
            if param.name in ignore_params:
                continue
            params_reduced_set.append(param)
        params = params_reduced_set

    # Extract docstring parameter descriptions and add them to Field(...) if not already present.
    function_docstring = (func.__doc__ or "").strip()
    parsed_doc_string = docstring_parser.parse(function_docstring)
    docstring_params = {p.arg_name: p.description for p in parsed_doc_string.params}

    for param in params:
        # Reject both '*args' and '**kwargs' arguments
        if forbid_varargs and param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"{func.__name__}: varargs (*args/**kwargs) are not supported for model generation "
                f"(parameter '{param.name}')"
            )
        # Reject positional-only parameters. E.g. for 'def func(var0, /, var1)' - 'var0' is positional-only.
        if forbid_positional_only and param.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"{func.__name__}: positional-only parameters are not supported for model generation "
                f"(parameter '{param.name}')"
            )
        # Ensure parameter has a type annotation 'param: int'
        if param.annotation is inspect.Parameter.empty:
            raise TypeError(
                f"{func.__name__}: parameter '{param.name}' must have a type annotation ('{param.name}: [TYPE]'). "
                f"An example of this would be '{param.name}: int' or '{param.name}: Optional[str] = None'."
            )

        base_type, field_info = function_param_as_type_and_field_info(param)

        # if issubclass(base_type, enum.Enum):
        if type(base_type) is enum.EnumMeta:
            raise TypeError(
                f"'Enum' CLI argument types are not supported yet in command builder schema generation.\n"
                f"Found that '{param.name}' uses enum type.\n"
                f"As a workaround, use 'Literal[...]' or 'str' type instead if possible for parameter '{param.name}'. "
                f"If this is not an option you may simply ignore this parameter from the schema generation "
                f"using the 'ignore_params' argument."
            )
        default = derive_default_value(field_info, param)
        if field_info.description is None and param.name in docstring_params:
            field_info_attributes = field_info.asdict()["attributes"]
            field_info_attributes["description"] = docstring_params[param.name]
            field_info = Field(**field_info_attributes)
        assert isinstance(field_info, FieldInfo), "Expected FieldInfo in Annotated metadata"
        annotation = Annotated[base_type, field_info]  # type: ignore[valid-type]
        fields[param.name] = (annotation, default)
    name = model_name or f"{func.__name__}Args"

    model_kwargs: dict[str, Any] = {}
    if config is not None:
        model_kwargs["__config__"] = config
    if use_docstring_as_description and function_docstring:
        model_kwargs["__doc__"] = inspect.getdoc(func)

    pydantic_model = create_model(name, **model_kwargs, **fields)

    return pydantic_model


def snake_case_to_kebab_case(s: str) -> str:
    return s.replace("_", "-")


def derive_default_value(field_info: FieldInfo, param: inspect.Parameter) -> Any:
    # The default value of an argument can be defined in two ways for Annotated types:
    # 1) After the equal sign in the function signature.
    #   Example: "param: int = 5"
    # 2) In the Field(...) info inside the Annotated metadata.
    #   Example: "param: Annotated[int, Field(default=5)]"

    default_from_equal_sign = param.default
    if default_from_equal_sign is inspect.Parameter.empty:
        default_from_equal_sign = PydanticUndefined
    default_from_field_info = field_info.default
    if default_from_field_info is PydanticUndefined:
        default_from_field_info = PydanticUndefined

    defined_default_from_equal_sign = default_from_equal_sign is not PydanticUndefined
    defined_default_from_field_info = default_from_field_info is not PydanticUndefined

    if not defined_default_from_equal_sign and not defined_default_from_field_info:
        return ...  # No default defined

    if defined_default_from_equal_sign and defined_default_from_field_info:
        if default_from_field_info != default_from_equal_sign:
            raise ValueError(
                f"The default value for '{param.name}' have been defined in two ways!! - and they are conflicting!!"
                f"The 'Field(default={default_from_field_info})' and ' = {default_from_equal_sign}'"
            )
    # After the equal sign takes precedence over Field(...) default.
    if default_from_equal_sign is not PydanticUndefined:
        return default_from_equal_sign
    return default_from_field_info


def is_field_info(obj: Any) -> bool:
    return hasattr(obj, "__class__") and obj.__class__.__name__ == "FieldInfo"


def is_cyclopts_parameter(obj: Any) -> bool:
    # Checks if the object is 'cyclopts.Parameter' without adding the 'cyclopts' package as a dependency.
    is_cyclopts = (
        hasattr(obj, "__class__") and obj.__class__.__name__ == "Parameter" and "cyclopts" in obj.__class__.__module__
    )
    return is_cyclopts


def is_typer(obj: Any) -> bool:
    # Checks if the object is 'typer.Argument'/'typer.Option' without adding the 'typer' package as a dependency.
    is_typer_arg = (
        hasattr(obj, "__class__")
        and obj.__class__.__name__ in ("ArgumentInfo", "OptionInfo")
        and "typer" in obj.__class__.__module__
    )

    is_argument = is_typer_arg and obj.__class__.__name__ == "ArgumentInfo"
    if is_argument:
        raise TypeError(
            "typer.Argument is not supported for schema generation, "
            "as it represents a positional argument. Use typer.Option(...) instead."
        )
    return is_typer_arg


def function_param_as_type_and_field_info(p: inspect.Parameter) -> Tuple[Type, FieldInfo]:
    if get_origin(p.annotation) is not Annotated:
        base_type = p.annotation
        return base_type, Field()

    args = get_args(p.annotation)
    base_type = args[0]
    annotations = convert_annotations_to_field_info(args[1:])
    for ann in annotations:
        if is_field_info(ann):
            return base_type, ann  # Return the first FieldInfo found

    return base_type, Field()


def convert_annotations_to_field_info(annotations: Tuple[Any, ...]) -> List[Any]:
    annotations_converted = []
    for ann in annotations:
        if isinstance(ann, str):
            annotations_converted.append(Field(description=ann))
        elif is_cyclopts_parameter(ann):
            param_help = getattr(ann, "help", PydanticUndefined)
            annotations_converted.append(Field(description=param_help))
        elif is_typer(ann):
            typer_help = getattr(ann, "help", PydanticUndefined)
            annotations_converted.append(Field(description=typer_help))
        else:
            annotations_converted.append(ann)  # Keep as is
    return annotations_converted
