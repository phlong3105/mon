import copy
import inspect
from pathlib import Path
from typing import Annotated, Callable, Literal, Optional

import cyclopts
import pytest
import typer
from flask import json
from pydantic import BaseModel
from pydantic import Field as PydanticField
from typer.testing import CliRunner

from hafnia.experiment import command_builder
from hafnia.experiment.command_builder import CommandBuilderSchema, auto_save_command_builder_schema, simulate_form_data


def test_docstring_description_example():
    """Test to extract parameter descriptions from docstrings."""

    def docstring_description_example(param0: str, param1: int):
        """
        Example function to demonstrate docstring description extraction.
        Below descriptions for each parameter are automatically extracted and used in the schema.

        Parameters
        ----------
        param0 : str
            Parameter docstring 0.
        param1 : int
            Parameter docstring 1.
        """
        pass

    schema = command_builder.schema_from_cli_function(docstring_description_example)
    for idx, (name, prop) in enumerate(schema["properties"].items()):
        assert "description" in prop, f"Parameter '{name}' is missing description in schema."
        assert prop["description"] == f"Parameter docstring {idx}.", (
            f"Parameter '{name}' has incorrect description in schema."
        )
        assert "type" in prop, f"Parameter '{name}' is missing type in schema."
        assert "title" in prop, f"Parameter '{name}' is missing title in schema."


def test_typer_argument_and_option_example():
    """Test to extract parameter descriptions from typer.Argument and typer.Option."""

    def typer_argument_and_option_example(
        param1: Annotated[str, typer.Option(help="Param without default.")],
        param2: Annotated[int, typer.Option(help="Param with default")] = 10,
    ):
        pass

    has_default_value = ["param2"]
    schema = command_builder.schema_from_cli_function(typer_argument_and_option_example)
    for name, prop in schema["properties"].items():
        assert "description" in prop, f"Parameter '{name}' is missing description in schema."
        assert "type" in prop, f"Parameter '{name}' is missing type in schema."
        assert "title" in prop, f"Parameter '{name}' is missing title in schema."
        if name in has_default_value:
            assert "default" in prop, f"Parameter '{name}' is missing default in schema."


def test_typer_argument_not_supported():
    """Test to extract parameter descriptions from typer.Argument and typer.Option."""

    def typer_argument_and_option_example(
        param1: Annotated[str, typer.Argument(help="This is an example parameter using typer.Argument.")],
        param2: Annotated[int, typer.Option(help="Another parameter with typer.Option description.")] = 10,
    ):
        pass

    with pytest.raises(TypeError, match="typer.Argument is not supported"):
        command_builder.schema_from_cli_function(typer_argument_and_option_example)


def test_cyclopts_parameter_example():
    """Test to extract parameter descriptions from cyclopts.Parameter."""

    def cyclopts_parameter_example(
        param1: Annotated[str, cyclopts.Parameter(help="This is an example parameter using cyclopts.Parameter.")],
        param2: Annotated[int, cyclopts.Parameter(help="Another parameter with cyclopts.Parameter description.")] = 10,
    ):
        pass

    has_default_value = ["param2"]
    schema = command_builder.schema_from_cli_function(cyclopts_parameter_example)
    for name, prop in schema["properties"].items():
        assert "description" in prop, f"Parameter '{name}' is missing description in schema."
        assert "type" in prop, f"Parameter '{name}' is missing type in schema."
        assert "title" in prop, f"Parameter '{name}' is missing title in schema."
        if name in has_default_value:
            assert "default" in prop, f"Parameter '{name}' is missing default in schema."


def test_pydantic_field_example():
    """Test to extract parameter descriptions from Pydantic Field."""

    def pydantic_field_example(
        param1: Annotated[str, PydanticField(description="This is an example parameter using Pydantic Field.")],
        param2: Annotated[int, PydanticField(description="Another parameter with Pydantic Field description.")] = 10,
    ):
        pass

    has_default_value = ["param2"]
    schema = command_builder.schema_from_cli_function(pydantic_field_example)
    for name, prop in schema["properties"].items():
        assert "description" in prop, f"Parameter '{name}' is missing description in schema."
        assert "type" in prop, f"Parameter '{name}' is missing type in schema."
        assert "title" in prop, f"Parameter '{name}' is missing title in schema."
        if name in has_default_value:
            assert "default" in prop, f"Parameter '{name}' is missing default in schema."


def test_default_mismatch_failure():
    """Test to extract parameter descriptions from docstrings."""

    def default_value_mismatch(
        param0: Annotated[str, PydanticField(default="asdf", description="Desc 0.")],
        param1: Annotated[str, PydanticField(default="One", description="Desc 1.")] = "Two",
    ):
        pass

    with pytest.raises(ValueError, match="The default value for 'param1' have been defined in two ways"):
        command_builder.schema_from_cli_function(default_value_mismatch)


def example_advanced_cli_example() -> Callable:
    """Advanced test function to generate and print Pydantic schema from function signature."""

    class NestedNestedModel(BaseModel):
        optimizer_name: Optional[str] = PydanticField(..., description="Name of the optimizer")
        learning_rate: float = PydanticField(0.01, description="Learning rate for optimizer")

    class ExampleNestedModel(BaseModel):
        model_name: Optional[str] = PydanticField(..., description="Name of the model")
        epochs: int = PydanticField(3, description="Number of epochs to train")
        optimizer: Optional[NestedNestedModel] = PydanticField(None, description="Nested nested model configuration")

    EXAMPLE_NESTED_MODEL_DEFAULT = ExampleNestedModel(
        model_name="default_model",
        epochs=5,
        optimizer=NestedNestedModel(optimizer_name="adam", learning_rate=0.01),
    )

    def cli_function_example(
        test: str,  # Required parameter without annotation
        string_description_from_doc: str,  # Description is automatically taken from docstring
        required_string: Annotated[
            str, "This description is used", PydanticField(description="This description is ignored", min_length=2)
        ],
        bool_flag: Annotated[bool, "A boolean flag"],
        nested_model: Annotated[ExampleNestedModel, "Nested model configuration"],
        string_description_from_doc_with_default: str = "blah",  # Description is automatically taken from docstring
        string_with_default_underscore: str = "blah_underscore",  # We should not convert underscore values to kebab-case
        string_min_length: Annotated[
            str, PydanticField(description="SomeDescription", min_length=2)
        ] = "default_string",
        int_or_string: Annotated[int | str, "An int or string value"] = 1,
        bool_flag_with_default: Annotated[bool, "A boolean flag with default"] = False,
        project_name1: Annotated[str, PydanticField(default="asdf", description="Project name1")] = "asdf",
        project_name2: Annotated[str, PydanticField(description="Project name2")] = "asdf",
        project_name3: Annotated[str, cyclopts.Parameter(help="Project name2")] = "asdf",
        name_typer: Annotated[str, typer.Option(help="This is a test typer option")] = "default_name",
        epochs: Annotated[int, cyclopts.Parameter(help="Number of epochs to train")] = 3,
        learning_rate: Annotated[float, "Learning rate for optimizer"] = 0.001,
        resize: Annotated[Optional[int], "Resize image to specified size. If None, will drop unless needed"] = None,
        batch_size: Annotated[int, PydanticField(description="Batch size for training", gt=0, lt=1000)] = 128,
        num_workers: Annotated[int, "Number of workers for DataLoader"] = 8,
        log_interval: Annotated[int, "Interval for logging"] = 5,
        max_steps_per_epoch: Annotated[int, "Max steps per epoch"] = 20,
        models: Annotated[Literal["resnet18", "vgg16"], "Model name"] = "resnet18",
        nested_model_with_default: Annotated[
            ExampleNestedModel, "Nested model configuration"
        ] = EXAMPLE_NESTED_MODEL_DEFAULT,
    ):
        """
        Example function with the same signature as train.py main()

        Parameters
        ----------
        string_description_from_doc : str
            Description is automatically taken from docstring.
        string_description_from_doc_with_default : str
            Description is automatically taken from docstring.
        """
        pass

    return cli_function_example


def test_command_builder_schema(tmp_path: Path):
    """Test function to generate and print Pydantic schema from function signature."""

    ## Setup example function with various parameter types and annotations
    cli_function_example = example_advanced_cli_example()
    path_schema_json = tmp_path / "command_schema.json"
    auto_save_command_builder_schema(
        name="Trainer",
        positional_args=[],
        cli_function=cli_function_example,
        path_schema=path_schema_json,
    )

    assert path_schema_json.exists(), "Schema JSON file was not created."

    schema = CommandBuilderSchema.from_json_file(path_schema_json)
    schema.model_dump(mode="json")
    assert schema.name == "Trainer"


def test_command_builder_from_function():
    """Test function to generate and print Pydantic schema from function signature."""

    ## Setup example function with various parameter types and annotations
    cli_function_example = example_advanced_cli_example()

    # Test: Generate schema
    command_builder_schema = CommandBuilderSchema.from_function(cli_function_example)
    schema_dict = command_builder_schema.json_schema

    # Assert:
    no_description_params = ["test", "string_with_default_underscore"]
    no_default_params = ["test", "string_description_from_doc", "required_string", "bool_flag", "nested_model"]
    required_params = no_default_params

    for name, prop in schema_dict["properties"].items():
        if "$ref" in prop:  # Skip $ref properties
            continue
        if name not in no_description_params:
            assert "description" in prop, f"Parameter '{name}' is missing description in schema."

        if "anyOf" not in prop:  # Skip complex types
            assert "type" in prop, f"Parameter '{name}' is missing type in schema."

        assert "title" in prop, f"Parameter '{name}' is missing title in schema."

        if name in no_default_params:
            assert "default" not in prop, f"Parameter '{name}' should not have a default in schema."
        else:
            assert "default" in prop, f"Parameter '{name}' is missing default in schema."

    props = schema_dict["properties"]
    assert props["string_with_default_underscore"]["default"] == "blah_underscore", (
        "Default value for 'string_with_default_underscore' is incorrect."
    )
    assert required_params == schema_dict.get("required", []), "Required parameters do not match schema."

    # Check nested model properties
    assert props["nested_model"]["$ref"] == "#/$defs/ExampleNestedModel", "Nested model $ref is incorrect."
    nested_model_def = schema_dict["$defs"]["ExampleNestedModel"]
    assert set(nested_model_def["properties"]) == {"model_name", "epochs", "optimizer"}, (
        "Nested model properties are incorrect."
    )

    # Check nested model with default
    nested_model_with_default = props["nested_model_with_default"]
    assert nested_model_with_default["$ref"] == "#/$defs/ExampleNestedModel", (
        "Nested model with default $ref is incorrect."
    )

    expected_defaults = {
        "model_name": "default_model",
        "epochs": 5,
        "optimizer": {"optimizer_name": "adam", "learning_rate": 0.01},
    }
    assert nested_model_with_default["default"] == expected_defaults, "Nested model with default value is incorrect."

    # Check union types
    union_field_key = [
        "properties.resize",
        "properties.int_or_string",
        "$defs.ExampleNestedModel.properties.model_name",
    ]
    for field_key in union_field_key:
        union_prop = copy.deepcopy(schema_dict)
        nested_field_keys = field_key.split(".")
        for key in nested_field_keys:
            if key not in union_prop:
                raise KeyError(f"Key '{key}' not found in schema while accessing field '{field_key}'.")
            union_prop = union_prop[key]

        assert "anyOf" in union_prop, "Union type should have 'anyOf' in schema."
        assert len(union_prop["anyOf"]) == 2, "Union type 'float_or_int' should have two types in 'anyOf'."
        assert all("type" in item for item in union_prop["anyOf"]), "All union types should anyOf should have a type."
        assert all("title" in item for item in union_prop["anyOf"]), "All union type should anyOf should have a title."


def test_command_args_from_form_data():
    cli_function_example = example_advanced_cli_example()
    update_params = {
        "test": "my_test_value",
        "required_string": "my_required_string",
        "nested_model": {"model_name": "custom_model"},
        "string_description_from_doc": "custom description",
        "bool_flag": True,
    }
    params = inspect.signature(cli_function_example).parameters
    n_root_params = len(params)
    form_dataset = command_builder.simulate_form_data(cli_function_example, update_params)
    assert len(form_dataset) == n_root_params, "Form dataset does not have the correct number of parameters."

    cmd_builder_schema = CommandBuilderSchema.from_function(cli_function_example)
    commands_args = cmd_builder_schema.command_args_from_form_data(form_dataset)
    command_str = " ".join(commands_args)
    assert command_str.count(" --") > n_root_params


def test_command_args_from_form_data_simple():
    class NestedModel(BaseModel):
        name: str  # type: ignore

    def some_function(
        param_value1: str,
        param_value2: int = 10,
        nested: NestedModel = NestedModel(name="some name"),
        bool_flag1: Annotated[bool, "A boolean flag 1"] = False,
        bool_flag2: Annotated[bool, "A boolean flag 2"] = True,
    ) -> None:
        return None

    update_params = {
        "param_value1": "custom_value",
    }

    params = inspect.signature(some_function).parameters
    n_root_params = len(params)
    form_dataset = command_builder.simulate_form_data(some_function, update_params)
    assert len(form_dataset) == n_root_params, "Form dataset does not have the correct number of parameters."

    # Use case 1: Using default settings
    cmd_builder1 = CommandBuilderSchema.from_function(some_function)
    commands_args = cmd_builder1.command_args_from_form_data(form_dataset)
    cmd_string = " ".join(commands_args)
    assert cmd_string.count(" --") == n_root_params - len(cmd_builder1.positional_args)
    assert "nested.name" in cmd_string, "Nested parameter not correctly represented. Expected '.' separator."
    assert "param-value1" in cmd_string, "Parameter value was not converted to kebab-case."
    assert "--no-bool-flag1" in cmd_string, "Boolean flag not correctly represented with default settings."
    assert "--bool-flag2" in cmd_string, "Boolean flag with default True not correctly represented."

    # Use case 2: Custom settings, with 'param_value1' declared as a positional argument
    cmd_builder2 = CommandBuilderSchema.from_function(
        some_function,
        positional_args=["param_value1"],
        parameter_prefix="++",
        nested_parameter_handling="dot",
        case_conversion="none",
        assignment_separator="equals",
        bool_handling="none",
    )

    commands_args = cmd_builder2.command_args_from_form_data(form_dataset)
    cmd_string = " ".join(commands_args)
    assert cmd_builder2.positional_args == ["param_value1"], "Positional args not stored on the schema."
    assert cmd_string.count(" ++") == n_root_params - len(cmd_builder2.positional_args)
    assert "nested.name" in cmd_string, "Nested parameter not correctly represented. Expected '.' separator."
    assert "param_value2" in cmd_string, "Parameter value was incorrectly converted to kebab-case."
    assert "++nested.name='some name'" in cmd_string, "Assignment separator '=' not correctly used."
    assert "++bool_flag1=False" in cmd_string, "Boolean flag not correctly represented with flag-negation handling."
    assert "++bool_flag2=True" in cmd_string, (
        "Boolean flag with default True not correctly represented with flag-negation."
    )
    # The positional argument is emitted first - as a bare value, before any '++' option.
    assert commands_args[len(cmd_builder2.cmd.split(" "))] == "'custom_value'", (
        "Positional argument should be emitted first as a bare value."
    )
    assert "++param_value1" not in cmd_string, "Positional argument should not be emitted as a named option."


def test_positional_args_unknown_name_raises():
    """A name in 'positional_args' that is not a parameter of the function must raise a ValueError."""

    def some_function(param_value1: str, param_value2: int = 10) -> None:
        return None

    with pytest.raises(ValueError, match="positional_args"):
        CommandBuilderSchema.from_function(some_function, positional_args=["does_not_exist"])


def test_cyclopts_cmdline():
    app = cyclopts.App(name="train", help="PyTorch Training")

    # For testing nested models
    class NestedModel(BaseModel):
        nested_param_str: Annotated[str, cyclopts.Parameter(help="This is nested param 1")]  # type: ignore
        nested_param_int: Annotated[int, cyclopts.Parameter(help="This is nested param 2")] = 5  # type: ignore

    # Simulate cyclopts cli application function that will be converted into a cli tool
    @app.default
    def cyclopts_example(
        string_param0: Annotated[str, cyclopts.Parameter(help="This is string")],
        nested_model_no_default: NestedModel,
        string_param1: Annotated[str, cyclopts.Parameter(help="This is string")] = "default value",
        nested_model: NestedModel = NestedModel(nested_param_str="nested default"),
        param_int: Annotated[int, cyclopts.Parameter(help="This is int")] = 10,
        bool_param: Annotated[bool, cyclopts.Parameter(help="This is bool")] = False,
        bool_param_true: Annotated[bool, cyclopts.Parameter(help="This is bool true")] = True,
    ):
        return {
            "string_param0": string_param0,
            "nested_model_no_default": nested_model_no_default,
            "string_param1": string_param1,
            "nested_model": nested_model,
            "param_int": param_int,
            "bool_param": bool_param,
            "bool_param_true": bool_param_true,
        }

    # Generate command builder schema from cyclopts function
    command_builder = CommandBuilderSchema.from_function(
        cyclopts_example,
        cli_tool="cyclopts",
    )

    # Simulate form data submission
    user_args = {
        "string_param0": "some string",
        "nested_model_no_default": NestedModel(nested_param_str="nested value"),
    }
    form_data = simulate_form_data(cyclopts_example, user_args=user_args)

    #
    commands_args = command_builder.command_args_from_form_data(form_data, include_cmd=False)
    str_args = " ".join(commands_args)
    func, arguments_parsed, _ = app.parse_args(str_args)

    returns = cyclopts_example(**arguments_parsed.arguments)
    assert returns["string_param0"] == "some string"
    assert returns["string_param1"] == "default value"
    assert returns["param_int"] == 10
    assert returns["bool_param"] is False
    assert returns["bool_param_true"] is True
    assert returns["nested_model_no_default"].nested_param_str == "nested value"
    assert returns["nested_model_no_default"].nested_param_int == 5
    assert returns["nested_model"].nested_param_str == "nested default"
    assert returns["nested_model"].nested_param_int == 5


def test_typer_cmdline(tmp_path: Path):
    app = typer.Typer(add_completion=False)
    function_data = tmp_path / "function_data.json"

    # Simulate typer cli application function that will be converted into a cli tool
    @app.command()
    def typer_example(
        string_param0: Annotated[str, typer.Option(help="This is string")],
        string_param1: Annotated[str, typer.Option(help="This is string")] = "default value",
        param_int: Annotated[int, typer.Option(help="This is int")] = 10,
        bool_param: Annotated[bool, typer.Option(help="This is bool")] = False,
        bool_param_true: Annotated[bool, typer.Option(help="This is bool true")] = True,
    ):
        parsed_data = {
            "string_param0": string_param0,
            "string_param1": string_param1,
            "param_int": param_int,
            "bool_param": bool_param,
            "bool_param_true": bool_param_true,
        }
        # Hack to save function data for verification since typer function return is not accessible via CliRunner
        function_data.write_text(json.dumps(parsed_data))
        return parsed_data  # This is not accessible via CliRunner

    # Generate command builder schema from typer function
    command_builder = CommandBuilderSchema.from_function(typer_example, cli_tool="typer")

    # Simulate form data submission
    user_args = {
        "string_param0": "some string",
    }
    form_data = simulate_form_data(typer_example, user_args=user_args)
    commands_args = command_builder.command_args_from_form_data(form_data, include_cmd=False)
    str_args = " ".join(commands_args)

    runner = CliRunner()
    result = runner.invoke(app, str_args)
    assert result.exit_code == 0, f"Command failed: {result.stderr}"

    # Load the saved function data to verify parameters
    parsed_data = json.loads(function_data.read_text())
    assert parsed_data["string_param0"] == "some string"
    assert parsed_data["string_param1"] == "default value"
    assert parsed_data["param_int"] == 10
    assert parsed_data["bool_param"] is False
    assert parsed_data["bool_param_true"] is True


def test_enum_not_supported():
    """Test to ensure that using Enum types raises an error."""

    from enum import Enum

    class Color(Enum):
        RED = "red"
        GREEN = "green"
        BLUE = "blue"

    def enum_parameter_example(
        color: Annotated[Color, "This is an example parameter using Enum."],
    ):
        pass

    with pytest.raises(TypeError, match="'Enum' CLI argument types are not supported yet"):
        command_builder.schema_from_cli_function(enum_parameter_example)


def _write_schema(path: Path, name: str, schema: dict) -> Path:
    schema_file = path / f"{name}.schema.json"
    schema_file.write_text(json.dumps(schema))
    return schema_file


def test_auto_discover_cmd_builder_schemas_ordering(tmp_path: Path):
    """Schemas are returned sorted by their 'order' field, missing 'order' treated as DEFAULT_ORDER."""
    from hafnia.experiment.command_builder import DEFAULT_ORDER
    from hafnia.platform.trainer_package import auto_discover_cmd_builder_schemas

    package_files = [
        _write_schema(tmp_path, "third", {"cmd": "third", "json_schema": {}, "order": 30}),
        _write_schema(tmp_path, "first", {"cmd": "first", "json_schema": {}, "order": 10}),
        _write_schema(tmp_path, "second", {"cmd": "second", "json_schema": {}, "order": 20}),
        # No 'order' field -> treated as DEFAULT_ORDER (100), so it sorts last here.
        _write_schema(tmp_path, "no_order", {"cmd": "no_order", "json_schema": {}}),
    ]

    schemas = auto_discover_cmd_builder_schemas(package_files)

    assert [s["cmd"] for s in schemas] == ["first", "second", "third", "no_order"]
    assert DEFAULT_ORDER > 30  # sanity check that 'no_order' is expected to sort after the explicit orders


def test_auto_discover_cmd_builder_schemas_skips_invalid(tmp_path: Path):
    """Schemas missing a required field ('cmd' or 'json_schema'), or that are not valid JSON, are skipped."""
    from hafnia.platform.trainer_package import auto_discover_cmd_builder_schemas

    malformed = tmp_path / "malformed.schema.json"
    malformed.write_text("{not valid json")

    package_files = [
        _write_schema(tmp_path, "valid", {"cmd": "valid", "json_schema": {}}),
        _write_schema(tmp_path, "missing_cmd", {"json_schema": {}}),
        _write_schema(tmp_path, "missing_json_schema", {"cmd": "missing_json_schema"}),
        _write_schema(tmp_path, "empty", {}),
        malformed,
    ]

    schemas = auto_discover_cmd_builder_schemas(package_files)

    assert [s["cmd"] for s in schemas] == ["valid"]


def test_auto_discover_cmd_builder_schemas_only_schema_json_files(tmp_path: Path):
    """Only files ending with '.schema.json' are considered; other packaged files are ignored."""
    from hafnia.platform.trainer_package import auto_discover_cmd_builder_schemas

    schema_file = _write_schema(tmp_path, "valid", {"cmd": "valid", "json_schema": {}})
    other_file = tmp_path / "train.py"
    other_file.write_text("print('hello')")
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"cmd": "config", "json_schema": {}}))

    schemas = auto_discover_cmd_builder_schemas([schema_file, other_file, config_file])

    assert [s["cmd"] for s in schemas] == ["valid"]
