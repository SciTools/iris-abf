from unittest.mock import Mock
import sys
import types


def _mock_module(module_name, attribute_names):
    module = types.ModuleType(module_name)
    for attribute_name in attribute_names:
        setattr(module, attribute_name, Mock())
    sys.modules[module_name] = module


_mock_module("iris.coords", ["AuxCoord", "DimCoord"])
_mock_module("iris.coord_systems", ["GeogCS"])
_mock_module("iris.io.format_picker", ["FileExtension", "FormatSpecification"])
_mock_module("iris.fileformats", [])
