# Copyright iris-abf contributors
#
# This file is part of iris-abf and is released under the BSD 3-clause
# licence. See LICENSE in the root of the repository for full licensing
# details.
"""
Unit tests for `iris.io.plugins.abf`.
"""
from unittest.mock import Mock, call, patch, sentinel

import numpy as np
import pytest

# Until we can pip install iris, use a pretend version.
import mock_iris

import iris.io.plugins.abf as abf


def test_ABFField__init__ok():
    filename = "/path/to/abf/file/" + "0" * 24
    field = abf.ABFField(filename)
    assert field._filename == filename


def test_ABFField__init__invalid():
    filename = "/path/to/abf/file/too_short"
    with pytest.raises(ValueError):
        abf.ABFField(filename)


@patch("iris.io.plugins.abf.ABFField._read")
def test_ABFField__getattr__data(_read):
    def read(*args):
        field.data = sentinel.DATA

    _read.side_effect = read
    field = abf.ABFField("0" * 24)
    data = field.data
    assert data == sentinel.DATA


@patch("iris.io.plugins.abf.np.fromfile")
def test_ABFField_read(fromfile):
    fromfile.return_value = np.empty((4320, 2160))
    fromfile.return_value = np.arange(4320 * 2160, dtype=np.uint8)
    field = abf.ABFField("/path/to/my-prefix12-9876marb-abf")
    field._read()
    assert field.version == 12
    assert field.year == 9876
    assert field.month == 3
    assert field.period == "b"
    assert field.format == "abf"
    data = field.data
    assert data.shape == (2160, 4320)
    assert np.ma.is_masked(data[0, 0])
    assert data[0, 4317] == 31
    assert data[2159, 0] == 0


@patch("iris.cube", create=True)
@patch("iris.io.plugins.abf.GeogCS")
@patch("iris.io.plugins.abf.DimCoord")
@patch("iris.io.plugins.abf.AuxCoord")
def test_ABFField_to_cube__format_abf_period_a(AuxCoord, DimCoord, GeogCS, cube):
    result_cube = Mock()
    result_cube.attributes = {}
    cube.Cube.return_value = result_cube
    x_coord = Mock()
    y_coord = Mock()
    DimCoord.side_effect = (x_coord, y_coord)
    GeogCS.return_value = sentinel.GEOG_CS
    AuxCoord.return_value = sentinel.TIME_COORD
    field = Mock(format="abf", period="a", year=2020, month=4)

    cube = abf.ABFField.to_cube(field)

    cube.rename.assert_called_once_with("FAPAR")
    assert cube.units == "%"
    assert cube.add_dim_coord.mock_calls == [call(x_coord, 1), call(y_coord, 0)]
    (_, (x_points,), x_kwargs), (_, (y_points,), y_kwargs) = DimCoord.mock_calls
    assert len(x_points) == 4320
    assert x_points[0] == -179.95833333333334
    assert x_points[-1] == 179.95833333333331
    assert x_kwargs == {
        "standard_name": "longitude",
        "units": "degrees",
        "coord_system": sentinel.GEOG_CS,
    }
    assert len(y_points) == 2160
    assert y_points[0] == -89.95833333333333
    assert y_points[-1] == 89.95833333333331
    assert y_kwargs == {
        "standard_name": "latitude",
        "units": "degrees",
        "coord_system": sentinel.GEOG_CS,
    }
    x_coord.guess_bounds.assert_called_once_with()
    y_coord.guess_bounds.assert_called_once_with()
    cube.add_aux_coord.assert_called_once_with(sentinel.TIME_COORD)
    AuxCoord.assert_called_once_with(
        737515,
        bounds=[737515, 737529],
        standard_name="time",
        units="days since 0001-01-01",
    )
    assert cube.attributes == {"source": "Boston University"}


@patch("iris.cube", create=True)
@patch("iris.io.plugins.abf.GeogCS")
@patch("iris.io.plugins.abf.DimCoord")
@patch("iris.io.plugins.abf.AuxCoord")
def test_ABFField_to_cube__format_abl_period_b(AuxCoord, DimCoord, GeogCS, cube):
    field = Mock(format="abl", period="b", year=2020, month=4)

    cube = abf.ABFField.to_cube(field)

    cube.rename.assert_called_once_with("leaf_area_index")
    AuxCoord.assert_called_once_with(
        737530,
        bounds=[737530, 737544],
        standard_name="time",
        units="days since 0001-01-01",
    )


@patch("iris.io.plugins.abf.ABFField.to_cube")
@patch("glob.glob")
def test_load_cubes__single_filename_no_callback(glob, to_cube):
    glob.return_value = [
        "/path/to/123456789012345678901234",
        "/another/123456789012345678901234",
    ]
    to_cube.side_effect = [sentinel.CUBE1, sentinel.CUBE2]
    filename = "some-filename"

    cubes = list(abf.load_cubes(filename))

    glob.assert_called_once_with(filename)
    assert cubes == [sentinel.CUBE1, sentinel.CUBE2]


@patch("iris.io.run_callback", create=True)
@patch("iris.io.plugins.abf.ABFField.to_cube")
@patch("glob.glob")
def test_load_cubes__multiple_filenames_with_callback(glob, to_cube, run_callback):
    glob.return_value = ["/path/to/123456789012345678901234"]
    to_cube.side_effect = [sentinel.CUBE1, sentinel.CUBE2]
    run_callback.side_effect = [None, sentinel.CALLBACK_CUBE]
    filenames = ["first-filename", "second-filename"]

    cubes = list(abf.load_cubes(filenames, sentinel.CALLBACK))

    assert glob.mock_calls == [call("first-filename"), call("second-filename")]
    assert cubes == [sentinel.CALLBACK_CUBE]
