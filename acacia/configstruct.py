# Copyright 2026, UNSW
# SPDX-License-Identifier: BSD-2-Clause
from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Iterable, Sized
from ctypes import (
    c_bool,
    c_byte,
    c_double,
    c_float,
    c_int8,
    c_int16,
    c_int32,
    c_int64,
    c_longdouble,
    c_ubyte,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
)
from typing import Any, Dict, List, Optional

from .dwarf import CType, DwarfTypes


class ConfigStruct:
    """
    A representation of a structure in a compiled program that we will pack with
    configuration data.

    This class provides a template for creating a configuration struct which will
    be serialised into a matching C struct found in a component's ELF file.

    All `ConfigStruct`s must have:
    * A `type_name` denoting the C type of this, e.g. `sddf_timer_client_config_t`.
    * A list of `fields` - name-value pairs matching the layout of a struct. E.g.
      "queue_vaddr" : 0x20000000. Names must exactly match the name of the field in
      C and the value should be an appropriate Python type. All fields in the real
      struct should be present, otherwise an error will be thrown.

    E.g.
    * a string -> character array
    * list of `int` -> list of integers of some C type (automatically checks if signed!)
    * a bytearray -> list of integers of some C type
    * Another `ConfigStruct` -> a struct.

    If this `ConfigStruct` is at the top level, i.e. it is corresponding to a symbol to patch
    as opposed to being a `ConfigStruct` nested inside another `ConfigStruct`, it must define:
    * `target_file`: the name of the ELF file this will be patched into.
    * `section_name`: The name of the ELF section this belongs to.
       NOTE: this currently is unused! https://github.com/au-ts/microkit_acacia/issues/48

    In the event that a ConfigStruct is unused, or we want to zero-fill a struct inside of another
    `ConfigStruct`, you may set the `empty` field to True. This will cause Acacia to skip trying
    to populate it while suppressing the "missing field" error that would normally appear if you
    ignore fields of a struct.
    """

    def __init__(
        self,
        fields: Dict[str, Any],
        type_name: Optional[str] = None,
        section_name: Optional[pathlib.Path] = None,
        target_file: Optional[pathlib.Path] = None,
        empty: bool = False,
    ):
        """
        :param fields: dictionary of field names -> values. Can be:
            * Another ConfigStruct
            * Any int-like value
            * Any string or bytes
            * A list of values
        :param type_name: type of struct in C
        :param section_name: name of section to patch config struct into
        :param target_file: ELF file this will be patched into
        """
        self.empty = empty
        if self.empty and len(fields):
            raise ValueError("Empty config structs cannot have fields!")

        self.fields = fields
        self.type_name = type_name
        self.section_name = section_name
        self.target_file = target_file

    def __repr__(self):
        return f"<ConfigStruct {self.type_name} {self.fields}>"


BaseTypesMap: Dict[str, Dict[int, Any]] = {
    "unsigned": {
        1: c_uint8,
        2: c_uint16,
        4: c_uint32,
        8: c_uint64,
    },
    "signed": {
        1: c_int8,
        2: c_int16,
        4: c_int32,
        8: c_int64,
    },
    "float": {
        4: c_float,
        8: c_double,
        16: c_longdouble,
    },
    "boolean": {
        1: c_bool,
    },
    "unsigned_char": {
        1: c_ubyte,
    },
    "signed_char": {
        1: c_byte,
    },
}


class ConfigStructResolver:
    """
    This class records all configuration structs of a system. As structs are
    added to this class, the DWARF of their target files is opened for looking up
    the types of the structs.

    Once all structs of the system are added, they are serialised one-by-one
    based on the memory layout of their types derived from the DWARF symbols in
    the ELF file. Each serialised struct is then written to a file in the build
    directory.

    This is an internal class for Acacia and is not used by end users.
    """

    def __init__(
        self,
        build_dir: pathlib.Path,
        endian="little",
    ):
        """
        :param build_dir: name of the build directory, used for emitting serialised
            structs
        :param endian: endianness of encoding
        """
        if endian != "little":
            raise NotImplementedError("Big endian is not currently supported!")

        self.build_dir = build_dir
        self.files: Dict[pathlib.Path, DwarfTypes] = dict()
        self.config_structs: List[ConfigStruct] = []

    def add_struct(self, s: ConfigStruct):
        """
        Add a configuration struct to the resolver.
        """
        try:
            if s.section_name is None:
                raise ValueError("Cannot emit a config struct without a section name!")
            if s.target_file is None:
                raise ValueError(
                    "Cannot patch in a config struct without a target file!"
                )
            if s.type_name is None:
                raise ValueError("Cannot patch in a config struct without a type name!")

            if s.target_file not in self.files:
                self.files[s.target_file] = self.resolve_target_file(s.target_file)
        except Exception as e:
            raise RuntimeError(f"Failed to add config struct {s}!") from e

        self.config_structs.append(s)

    def add_structs(self, lst: List[ConfigStruct]):
        """
        Add a list of configuration structs to the resolver.
        """
        for s in lst:
            self.add_struct(s)

    def resolve_target_file(self, target_file: pathlib.Path) -> DwarfTypes:
        """
        Open a target file's DWARF for looking up types.
        """
        return DwarfTypes(target_file)

    def find_type(self, config_struct: ConfigStruct) -> CType:
        """
        Find the type into which `config_struct` is to be serialised: the type of the variable in
        its section if there is one, and otherwise the type with its type name.
        """
        assert config_struct.target_file is not None
        assert config_struct.type_name is not None
        assert config_struct.section_name is not None
        types = self.files[config_struct.target_file]
        c_type = types.type_for_section(str(config_struct.section_name), config_struct.type_name)
        if c_type is None:
            c_type = types.type_by_name(config_struct.type_name)
        return c_type

    def _flatten_and_write(self, user_value: Any, c_type: CType) -> bytearray:
        """
        Returns the serialised bytes of the C type with value user_value. Used
        to create serialised configuration structs for Acacia components.

        This function is recursive. If the C type is an array or struct type,
        the function will call itself on each of the entries/members, until
        eventually a "base" type or "pointer" type is reached.

        The user value will then be serialised into bytes of the base C type
        using the ctypes module.

        The bytes of each struct member are then placed at the member's offset
        within the struct. Members are filled from offset 0 onwards. Zero-filled
        padding is added between members if the member's bytes do not reach the
        next offset.

        We only support pointer types for 64 bit architectures. Pointers are
        translated into 8 byte unsigned C types.
        """
        out_blob = bytearray()
        type_size = c_type.attributes.byte_size

        # If we were provided an "empty" config struct for this C type, 0 fill
        # the region
        if isinstance(user_value, ConfigStruct) and user_value.empty:
            out_blob.extend(0 for _ in range(type_size))
            return out_blob

        match c_type.tag_type:
            case "pointer_tag":
                assert type_size == None
                type_size = 8
                try:
                    # We assume pointers are 8 byte unsigned
                    py_ctype = c_uint64(user_value)
                except Exception as e:
                    raise ValueError(
                        f"Could not create C pointer type from value '{user_value}'"
                    ) from e
                if user_value != py_ctype.value:
                    raise ValueError(
                        f"Could not represent '{user_value}' as C base type '{c_uint64.__name__}', got '{py_ctype.value}'"
                    )

                py_bytes = bytes(py_ctype)
                out_blob.extend(py_bytes)
                assert len(out_blob) == type_size
            case "base_tag":
                if (
                    c_type.attributes.encoding not in BaseTypesMap
                    or type_size not in BaseTypesMap[c_type.attributes.encoding]
                ):
                    raise ValueError(
                        f"Found base tag type with unexpected encoding {c_type.attributes.encoding} and byte size {type_size}"
                    )
                ctype_cls = BaseTypesMap[c_type.attributes.encoding][type_size]

                if isinstance(user_value, str) or isinstance(user_value, bytes):
                    # Characters may be packed into char types, or into any single-byte integer
                    # type (Rust's c_char is u8 or i8, which are not encoded as characters).
                    is_char = c_type.attributes.encoding in (
                        "unsigned_char",
                        "signed_char",
                    ) or (
                        c_type.attributes.encoding in ("unsigned", "signed")
                        and type_size == 1
                    )
                    if not is_char:
                        raise ValueError(
                            f"User provided string or bytes value '{user_value!r}' which cannot be used for C Base type '{ctype_cls.__name__}', CType '{c_type}'"
                        )

                    if len(user_value) != 1:
                        raise ValueError(
                            f"User provided value '{user_value!r}' which cannot be used for C Base type '{ctype_cls.__name__}', CType '{c_type}'"
                        )

                    if isinstance(user_value, str):
                        py_bytes = user_value.encode()
                    else:
                        py_bytes = user_value

                else:
                    try:
                        py_ctype = ctype_cls(user_value)
                    except Exception as e:
                        raise ValueError(
                            f"Could not create C base type '{ctype_cls.__name__}' from value '{user_value}'"
                        ) from e

                    if user_value != py_ctype.value:
                        raise ValueError(
                            f"Could not represent '{user_value}' as C base type '{ctype_cls.__name__}', got '{py_ctype.value}'"
                        )

                    py_bytes = bytes(py_ctype)

                out_blob.extend(py_bytes)
                assert len(out_blob) == type_size
            case "array_tag":
                entry_type = c_type.base_type()
                array_count = c_type.array_count()
                if not array_count or not entry_type.attributes.byte_size:
                    raise ValueError(
                        f"Can't fill an array with size = count '{array_count}' * entry_size '{entry_type.attributes.byte_size}' == 0, CType '{c_type}'"
                    )

                if (
                    not (
                        isinstance(user_value, list)
                        or isinstance(user_value, tuple)
                        or isinstance(user_value, str)
                        or isinstance(user_value, bytes)
                        or isinstance(user_value, bytearray)
                    )
                    or len(user_value) > array_count
                ):
                    raise ValueError(
                        f"Can't fill an array with a non-iterable or too-large an iterable '{user_value}', CType '{c_type}'"
                    )

                for entry_value in user_value:
                    try:
                        out_blob.extend(
                            self._flatten_and_write(entry_value, entry_type)
                        )
                    except Exception as e:
                        raise Exception(
                            f"Exception occurred while filling field of array CType '{c_type}'`"
                        ) from e

                type_size = array_count * entry_type.attributes.byte_size

                # Strings must be null-terminated
                if entry_type.attributes.name == "char":
                    if len(out_blob) == type_size and out_blob[-1] != 0:
                        raise ValueError(
                            f"Arrays of chars must be zero terminated, string '{user_value!r}', CType '{c_type}'"
                        )

                # Pad the remaining array entries
                assert len(out_blob) <= type_size
                out_blob.extend(bytearray(type_size - len(out_blob)))
            case "structure_tag":
                member_types = list(
                    c_type.collector[member_id] for member_id in c_type.members
                )

                # Every member must have a field value, and vice versa
                if set(user_value.fields.keys()) != set(
                    member.attributes.name for member in member_types
                ):
                    raise ValueError(
                        f"User provided keys '{set(user_value.fields.keys())}' do not match structure type member names '{set(member.attributes.name for member in member_types)}', CType '{c_type}'"
                    )

                # Every structure tag member must have a specified data member location
                if None in set(m.attributes.data_member_location for m in member_types):
                    raise ValueError(
                        f"Structure type '{c_type}' has member without offset location, struct bit fields are not supported."
                    )

                for member in sorted(
                    member_types, key=lambda m: m.attributes.data_member_location
                ):
                    start_byte = member.attributes.data_member_location

                    # Pad up to this member
                    assert len(out_blob) <= start_byte
                    out_blob.extend(bytearray(start_byte - len(out_blob)))

                    member_value = user_value.fields[member.attributes.name]
                    try:
                        out_blob.extend(
                            self._flatten_and_write(member_value, member.base_type())
                        )
                    except Exception as e:
                        raise Exception(
                            f"Exception occurred while filling member of structure CType '{c_type}'`"
                        ) from e

                assert len(out_blob) <= type_size
                out_blob.extend(bytearray(type_size - len(out_blob)))
            case _:
                raise ValueError(f"Trying to flatten unexpected CType '{c_type}'")

        return out_blob

    def resolve_and_create_all(self, dump: bool = False):
        """
        Try generate data files for all currently known config structs. Output
        will be placed in `self.build_dir`.
        """
        for config_struct in self.config_structs:
            # These error checks aren't actually needed, since this is prevented by now. Do it anyway
            # to satisfy MyPy
            if config_struct.target_file is None:
                raise RuntimeError(
                    f"Top level config struct {config_struct} has no target file!"
                )
            if config_struct.type_name is None:
                raise RuntimeError(
                    f"Top level config struct {config_struct} has no type name!"
                )

            c_type = self.find_type(config_struct)
            try:
                blob = self._flatten_and_write(config_struct, c_type)
            except Exception as e:
                raise Exception(
                    f"Exception occurred while flattening config struct '{config_struct}' into structure CType '{c_type}'`"
                ) from e
            blob_name = f"{str(config_struct.target_file).removesuffix('.elf')}_{config_struct.section_name}.data"
            blob_path = os.path.join(self.build_dir, blob_name)
            with open(blob_path, "wb") as f:
                f.write(blob)

            if dump:
                json_path = f"{blob_path.removesuffix('.data')}.json"
                with open(json_path, "w") as f:
                    json.dump(
                        {
                            "type_name": config_struct.type_name,
                            "section_name": str(config_struct.section_name),
                            "target_file": str(config_struct.target_file),
                            "empty": config_struct.empty,
                            "fields": {
                                name: _value_to_jsonable(value)
                                for name, value in config_struct.fields.items()
                            },
                        },
                        f,
                        indent=2,
                    )
                    f.write("\n")


def _value_to_jsonable(value: Any) -> Any:
    """
    Create JSON-friendly values for configstruct fields.
    ConfigStruct            -> {"empty": <bool>, "fields": {...}}
    bytes/bytearray/int     -> hex string
    list/tuple              -> list
    str/float/bool/None     -> unchanged
    """
    if isinstance(value, ConfigStruct):
        return {
            "empty": value.empty,
            "fields": {name: _value_to_jsonable(v) for name, v in value.fields.items()},
        }
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if isinstance(value, int):
        return hex(value)
    if isinstance(value, (list, tuple)):
        return [_value_to_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, float, bool)):
        return value
    raise TypeError(
        f"Value '{value!r}' of type '{type(value).__name__}' is not JSON-serialisable"
    )
