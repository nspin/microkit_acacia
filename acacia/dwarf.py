# Copyright 2026, UNSW
# SPDX-License-Identifier: BSD-2-Clause

"""
Discovery of ConfigStruct layouts from the DWARF of ELF files, using pyelftools.

Types are only decoded on demand, starting from the type of interest, so large amounts of
irrelevant DWARF (e.g. from Rust code) are skipped rather than parsed.
"""

from __future__ import annotations

import pathlib
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, Tuple

from elftools.dwarf.descriptions import describe_attr_value
from elftools.dwarf.die import DIE
from elftools.dwarf.dwarf_expr import DWARFExprParser
from elftools.dwarf.enums import ENUM_DW_ATE
from elftools.elf.elffile import ELFFile

_ATE_NAMES = {v: k.removeprefix("DW_ATE_") for k, v in ENUM_DW_ATE.items()}

_TAG_TYPES = {
    "DW_TAG_base_type": "base_tag",
    "DW_TAG_typedef": "typedef_tag",
    "DW_TAG_pointer_type": "pointer_tag",
    "DW_TAG_structure_type": "structure_tag",
    "DW_TAG_union_type": "union_tag",
    "DW_TAG_array_type": "array_tag",
    "DW_TAG_enumeration_type": "enumeration_tag",
    "DW_TAG_member": "member_tag",
    "DW_TAG_subrange_type": "subrange_tag",
    "DW_TAG_enumerator": "enumerator_tag",
}

_MEMBER_TAGS = {
    "structure_tag": ("DW_TAG_member",),
    "union_tag": ("DW_TAG_member",),
    "array_tag": ("DW_TAG_subrange_type",),
    "enumeration_tag": ("DW_TAG_enumerator",),
}


class Attributes:
    """
    The attributes of a DWARF entry which are relevant to laying out ConfigStructs. This is used
    internally by Acacia when discovering the layout of ConfigStructs in compiled code.

    Attributes which were not present in the entry read as None.
    """

    valid_attributes = (
        "name",
        "encoding",
        "byte_size",
        "const_value",
        "data_member_location",
        "upper_bound",
        "count",
        "type",
        "bit_size",
        "data_bit_offset",
    )

    def __init__(self, values: Dict[str, Any]):
        """
        :param values: attribute values. `encoding` omits the `DW_ATE_` prefix, and `type` is a
            tuple of the ID and name of the referenced entry.
        """
        for name in values:
            if name not in self.valid_attributes:
                raise AttributeError(f"Attributes class has no attribute '{name}'")
        self.__dict__.update(values)

    def __getattr__(self, name: str):
        if name in self.valid_attributes:
            return None
        raise AttributeError(f"Attributes class has no attribute '{name}'")

    def __repr__(self):
        return (
            "Attributes:\n    "
            + "\n    ".join(
                [
                    f"{at_name}: {getattr(self, at_name)}"
                    for at_name in self.valid_attributes
                ]
            )
            + "\n"
        )


class CType:
    """
    A type (or a component of a type, such as a struct member) described by a DWARF entry. This
    is used internally by Acacia to represent types found in a compiled file when determining how
    to pack ConfigStructs.

    `tag_type` is one of:
        base_tag, typedef_tag, pointer_tag, structure_tag, union_tag, array_tag,
        enumeration_tag, member_tag, subrange_tag, enumerator_tag, or unknown_tag (for any other
        kind of entry)

    `members` holds the IDs of the member_tag, subrange_tag, or enumerator_tag entries of
    structure_tag/union_tag, array_tag, and enumeration_tag types respectively.
    """

    member_tags = (
        "subrange_tag",
        "enumerator_tag",
        "member_tag",
    )

    def __init__(
        self,
        type_collector: Dict[int, CType],
        id: int,
        tag_type: str,
        attributes: Dict[str, Any],
    ):
        """
        :param type_collector: A dictionary of ID to CType mappings, to which this CType is added
        :param id: numeric identifier (the offset of the DWARF entry)
        :param tag_type: kind of entry
        :param attributes: attribute values (see `Attributes`)
        """
        if id in type_collector:
            raise ValueError(
                f"Attempted to reuse ID '{id}' from existing CType '{type_collector[id]}'"
            )
        self.collector = type_collector
        self.id = id
        self.tag_type = tag_type
        self.attributes = Attributes(attributes)
        self.members: List[int] = []
        type_collector[id] = self

    def base_type(self) -> CType:
        """
        Find the underlying base type of a CType.

        Returns self for all CTypes with types not in "typedef_tag",
        "member_tag" or "array_tag".
        """
        if self.tag_type not in ("typedef_tag", "member_tag", "array_tag"):
            return self

        base_type = self.collector[self.attributes.type[0]]
        while base_type.tag_type == "typedef_tag":
            base_type = self.collector[base_type.attributes.type[0]]

        if base_type.tag_type in ("unknown_tag", *self.member_tags):
            raise ValueError(
                f"CType '{self}' resolved to base type '{base_type}' of invalid tag type '{base_type.tag_type}'"
            )
        return base_type

    def array_dimensions(self) -> Tuple[int, ...]:
        """
        Return the array's dimensions in DWARF/source order.

        A dimension is 0 when its subrange does not have a known
        bound. C arrays have an implicit lower bound of zero, so a
        DW_AT_upper_bound of N describes N + 1 entries.
        """
        if self.tag_type != "array_tag":
            raise TypeError(
                f"Provided CType tag '{self.tag_type}' is not an 'array_tag'"
            )

        dimensions: List[int] = []
        for member_id in self.members:
            subrange_type = self.collector[member_id]
            if subrange_type.attributes.count is not None:
                dimensions.append(subrange_type.attributes.count)
            elif subrange_type.attributes.upper_bound is not None:
                dimensions.append(subrange_type.attributes.upper_bound + 1)
            else:
                dimensions.append(0)
        return tuple(dimensions)

    def array_count(self) -> int:
        """
        Find the number of entries in an "array_tag" CType. Returns 0 if the
        subrange type does not have a count attribute.
        """
        dimensions = self.array_dimensions()
        if not dimensions:
            return 0

        count = 1
        for dimension in dimensions:
            count *= dimension
        return count

    def __repr__(self):
        return f"CType:\n    ID: {self.id}\n    type: {self.tag_type}\n    members: {self.members}\n{self.attributes!r}"


class DwarfTypes:
    """
    The types described by the DWARF of an ELF file.
    """

    def __init__(self, path: pathlib.Path):
        self.path = path
        self._file: BinaryIO = open(path, "rb")
        self.elf = ELFFile(self._file)
        if not self.elf.has_dwarf_info():
            raise RuntimeError(f"{path} has no DWARF information!")
        self.dwarf = self.elf.get_dwarf_info()
        self.collector: Dict[int, CType] = {}

    def _dies(self) -> Iterator[DIE]:
        for cu in self.dwarf.iter_CUs():
            yield from cu.iter_DIEs()

    def type_for_section(self, section_name: str, type_name: str) -> Optional[CType]:
        """
        Find the type of the variable placed in `section_name`, if there is one.

        Single-member wrappers around the configuration structure (such as those used to place it
        in a section from Rust) are unwrapped, stopping at a type named `type_name`.
        """
        section = self.elf.get_section_by_name("." + section_name)
        if section is None:
            return None
        start = section["sh_addr"]
        end = start + section["sh_size"]

        variable = None
        for die in self._dies():
            if die.tag == "DW_TAG_variable":
                addr = self._variable_address(die)
                if addr is not None and start <= addr < end:
                    variable = die
                    break
        if variable is None:
            return None

        c_type = self._ctype(variable.get_DIE_from_attribute("DW_AT_type")).base_type()
        while c_type.attributes.name != type_name and len(c_type.members) == 1:
            member = self.collector[c_type.members[0]]
            inner = member.base_type()
            if (
                c_type.tag_type not in ("structure_tag", "union_tag")
                or member.attributes.data_member_location != 0
                or inner.attributes.byte_size != c_type.attributes.byte_size
            ):
                break
            c_type = inner
        return c_type

    def type_by_name(self, type_name: str) -> CType:
        """
        Find a type by name.
        """
        for die in self._dies():
            if die.tag in _TAG_TYPES and "DW_AT_declaration" not in die.attributes:
                name = die.attributes.get("DW_AT_name")
                if name is not None and name.value.decode() == type_name:
                    return self._ctype(die).base_type()
        raise ValueError(f"Could not find a type with name '{type_name}' in {self.path}")

    def _variable_address(self, die: DIE) -> Optional[int]:
        location = die.attributes.get("DW_AT_location")
        if location is None or location.form not in ("DW_FORM_exprloc", "DW_FORM_block1"):
            return None
        ops = DWARFExprParser(die.dwarfinfo.structs).parse_expr(location.value)
        if len(ops) == 1 and ops[0].op_name == "DW_OP_addr":
            return ops[0].args[0]
        return None

    def _ctype(self, die: DIE) -> CType:
        if die.offset in self.collector:
            return self.collector[die.offset]

        tag_type = _TAG_TYPES.get(die.tag, "unknown_tag")
        attributes: Dict[str, Any] = {}
        refs: List[Tuple[str, DIE]] = []

        # Only these attributes are relevant to laying out config structs. Others, such as
        # DW_AT_alignment (which rustc emits for every member), are ignored.
        for name, attr in die.attributes.items():
            match name:
                case "DW_AT_name":
                    attributes["name"] = attr.value.decode()
                case "DW_AT_byte_size" if tag_type != "pointer_tag":
                    # Pointers are assumed to be 8 bytes
                    attributes["byte_size"] = attr.value
                case "DW_AT_encoding":
                    attributes["encoding"] = _ATE_NAMES[attr.value]
                case "DW_AT_const_value":
                    attributes["const_value"] = attr.value
                case "DW_AT_data_member_location":
                    attributes["data_member_location"] = self._member_location(die, attr)
                case "DW_AT_upper_bound" | "DW_AT_count" | "DW_AT_bit_size" | "DW_AT_data_bit_offset":
                    attributes[name.removeprefix("DW_AT_")] = attr.value
                case "DW_AT_type":
                    refs.append(("type", die.get_DIE_from_attribute(name)))

        for name, target in refs:
            target_name = target.attributes.get("DW_AT_name")
            attributes[name] = (
                target.offset,
                target_name.value.decode() if target_name is not None else "",
            )

        c_type = CType(self.collector, die.offset, tag_type, attributes)

        # Resolve referenced types eagerly (except through pointers, whose targets are not needed
        # for layout), so that CType.base_type() can follow them.
        if tag_type != "pointer_tag":
            for _, target in refs:
                self._ctype(target)

        for child in die.iter_children():
            if child.tag in _MEMBER_TAGS.get(tag_type, ()):
                c_type.members.append(self._ctype(child).id)

        return c_type

    @staticmethod
    def _member_location(die: DIE, attr) -> int:
        if isinstance(attr.value, int):
            return attr.value
        # DWARF 2 style: a location expression
        ops = DWARFExprParser(die.dwarfinfo.structs).parse_expr(attr.value)
        if len(ops) == 1 and ops[0].op_name == "DW_OP_plus_uconst":
            return ops[0].args[0]
        raise ValueError(f"Unsupported member location {describe_attr_value(attr, die, 0)}")
