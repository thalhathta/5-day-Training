from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from dataclasses import dataclass
from typing import Optional


# -----------------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------------
# json_response=True enables structuredContent in tool outputs (useful for clients)
mcp = FastMCP("campus_tools_mcp", json_response=True)

DATA_PATH = Path(__file__).with_name("campus_data.json")
DATA = json.loads(DATA_PATH.read_text())


# -----------------------------------------------------------------------------
# Typed records (schema-friendly)
# -----------------------------------------------------------------------------

@dataclass
class ErrorResult:
    error: str

@dataclass
class OfficeHoursResponse:
    staff_name: str
    office_hours: list[OfficeHours]


@dataclass
class StaffRecord:
    name: str
    department: str
    email: str
    office: str


@dataclass
class OfficeHours:
    staff_name: str
    day: str
    time: str
    mode: str
    location: str


@dataclass
class TimetableSlot:
    course_code: str
    week: int
    day: str
    time: str
    venue: str


@dataclass
class RoomInfo:
    building: str
    room: str
    capacity: int
    facilities: list[str]


@dataclass
class Contact:
    contact_type: str
    name: str
    email: str
    phone: str


def _norm(s: str) -> str:
    return s.strip().lower()


@mcp.tool()
def find_staff(query: str, department: Optional[str] = None) -> list[dict]:
    """Find staff members by name fragment, optionally filtered by department."""
    q = _norm(query)
    dept = _norm(department) if department else None

    results: list[StaffRecord] = []
    for s in DATA["staff"]:
        if q in _norm(s["name"]) and (dept is None or dept == _norm(s["department"])):
            results.append(StaffRecord(**s))

    return [asdict(r) for r in results]

@mcp.tool()
def get_office_hours(staff_name: str) -> list[dict]:
    target = _norm(staff_name)
    matches: list[OfficeHours] = []
    for oh in DATA["office_hours"]:
        if _norm(oh["staff_name"]) == target:
            matches.append(OfficeHours(**oh))

    if not matches:
        return [{"error": f"No office hours found for: {staff_name}"}]

    return [{
        "staff_name": staff_name,
        "office_hours": [asdict(m) for m in matches]
    }]


@mcp.tool()
def find_timetable(course_code: str, week: Optional[int] = None) -> list[dict]:
    """Find timetable slots by course code, optionally filtered by week."""
    cc = _norm(course_code)

    if week is not None and int(week) <= 0:
        return [{"error": "week must be a positive integer"}]

    results: list[TimetableSlot] = []
    for t in DATA["timetable"]:
        if _norm(t["course_code"]) == cc and (week is None or int(t["week"]) == int(week)):
            results.append(TimetableSlot(**t))

    return [asdict(r) for r in results]


@mcp.tool()
def find_room(building: str, room: str) -> list[dict]:
    b = _norm(building)
    r = _norm(room)

    for item in DATA["rooms"]:
        if _norm(item["building"]) == b and _norm(item["room"]) == r:
            return [asdict(RoomInfo(**item))]

    return [{"error": f"Room not found: building={building}, room={room}"}]

@mcp.tool()
def list_contacts(contact_type: str) -> list[dict]:
    """List contacts by type."""
    ct = _norm(contact_type)
    results: list[Contact] = []
    for c in DATA["contacts"]:
        if _norm(c["contact_type"]) == ct:
            results.append(Contact(**c))

    return [asdict(r) for r in results]


if __name__ == "__main__":
    # Default transport is stdio for FastMCP servers.
    mcp.run()
