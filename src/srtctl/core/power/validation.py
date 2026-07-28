# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publication gates over the persisted artifact contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from srtctl.core.power.contract import Reason, dedupe
from srtctl.core.power.samples import ObservedDevice
from srtctl.core.power.topology import DeviceKey, ExpectedDevice, resolve_het_groups, resolve_roles


@dataclass(frozen=True)
class DeviceValidation:
    """Whether device identity and topology permit publication."""

    valid: bool
    reason_codes: tuple[str, ...]
    roles: dict[DeviceKey, str]
    het_groups: dict[str, int | None]


def validate_devices(
    expected: Sequence[ExpectedDevice],
    observed: Sequence[ObservedDevice],
) -> DeviceValidation:
    """Require a non-empty expected set that exactly matches stable observations."""
    reasons: list[str] = []

    expected_keys = {device.key for device in expected}
    observed_keys = {device.key for device in observed}

    if not expected_keys or expected_keys - observed_keys:
        reasons.append(Reason.EXPECTED_DEVICE_MISSING)
    if observed_keys - expected_keys:
        reasons.append(Reason.UNEXPECTED_DEVICE)
    # Note (wenyao): a UUID must map 1:1 to a device key, or one physical GPU is counted twice.
    if any(len(device.gpu_uuids) != 1 for device in observed):
        reasons.append(Reason.GPU_UUID_CHANGED)
    else:
        uuids = [device.gpu_uuids[0] for device in observed]
        if len(set(uuids)) != len(uuids):
            reasons.append(Reason.GPU_UUID_CHANGED)

    roles, role_conflicts = resolve_roles(expected)
    het_groups, group_conflicts = resolve_het_groups(expected)
    reasons.extend(role_conflicts)
    reasons.extend(group_conflicts)

    return DeviceValidation(
        valid=not reasons,
        reason_codes=dedupe(reasons),
        roles=roles,
        het_groups=het_groups,
    )
