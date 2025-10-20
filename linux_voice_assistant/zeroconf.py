"""Runs mDNS zeroconf service for Home Assistant discovery."""

import asyncio
import logging
import socket
import uuid
from typing import Optional

from zeroconf import IPVersion, ServiceInfo, Zeroconf

_LOGGER = logging.getLogger(__name__)

MDNS_TARGET_IP = "224.0.0.251"


class HomeAssistantZeroconf:
    def __init__(
        self, port: int, name: Optional[str] = None, host: Optional[str] = None
    ) -> None:
        self.port = port
        self.name = name or _get_mac_address()

        if not host:
            test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            test_sock.setblocking(False)
            test_sock.connect((MDNS_TARGET_IP, 1))
            host = test_sock.getsockname()[0]
            _LOGGER.debug("Detected IP: %s", host)

        assert host
        self.host = host
        self._zeroconf: Optional[Zeroconf] = None
        self._service_info: Optional[ServiceInfo] = None

    async def _ensure_zeroconf(self) -> Zeroconf:
        if self._zeroconf is None:
            self._zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
        return self._zeroconf

    async def register_server(self) -> None:
        zeroconf = await self._ensure_zeroconf()

        service_info = ServiceInfo(
            "_esphomelib._tcp.local.",
            f"{self.name}._esphomelib._tcp.local.",
            addresses=[socket.inet_aton(self.host)],
            port=self.port,
            properties={
                b"version": b"2025.9.0",
                b"mac": _get_mac_address().encode("ascii"),
                b"board": b"host",
                b"platform": b"HOST",
                b"network": b"ethernet",  # or b"wifi"
            },
            server=f"{self.name}.local.",
        )

        await asyncio.to_thread(zeroconf.register_service, service_info, allow_name_change=False)
        self._service_info = service_info
        _LOGGER.debug("Zeroconf discovery enabled: %s", service_info)

    async def unregister_server(self) -> None:
        if not self._service_info or not self._zeroconf:
            return

        service_info = self._service_info
        zeroconf = self._zeroconf
        self._service_info = None

        try:
            await asyncio.to_thread(zeroconf.unregister_service, service_info)
            _LOGGER.debug("Zeroconf discovery disabled: %s", service_info.name)
        except Exception:
            _LOGGER.exception("Failed to unregister zeroconf service")

    async def shutdown(self) -> None:
        await self.unregister_server()

        if self._zeroconf is None:
            return

        zeroconf = self._zeroconf
        self._zeroconf = None
        await asyncio.to_thread(zeroconf.close)

        _LOGGER.debug("Zeroconf closed")


def _get_mac_address() -> str:
    """Return MAC address formatted as hex with no colons."""
    return "".join(
        # pylint: disable=consider-using-f-string
        ["{:02x}".format((uuid.getnode() >> ele) & 0xFF) for ele in range(0, 8 * 6, 8)][
            ::-1
        ]
    )
